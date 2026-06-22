#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vdw_to_stl.py

gamess_to_json.py が生成した JSON（原子座標・元素・ファンデルワールス半径）を
読み込み、ファンデルワールス球モデルを「原子ごとに分割」した STL を生成する。

== 分割の考え方 ==
各原子は「原子核座標を中心とする半径 = vdW 半径の球」。球が重なる相手ごとに、
2 球の交線が乗る平面 (radical plane) で球を切り、相手側の出っ張りを削る。
radical plane は半径差を反映し、原子 A の中心からの距離は
    dA = (d^2 + rA^2 - rB^2) / (2 d)   (d = 中心間距離)
となる。重なる全ペアを切るので、空間は隙間なく分割され（パワー・ダイアグラム）、
各ピースは互いに貫通せずぴたりと組み合う。各原子は独立に処理し、平面は常に
「その原子の球」にしか作用しないので二重カットは起きない。

== 組み立て補助の刻印 (重原子-重原子結合の面のみ) ==
水素を含まない化学結合 (C-C, C-N, ...) の切断面には、結合を一意に識別し、かつ
二面角を決めるための刻印を施す。各結合に 0..255 の ID を割り当て、
    上位ニブル (0..15) … 4 ビットの「凹みドット」で表現
    下位ニブル (0..15) … リムの「大小 2 本の切り欠き」の角度で表現
両者で 16 x 16 = 256 結合まで一意に識別できる。

  ・凹みドット: 切断面に 4 個のビット穴 (有=1)。左右(読み向き)が分かるよう
    左端に開始マーカー、下側に上下識別ラインを付けた非対称配置。
  ・切り欠き: リム(切断面の縁)に結合軸と平行な溝を 2 本。
      大きい溝 = 二面角の基準 (両面で同じ world 方向 = 参照方向)。
      小さい溝 = 値 (基準からの角度 = 下位ニブル)。
    radical plane の性質上、結合する 2 原子の切断面は同一平面・同一の交円を
    共有するので、両ピースの溝は同じ world 位置に刻まれる。組み立て時に横から
    見て大小の溝を相手と合わせれば、二面角が一意に決まる。

使い方:
    python vdw_to_stl.py benzene.json
    python vdw_to_stl.py benzene.json --scale 15 --subdivisions 4 --combined
    python vdw_to_stl.py benzene.json --no-marks      # 刻印なし(分割のみ)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh


# 共有結合半径 [Angstrom] (Cordero et al. 2008)。化学結合の判定に使う。
COVALENT_RADII = {
    "H": 0.31, "He": 0.28,
    "Li": 1.28, "Be": 0.96, "B": 0.84, "C": 0.76, "N": 0.71, "O": 0.66,
    "F": 0.57, "Ne": 0.58,
    "Na": 1.66, "Mg": 1.41, "Al": 1.21, "Si": 1.11, "P": 1.07, "S": 1.05,
    "Cl": 1.02, "Ar": 1.06,
    "K": 2.03, "Ca": 1.76, "Fe": 1.32, "Cu": 1.32, "Zn": 1.22,
    "Br": 1.20, "I": 1.39,
}
BOND_TOLERANCE = 0.45  # |結合| <= rcovA + rcovB + tol を結合とみなす [Angstrom]

DEG = np.pi / 180.0


# ---------------------------------------------------------------------------
# 入力
# ---------------------------------------------------------------------------
def load_atoms(json_path: Path):
    """JSON を読み込み (positions[N,3], radii[N], elements[N]) を返す。"""
    doc = json.loads(json_path.read_text(encoding="utf-8"))
    atoms = doc["atoms"]
    positions = np.array([[a["x"], a["y"], a["z"]] for a in atoms], dtype=float)
    radii = np.array(
        [a["vdw_radius"] if a["vdw_radius"] is not None else np.nan for a in atoms],
        dtype=float,
    )
    elements = [a["element"] for a in atoms]
    if np.isnan(radii).any():
        missing = sorted({elements[i] for i in np.where(np.isnan(radii))[0]})
        raise ValueError(
            f"ファンデルワールス半径が未設定の元素があります: {', '.join(missing)}"
        )
    return positions, radii, elements


# ---------------------------------------------------------------------------
# 幾何ユーティリティ
# ---------------------------------------------------------------------------
def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def radical_plane(center_a, ra, center_b, rb):
    """
    2 球の交線が乗る平面を返す: (origin, normal, da, rho)
        origin : 平面上の点 (a から b 方向へ da)
        normal : a 側を向いた単位法線 (slice_mesh_plane は法線側を残す)
        da     : a の中心から平面までの距離
        rho    : 交円の半径
    切る必要がない / 切れない場合は None。
    """
    ab = center_b - center_a
    d = float(np.linalg.norm(ab))
    if d <= 1e-9 or d >= ra + rb or d <= abs(ra - rb):
        return None
    u = ab / d
    da = (d * d + ra * ra - rb * rb) / (2.0 * d)
    rho = float(np.sqrt(max(ra * ra - da * da, 0.0)))
    origin = center_a + da * u
    return origin, -u, da, rho


def halfspace_box(origin, normal, size):
    """平面 (origin, normal) の normal 側を埋める十分大きな直方体。"""
    normal = np.asarray(normal, dtype=float)
    T = trimesh.geometry.align_vectors([0, 0, 1], normal)
    box = trimesh.creation.box(extents=[size, size, size])
    box.apply_transform(T)
    box.apply_translation(np.asarray(origin, dtype=float) + normal * (size / 2.0))
    return box


def detect_bonds(positions, radii_cov):
    """化学結合の (i,j) ペア一覧を返す (i<j)。距離 <= rcovi+rcovj+tol。"""
    n = len(positions)
    bonds = []
    for i in range(n):
        for j in range(i + 1, n):
            d = np.linalg.norm(positions[i] - positions[j])
            if d <= radii_cov[i] + radii_cov[j] + BOND_TOLERANCE:
                bonds.append((i, j))
    return bonds


# ---------------------------------------------------------------------------
# 刻印フィーチャ (差し引く小立体を生成)
# ---------------------------------------------------------------------------
def _box_on_face(center, sx, sy, depth, right, up, normal, over=0.5):
    """
    切断面上の点 center を中心に、面に沿って sx*sy、面の法線方向(内側)へ depth の
    直方体を作る。over だけ表面から外に出して確実に差し引けるようにする。
    """
    right, up, normal = _unit(right), _unit(up), _unit(normal)
    T = np.eye(4)
    T[:3, 0] = right
    T[:3, 1] = up
    T[:3, 2] = normal
    h = depth + over
    # 面より over だけ外、内側へ depth。中心は法線方向に (over-depth)/2。
    T[:3, 3] = np.asarray(center) + normal * ((over - depth) / 2.0)
    return trimesh.creation.box(extents=[sx, sy, h], transform=T)


def _cyl(p0, p1, radius):
    return trimesh.creation.cylinder(radius=radius, segment=[p0, p1], sections=20)


def build_identifier_tools(center, normal, ref, rho, value):
    """
    切断面 (中心 center, 外向き法線 normal, 参照方向 ref, 半径 rho) に、
    4 ビット値 value を表す凹みドット群＋開始マーカー＋上下ラインを作る。
    読み枠: up = ref, right = ref x normal (面を外から見て左に MSB)。
    """
    up = _unit(ref)
    right = _unit(np.cross(ref, normal))  # 面を外側から見たときの右方向

    spacing = rho * 0.30
    dot_r = float(np.clip(rho * 0.10, 0.6, 1.6))
    depth = 1.2
    tools = []

    # 4 ビットドット (k=0 が MSB, 左端)
    for k in range(4):
        if (value >> (3 - k)) & 1:
            x = (k - 1.5) * spacing
            p = center + x * right
            tools.append(_cyl(p + normal * 0.6, p - normal * depth, dot_r))

    # 開始マーカー: 左端に縦長バー (丸ドットと形が違う → 左右が分かる)
    xm = -2.2 * spacing
    pm = center + xm * right
    tools.append(_box_on_face(pm, dot_r * 0.9, spacing * 1.3, depth, right, up, normal))

    # 上下識別ライン: ドット列の下に水平な溝
    pl = center - up * (spacing * 0.95)
    tools.append(_box_on_face(pl, spacing * 3.4, dot_r * 0.8, depth * 0.8,
                              right, up, normal))
    return tools


def build_notch_tools(center, axis, ref, rho, r_atom, value):
    """
    リム(縁)に大小 2 本の切り欠き(結合軸 axis と平行な溝)を作る。
        大きい溝 : 参照方向から -11.25 度 (二面角の基準)
        小さい溝 : 参照方向から value * 22.5 度 (下位ニブル)
    axis は結合の向き (i->j) で、結合する両原子で共通に使うこと。
    """
    e_r = _unit(ref)
    e_t = _unit(np.cross(axis, ref))  # axis 周りの接線方向

    def direction(phi):
        return np.cos(phi) * e_r + np.sin(phi) * e_t

    hl = 0.05 * r_atom  # 溝(円柱)の半長。リム付近だけを削れば十分。
    tools = []
    for phi, w in ((-11.25 * DEG, rho * 0.10), (value * 22.5 * DEG, rho * 0.05)):
        p = center + rho * direction(phi)
        tools.append(_cyl(p - axis * hl, p + axis * hl, w))
    return tools


# ---------------------------------------------------------------------------
# 原子メッシュ生成
# ---------------------------------------------------------------------------
def build_atom_mesh(i, positions, radii, subdivisions, bond_marks):
    """
    原子 i の分割済みメッシュを生成。

    bond_marks: dict[(min,max)] -> (axis_ij, value)
        刻印対象の結合のみを含む。axis_ij は i->j 向き(=結合の正方向)。
        value は ID。i がその結合に関与していれば、面に刻印を施す。
    """
    center = positions[i]
    r = radii[i]
    sphere = trimesh.creation.icosphere(subdivisions=subdivisions, radius=r)
    sphere.apply_translation(center)

    box_size = 4.0 * r
    solids = [sphere]
    for j in range(len(positions)):
        if j == i:
            continue
        plane = radical_plane(center, r, positions[j], radii[j])
        if plane is None:
            continue
        origin, normal, _, _ = plane
        solids.append(halfspace_box(origin, normal, box_size))

    mesh = sphere if len(solids) == 1 else trimesh.boolean.intersection(
        solids, engine="manifold"
    )
    if mesh is None or len(mesh.faces) == 0:
        return None

    # --- 刻印 (重原子-重原子結合の面のみ) ---
    tools = []
    for j in range(len(positions)):
        key = (min(i, j), max(i, j))
        if key not in bond_marks:
            continue
        plane = radical_plane(center, r, positions[j], radii[j])
        if plane is None:
            continue
        origin, normal, _, rho = plane
        # normal は i 側を向く単位法線。外向き(相手側)はその逆。
        outward = -normal
        axis_ij, value = bond_marks[key]
        # 参照方向: 結合軸に直交する安定な world ベクトル(両原子で共通)
        ref = _reference_dir(axis_ij)
        tools += build_identifier_tools(origin, outward, ref, rho, value)
        tools += build_notch_tools(origin, axis_ij, ref, rho, r, value)

    if tools:
        cutter = tools[0] if len(tools) == 1 else trimesh.boolean.union(
            tools, engine="manifold"
        )
        engraved = trimesh.boolean.difference([mesh, cutter], engine="manifold")
        if engraved is not None and len(engraved.faces) > 0:
            mesh = engraved
    return mesh


def _reference_dir(axis):
    """結合軸 axis に直交する基準方向 (world)。両原子で同じ結果になる。"""
    axis = _unit(axis)
    up = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(axis, up)) > 0.9:
        up = np.array([1.0, 0.0, 0.0])
    return _unit(up - np.dot(up, axis) * axis)


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------
def main(argv=None):
    parser = argparse.ArgumentParser(
        description="原子座標 JSON から、識別刻印付きで原子ごとに分割した vdW STL を生成する。"
    )
    parser.add_argument("input", help="gamess_to_json.py が出力した JSON ファイル")
    parser.add_argument("-o", "--output-dir", default=None,
                        help="STL 出力ディレクトリ (既定: <入力名>_stl)")
    parser.add_argument("--scale", type=float, default=10.0,
                        help="Angstrom -> mm のスケール係数 (既定: 10.0)")
    parser.add_argument("--subdivisions", type=int, default=3,
                        help="球メッシュの細分化レベル (既定: 3)")
    parser.add_argument("--combined", action="store_true",
                        help="確認用に全原子を結合した STL も出力")
    parser.add_argument("--no-marks", action="store_true",
                        help="識別刻印・切り欠きを付けない (分割のみ)")
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"エラー: 入力ファイルが見つかりません: {in_path}", file=sys.stderr)
        return 1

    try:
        positions, radii, elements = load_atoms(in_path)
    except (KeyError, ValueError, json.JSONDecodeError) as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    positions = positions * args.scale
    radii = radii * args.scale
    n = len(positions)

    # 化学結合検出 -> 重原子-重原子結合に ID を割り当て
    bond_marks = {}
    if not args.no_marks:
        radii_cov = np.array([COVALENT_RADII.get(e, 0.8) for e in elements])
        bonds = detect_bonds(positions / args.scale, radii_cov)  # 共有半径は Angstrom
        heavy = [(i, j) for (i, j) in bonds
                 if elements[i] != "H" and elements[j] != "H"]
        if len(heavy) > 256:
            print(f"警告: 重原子-重原子結合が {len(heavy)} 本あり 256 を超えます。"
                  "ID が一意になりません。", file=sys.stderr)
        for bid, (i, j) in enumerate(heavy):
            axis_ij = _unit(positions[j] - positions[i])
            bond_marks[(i, j)] = (axis_ij, bid)
        print(f"刻印対象の重原子-重原子結合: {len(heavy)} 本 "
              f"(全化学結合 {len(bonds)} 本中)")

    out_dir = Path(args.output_dir) if args.output_dir else in_path.with_name(
        in_path.stem + "_stl")
    out_dir.mkdir(parents=True, exist_ok=True)
    base = in_path.stem
    combined_parts = []
    print(f"{n} 原子を処理 (scale={args.scale} mm/A, subdivisions={args.subdivisions})")

    for i in range(n):
        mesh = build_atom_mesh(i, positions, radii, args.subdivisions, bond_marks)
        name = f"{base}_{i:02d}_{elements[i]}.stl"
        if mesh is None or len(mesh.faces) == 0:
            print(f"  [{i:2d}] {elements[i]}: 立体が生成されませんでした", file=sys.stderr)
            continue
        mesh.export(out_dir / name)
        wt = "watertight" if mesh.is_watertight else "NOT watertight(!)"
        print(f"  [{i:2d}] {elements[i]}: faces={len(mesh.faces):5d} "
              f"vol={mesh.volume:8.1f} {wt} -> {name}")
        if args.combined:
            combined_parts.append(mesh)

    if args.combined and combined_parts:
        trimesh.util.concatenate(combined_parts).export(out_dir / f"{base}_combined.stl")
        print(f"結合モデル -> {base}_combined.stl")

    print(f"出力先: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
