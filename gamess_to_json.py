#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gamess_to_json.py

GAMESS の出力ファイル (例: benzene.out) から平衡構造（最適化された分子構造）の
原子座標を抽出し、JSON 形式に変換するプログラム。

最終目標は、この JSON を読み込んでファンデルワールス球モデルを原子ごとに
分割した 3D プリント用 STL を生成することなので、座標に加えて各原子の
ファンデルワールス半径（Angstrom）も出力に含めている。

使い方:
    python gamess_to_json.py benzene.out
    python gamess_to_json.py benzene.out -o benzene.json
    python gamess_to_json.py benzene.out --geometry last   # 最後の幾何ステップを使う
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


# ---------------------------------------------------------------------------
# 元素データ
#   - 原子番号 -> 元素記号（GAMESS は CHARGE 列に核電荷を整数値で出力する）
#   - 元素記号 -> ファンデルワールス半径 [Angstrom]
#     (A. Bondi, J. Phys. Chem. 1964, 68, 441 などの一般的な値)
# ---------------------------------------------------------------------------
ATOMIC_NUMBER_TO_SYMBOL = {
    1: "H", 2: "He",
    3: "Li", 4: "Be", 5: "B", 6: "C", 7: "N", 8: "O", 9: "F", 10: "Ne",
    11: "Na", 12: "Mg", 13: "Al", 14: "Si", 15: "P", 16: "S", 17: "Cl", 18: "Ar",
    19: "K", 20: "Ca", 26: "Fe", 29: "Cu", 30: "Zn", 35: "Br", 53: "I",
}

# ファンデルワールス半径 [Angstrom]。未登録の元素は None になる。
VDW_RADII = {
    "H": 1.20, "He": 1.40,
    "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52,
    "F": 1.47, "Ne": 1.54,
    "Na": 2.27, "Mg": 1.73, "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80,
    "Cl": 1.75, "Ar": 1.88,
    "K": 2.75, "Ca": 2.31, "Fe": 2.04, "Cu": 1.40, "Zn": 1.39,
    "Br": 1.85, "I": 1.98,
}


@dataclass
class Atom:
    """1 つの原子の情報。"""
    index: int            # 0 始まりの通し番号
    element: str          # 元素記号 (例: "C")
    atomic_number: int    # 原子番号
    x: float              # 座標 [Angstrom]
    y: float
    z: float
    vdw_radius: float | None  # ファンデルワールス半径 [Angstrom] (未知なら None)


# GAMESS の座標行:
#  C           6.0   0.0036153940  -0.0000076958   0.0000891057
_ATOM_LINE = re.compile(
    r"""^\s*
        ([A-Za-z]{1,2})\s+        # 元素記号
        (\d+(?:\.\d+)?)\s+        # 核電荷 (CHARGE)
        ([-+]?\d+\.\d+)\s+        # X
        ([-+]?\d+\.\d+)\s+        # Y
        ([-+]?\d+\.\d+)\s*$       # Z
    """,
    re.VERBOSE,
)

# 座標ブロックの開始を示すヘッダ
_COORD_HEADER = "COORDINATES OF ALL ATOMS ARE (ANGS)"
_EQUILIBRIUM_MARKER = "EQUILIBRIUM GEOMETRY LOCATED"


def _parse_atom_block(lines: list[str], start: int) -> list[Atom]:
    """
    座標ブロックのヘッダ位置 (start = "COORDINATES OF ALL ATOMS ARE (ANGS)" の行番号)
    から原子行を読み取り、Atom のリストを返す。

    ブロックは次のような構造:
        COORDINATES OF ALL ATOMS ARE (ANGS)
          ATOM   CHARGE       X              Y              Z
        ------------------------------------------------------------
        C           6.0   ...   ...   ...
        H           1.0   ...   ...   ...
        （空行 or 数値でない行で終了）
    """
    atoms: list[Atom] = []
    i = start + 1
    # "ATOM CHARGE X Y Z" のヘッダ行と区切り線 "----" を読み飛ばす
    while i < len(lines):
        stripped = lines[i].strip()
        if stripped.startswith("ATOM") or set(stripped) <= {"-"} and stripped:
            i += 1
            continue
        if stripped == "":
            i += 1
            # 区切り線直後の空行は許容するが、原子行がまだなら継続
            if not atoms:
                continue
            break
        m = _ATOM_LINE.match(lines[i])
        if not m:
            break
        symbol_raw, charge_raw, x, y, z = m.groups()
        atomic_number = int(round(float(charge_raw)))
        element = ATOMIC_NUMBER_TO_SYMBOL.get(atomic_number, symbol_raw.capitalize())
        atoms.append(
            Atom(
                index=len(atoms),
                element=element,
                atomic_number=atomic_number,
                x=float(x),
                y=float(y),
                z=float(z),
                vdw_radius=VDW_RADII.get(element),
            )
        )
        i += 1
    return atoms


def parse_gamess(text: str, geometry: str = "equilibrium") -> tuple[list[Atom], str]:
    """
    GAMESS 出力テキストから原子座標を抽出する。

    geometry:
        "equilibrium" : "EQUILIBRIUM GEOMETRY LOCATED" 直後の座標ブロックを使う。
                        見つからなければ最後の座標ブロックにフォールバック。
        "last"        : ファイル中で最後に現れる座標ブロックを使う。
        "first"       : 最初の座標ブロック（初期構造）を使う。

    戻り値: (atoms, 使用したブロックのラベル)
    """
    lines = text.splitlines()

    # すべての座標ブロックのヘッダ位置を収集
    coord_headers = [i for i, ln in enumerate(lines) if _COORD_HEADER in ln]
    if not coord_headers:
        raise ValueError(
            "座標ブロック ('COORDINATES OF ALL ATOMS ARE (ANGS)') が見つかりません。"
        )

    if geometry == "equilibrium":
        eq_line = next(
            (i for i, ln in enumerate(lines) if _EQUILIBRIUM_MARKER in ln), None
        )
        if eq_line is not None:
            # マーカー以降で最初の座標ヘッダ
            header = next((h for h in coord_headers if h >= eq_line), None)
            if header is not None:
                return _parse_atom_block(lines, header), "EQUILIBRIUM GEOMETRY"
        # フォールバック
        return _parse_atom_block(lines, coord_headers[-1]), "LAST GEOMETRY (fallback)"
    elif geometry == "last":
        return _parse_atom_block(lines, coord_headers[-1]), "LAST GEOMETRY"
    elif geometry == "first":
        return _parse_atom_block(lines, coord_headers[0]), "FIRST GEOMETRY"
    else:
        raise ValueError(f"未知の geometry 指定: {geometry!r}")


def build_document(atoms: list[Atom], source: str, label: str) -> dict:
    """JSON 出力用の辞書を組み立てる。"""
    return {
        "source": source,
        "geometry": label,
        "units": "angstrom",
        "atom_count": len(atoms),
        "atoms": [asdict(a) for a in atoms],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="GAMESS 出力ファイルから平衡構造の原子座標を JSON に変換する。"
    )
    parser.add_argument("input", help="GAMESS 出力ファイル (例: benzene.out)")
    parser.add_argument(
        "-o", "--output",
        help="出力 JSON ファイル名 (省略時は入力名の拡張子を .json に変えたもの)",
    )
    parser.add_argument(
        "--geometry",
        choices=["equilibrium", "last", "first"],
        default="equilibrium",
        help="抽出する構造 (既定: equilibrium)",
    )
    parser.add_argument(
        "--stdout", action="store_true",
        help="ファイルに書かず標準出力に JSON を出す",
    )
    args = parser.parse_args(argv)

    in_path = Path(args.input)
    if not in_path.is_file():
        print(f"エラー: 入力ファイルが見つかりません: {in_path}", file=sys.stderr)
        return 1

    text = in_path.read_text(encoding="utf-8", errors="replace")

    try:
        atoms, label = parse_gamess(text, geometry=args.geometry)
    except ValueError as e:
        print(f"エラー: {e}", file=sys.stderr)
        return 1

    if not atoms:
        print("エラー: 原子が 1 つも抽出できませんでした。", file=sys.stderr)
        return 1

    doc = build_document(atoms, source=in_path.name, label=label)
    json_text = json.dumps(doc, indent=2, ensure_ascii=False)

    if args.stdout:
        print(json_text)
    else:
        out_path = Path(args.output) if args.output else in_path.with_suffix(".json")
        out_path.write_text(json_text + "\n", encoding="utf-8")
        print(f"{label}: {len(atoms)} 原子を {out_path} に書き出しました。")
        # 欠落しているファンデルワールス半径があれば警告
        missing = sorted({a.element for a in atoms if a.vdw_radius is None})
        if missing:
            print(
                f"  警告: ファンデルワールス半径が未登録の元素: {', '.join(missing)}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
