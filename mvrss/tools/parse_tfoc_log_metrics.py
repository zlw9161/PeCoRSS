"""Parse training logs for Val/Test Dice lines.

Usage:
  python tools/parse_tfoc_log_metrics.py "C:\\path\\to\\log.txt"
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python tools/parse_tfoc_log_metrics.py <log_path>")
        return 2
    p = Path(sys.argv[1])
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()

    val_pat = re.compile(r"\[Epoch (\d+)/\d+\] Val Dice: RD=([\d.]+), RA=([\d.]+)")
    test_pat = re.compile(r"\[Epoch (\d+)/\d+\] Test Dice: RD=([\d.]+), RA=([\d.]+)")

    vals: list[tuple[int, float, float]] = []
    tests: list[tuple[int, float, float]] = []
    for line in lines:
        m = val_pat.search(line)
        if m:
            vals.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))
        m = test_pat.search(line)
        if m:
            tests.append((int(m.group(1)), float(m.group(2)), float(m.group(3))))

    def summarize(name: str, items: list[tuple[int, float, float]]) -> None:
        if not items:
            print(f"{name}: (no entries)")
            return
        br = max(items, key=lambda x: x[1])
        bra = max(items, key=lambda x: x[2])
        bm = max(items, key=lambda x: (x[1] + x[2]) / 2.0)
        print(f"{name}: n={len(items)}  epoch_range={items[0][0]}..{items[-1][0]}")
        print(f"  best RD: epoch {br[0]}  RD={br[1]:.6f}  RA={br[2]:.6f}")
        print(f"  best RA: epoch {bra[0]}  RD={bra[1]:.6f}  RA={bra[2]:.6f}")
        print(
            f"  best meanDice: epoch {bm[0]}  mean={(bm[1] + bm[2]) / 2:.6f}  "
            f"RD={bm[1]:.6f}  RA={bm[2]:.6f}"
        )

    print(f"log: {p}")
    summarize("Val Dice", vals)
    summarize("Test Dice", tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
