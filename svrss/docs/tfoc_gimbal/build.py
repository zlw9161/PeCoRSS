"""Inline ``data.json`` into ``template.html`` -> ``svrss/docs/tfoc_gimbal.html``.

The result is a single self-contained page (no network requests, no CDN).

Usage::

    python -m svrss.docs.tfoc_gimbal.export_data   # refresh the real tensors
    python -m svrss.docs.tfoc_gimbal.build
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "template.html"
DATA = HERE / "data.json"
OUT = HERE.parent / "tfoc_gimbal.html"
TOKEN = "/*__TFOC_DATA__*/null"


def main() -> None:
    html = TEMPLATE.read_text(encoding="utf-8")
    if TOKEN not in html:
        raise SystemExit(f"placeholder {TOKEN!r} not found in {TEMPLATE}")
    data = DATA.read_text(encoding="utf-8") if DATA.is_file() else "null"
    if not DATA.is_file():
        print("! data.json missing - building without the real-data section")
    # the payload is plain JSON (no '<'), so it is safe inside a <script> block
    if "</script" in data.lower():
        raise SystemExit("payload contains a script terminator")
    OUT.write_text(html.replace(TOKEN, data), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
