# TFoC Gimbal Demo (HTML)

Interactive explainer for **TFoC** (Temporal-Spatial Focusing Convolution):
the “camera gimbal” view of soft spatial alignment + temporal aggregation.

给卷积装一台云台 — 单文件、离线可打开的动态演示页。

## Quick start

Open the built page in a browser (no server required):

```text
svrss/docs/tfoc_gimbal.html
```

Or from the repo root:

```bash
# Windows
start svrss/docs/tfoc_gimbal.html

# macOS
open svrss/docs/tfoc_gimbal.html

# Linux
xdg-open svrss/docs/tfoc_gimbal.html
```

The page is self-contained (HTML + CSS + Canvas JS + inlined real tensors).
No CDN and no network requests.

## What’s inside

| Section | Content |
|--------|---------|
| Hero | Gimbal rings tracking a moving target |
| §1 | Rigid mount vs 3-axis gimbal (interactive shake / follow gain) |
| §2 | Fixed Conv3d weights vs adaptive TFoC weights (toy energy purity) |
| §3 | Five-step forward pass walkthrough (gather → pair → spatial → temporal → fuse) |
| §4 | Real tensors from the trained SV checkpoint on the SeaUAV preset |
| §5 | Compact equations + design notes |

Colour palette matches the Gradio demo (dark indigo / purple / amber / red).

## Layout

```text
svrss/docs/
  tfoc_gimbal.html          # ready-to-open page (built)
  tfoc_gimbal/
    README.md               # this file
    template.html           # page source (DATA placeholder)
    data.json               # exported real internals (uint8 maps + probes)
    export_data.py          # run model → data.json
    build.py                # template + data.json → ../tfoc_gimbal.html
```

## Rebuild (optional)

Needs the `svrss` package, the demo preset under `svrss/demo/assets/preset/`,
and a trained `tfoc_sv_v1_revise` checkpoint resolvable by the demo registry.

From the **repo root** (`PeCoRSS` / `Code`):

```bash
# refresh real attention / feature maps from the checkpoint
python -m svrss.docs.tfoc_gimbal.export_data

# bake data.json into the standalone HTML
python -m svrss.docs.tfoc_gimbal.build
```

If `data.json` is missing, `build.py` still emits a page; §4 (real data) is omitted.

## Notes

- §2’s “+X pt” gain is a **schematic** purity difference, not a dataset mIoU/Dice.
- §4 numbers are a real forward pass; spatial weights stay near uniform \(1/n\)
  (gentle reweighting under \(\theta=1\)), which is intentional in the gimbal narrative.
- The supervised target frame is resolved from per-frame IoU against the prediction
  (preset → input frame `#4`), not from the demo’s temporal-RF centre offset.

## License

Same as the parent PeCoRSS / TFoC repository.
