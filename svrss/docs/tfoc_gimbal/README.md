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
and a trained TFoC checkpoint. The path is the `CKPT` constant at the top of
`export_data.py`; the shipped `data.json` comes from the **last-frame-anchored**
model (`pdr-ss3-t9-0.90.95-…-s42_0`, `best_dice = 0.809`).

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
- §4 numbers are a real forward pass; spatial weights stay near uniform \(1/n\).
  That is **not** a flat-softmax artefact: with \(\theta=\sigma_{spa}=0.9\) and
  \(s\in[0,1]\) the head-room is \(e^{1/\theta}\approx 3.0\times\), but the nine
  candidates of a 3×3 window differ by at most \(0.15\) in score, so the observed
  weight ratio is \(\le 1.18\times\). The largest spread sits on the target point
  (~3× the neighbour/clutter probes) — the reweighting is gentle *and* targeted.
- `attn_temperature` (\(\sigma_{spa}\)) is a plain float on `TFocConvV1`, **not**
  part of the `state_dict`, so a checkpoint never restores it. `export_data.py`
  sets it explicitly via `ATTN_T`; it must match the value used for training
  (this run: `0.9`, while the model source currently defaults to `1.0`).
- The supervised target frame is resolved from per-frame IoU against the prediction,
  not from the demo’s temporal-RF centre offset. With the current checkpoint the
  anchor is the **last** input frame (preset → `#8`, IoU `1.000`, all other frames `0`),
  so the target feature position’s temporal receptive field ends exactly on it.

## License

Same as the parent PeCoRSS / TFoC repository.
