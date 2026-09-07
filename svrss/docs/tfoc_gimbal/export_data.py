"""Export real TFoC internals for the animated explainer page.

Runs the bundled 9-frame preset through the trained ``tfoc_sv_v1_revise``
checkpoint, captures the TFocConv tensors and writes a compact JSON payload
(uint8 maps are base64-encoded) consumed by ``template.html``.

Usage::

    python -m svrss.docs.tfoc_gimbal.export_data          # from the repo root
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import torch

from svrss.demo import data_utils as du
from svrss.demo.inference import InferenceEngine
from svrss.demo.model_registry import get_spec

OUT = Path(__file__).resolve().parent / "data.json"

# Checkpoint used for the real-data section. This model anchors on the LAST
# frame of the input window (the earlier one anchored on the middle frame).
CKPT = (r"D:\PaperWork\TFoc\Model\tfoc_pt"
        r"\pdr-ss3-t9-0.90.95-tfoc_sv_v1_revise_range_doppler_e300_lr9.6e-05_s42_0"
        r"\results\model.pt")
TGT = -1                      # -1 = last frame of the window
# sigma_spa: the spatial-softmax temperature the run was trained with. It is a
# plain float on the module (not part of the state_dict), so loading the
# checkpoint does NOT restore it -- it has to be set by hand to match training.
ATTN_T = 0.9

# crop covering the whole target track with margin (rows, cols of the 64x800 RD)
R0, R1, C0, C1 = 12, 44, 72, 208


def _u8(a: np.ndarray, lo=None, hi=None) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    lo = float(a.min()) if lo is None else float(lo)
    hi = float(a.max()) if hi is None else float(hi)
    if hi - lo < 1e-12:
        return np.zeros(a.shape, np.uint8)
    return np.clip((a - lo) / (hi - lo) * 255.0, 0, 255).astype(np.uint8)


def _b64(a: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(a, dtype=np.uint8).tobytes()).decode()


def _crop(a: np.ndarray) -> np.ndarray:
    return a[..., R0:R1, C0:C1]


def _energy(feat: np.ndarray) -> np.ndarray:
    """(c, h, w) -> (h, w) channel L2."""
    return np.sqrt(np.sum(feat.astype(np.float32) ** 2, axis=0))


def main() -> None:
    seq = du.load_preset()
    n_frames = seq.num_frames
    rd, masks, ids, _ = du.select_window(seq, n_frames)
    x = du.build_input_tensor(rd, seq.signal_type)

    spec = get_spec("tfoc_sv_v1_revise")
    eng = InferenceEngine(spec, n_frames=n_frames, device="cpu")
    info = eng.load_checkpoint(CKPT)
    print("checkpoint:", info.message)
    eng.set_tgt(TGT)
    tfoc = eng.net.encoding_branch.tfoc_conv_block
    print(f"attn_temperature: {tfoc.attn_temperature} -> {ATTN_T}")
    tfoc.attn_temperature = float(ATTN_T)

    cap = eng.capture_tfoc(x)
    out = eng.forward(x)
    d = cap["dims"]
    t, h, w, n, k_n, pad = d["t"], d["h"], d["w"], d["n"], d["k_n"], d["pad"]
    hist_idx, tgt_i = cap["hist_idx"], cap["tgt_i"]
    n_hist = len(hist_idx)
    print("dims:", d, "hist_idx:", hist_idx, "tgt_i:", tgt_i)

    attn = cap["attn"].numpy()[0]                    # (n_hist, h*w, n)
    attn_score = cap["attn_score"].numpy()[0]        # (n_hist, h*w, n)
    time_score = cap["time_score"].numpy()[0]        # (n_hist, h*w)
    time_w = cap["time_w"].numpy()[0]                # (n_hist, h*w)
    x_tgt = cap["x_tgt_ori"].numpy()[0, :, 0]        # (c, h, w)
    x_fin = cap["x_hist_agg_final"].numpy()[0, :, 0]  # (c, h, w)
    seg = out["seg_argmax"].numpy()[0]
    if seg.ndim == 3:
        seg = seg[0]

    # per-frame target centre from the dense masks
    track = []
    for k in range(n_frames):
        am = masks[k].argmax(0)
        ys, xs = np.nonzero(am > 0)
        track.append([int(round(ys.mean())), int(round(xs.mean()))] if len(ys) else None)

    # Which input frame does the prediction actually align with? The demo's
    # feature->input mapping is the temporal receptive-field centre, which is not
    # the frame the network was supervised on; resolve it from the data instead.
    ious = []
    for k in range(n_frames):
        g, s = masks[k].argmax(0) > 0, seg > 0
        u = int((g | s).sum())
        ious.append(round(float((g & s).sum() / u) if u else 0.0, 4))
    tgt_in = int(np.argmax(ious))
    ti, tj = track[tgt_in]

    # probe points: the target, a point just outside it, and clutter
    probes = [
        {"key": "tgt", "i": ti, "j": tj, "kind": "fg"},
        {"key": "near", "i": ti + 3, "j": tj + 5, "kind": "near"},
        {"key": "bg", "i": 46, "j": 300, "kind": "bg"},
    ]
    for p in probes:
        q = p["i"] * w + p["j"]
        p["attn"] = [[round(float(v), 5) for v in attn[hi, q]] for hi in range(n_hist)]
        p["score"] = [[round(float(v), 5) for v in attn_score[hi, q]] for hi in range(n_hist)]
        p["time_score"] = [round(float(time_score[hi, q]), 5) for hi in range(n_hist)]
        p["time_w"] = [round(float(time_w[hi, q]), 5) for hi in range(n_hist)]

    ch, cw = R1 - R0, C1 - C0
    rd_lo, rd_hi = float(rd.min()), float(rd.max())

    payload = {
        "seq": seq.name,
        "frames": ids,
        "crop": {"r0": R0, "c0": C0, "h": ch, "w": cw, "H": h, "W": w},
        "dims": {"t": t, "n_hist": n_hist, "k_n": k_n, "pad": pad, "n": n,
                 "c": d["c"], "h": h, "w": w, "n_input": n_frames},
        "tgt_i": tgt_i, "tgt_in": tgt_in, "attn_t": float(ATTN_T),
        "hist_idx": hist_idx, "seg_iou": ious,
        "track": track,
        "probes": probes,
        "attn_vmax": round(float(attn.max()), 5),
        "maps": {
            "rd": _b64(_crop(_u8(rd, rd_lo, rd_hi))),
            "gt": _b64(_crop(np.stack([m.argmax(0) for m in masks]).astype(np.uint8))),
            "seg": _b64(_crop(seg.astype(np.uint8))),
            "feat_tgt": _b64(_crop(_u8(_energy(x_tgt)))),
            "feat_final": _b64(_crop(_u8(_energy(x_fin)))),
        },
        "score_range": [round(float(time_score.min()), 4),
                        round(float(time_score.max()), 4)],
    }
    OUT.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size/1024:.1f} KB)")

    print("\nper-frame seg IoU:", ious, "-> target frame", tgt_in)
    print("-- probe readouts --")
    for p in probes:
        print(f"{p['key']:5s} @({p['i']},{p['j']})  time_w={p['time_w']}")
        for hi in range(n_hist):
            print(f"       tau={hi} attn={p['attn'][hi]}")
    print("\ntrack:", track)


if __name__ == "__main__":
    main()
