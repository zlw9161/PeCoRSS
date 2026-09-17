"""
Central asset path resolution for the EchMo radar echo segmentation app.

Why this module exists
----------------------
Model weights and datasets are not stored in the code repository; they are
pulled from Hugging Face. Those paths used to be hard-coded in `app.py`
(a `MODEL_PATHS` dict plus several `if dataset_name == ...` branches), so any
retrain that renamed a run directory -- or any new dataset -- required editing
the application code.

Now everything dataset-specific is *discovered from disk*:

    models/<dataset>/<run>/config.json          -> run folder + "model" field
    models/<dataset>/<run>/results/<model>.pt   -> weights (filename == model)
    datasets/<dataset>/dataset_meta.json        -> per-dataset behaviour

`datasets/<dataset>/dataset_meta.json` fields (all optional)
-----------------------------------------------------------
    display_name   str    label shown in the UI      (default: folder name)
    full_name      str    paper abbreviation / expansion (informational)
    scenario       str    short description             (informational)
    order          int    sort order of the dataset dropdown
    mtd_fftshift   bool   apply fftshift after the Doppler FFT in MTD
    mtd_amp_dir    str    sequence sub-folder holding the MTD amplitude .npy
    valid_classes  list   classes offered by the class filter in the UI
    run_overrides  dict   {"model": "run_folder"} to pin a specific run when a
                          dataset contains several runs of the same model

IMPORTANT: `valid_classes` is *not* `nb_classes`. `nb_classes` comes from the
model `config.json` and is the number of output channels used for the
segmentation colour scale. HiL-RED, for instance, trains with 9 channels while
only classes 0-6 actually occur in the annotations.

This module uses the standard library only, so it can be imported by scripts
without pulling in torch/gradio.
"""

from __future__ import annotations

import json
import os

# Pseudo dataset used by the "upload your own data" mode. It is not a real
# folder under datasets/ and must never be treated as one.
CUSTOM_DATASET = "Custom Data"

DATASETS_ROOT = "datasets"
MODELS_ROOT = "models"

INDEX_FILENAME = "light_dataset_frame_oriented.json"
META_FILENAME = "dataset_meta.json"
COEFS_DIRNAME = "coefs"
TRANSMIT_FILENAME = "s_tx.mat"

DEFAULT_META = {
    "mtd_fftshift": True,
    "mtd_amp_dir": "range_doppler_MTD",
    "valid_classes": [0, 1],
    "run_overrides": {},
}

_meta_cache: dict[str, dict] = {}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def read_json(path: str, default=None):
    # utf-8-sig reads plain UTF-8 unchanged and additionally strips a leading BOM.
    # Without it a BOM (editors and Excel-like tools add one happily) makes
    # json.load raise, the file is silently ignored and every setting falls back to
    # DEFAULT_META - that is how HiL-RED once ended up with valid_classes [0, 1].
    try:
        with open(path, "r", encoding="utf-8-sig") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return default


# --------------------------------------------------------------------------
# datasets
# --------------------------------------------------------------------------
def dataset_dir(dataset: str) -> str:
    return os.path.join(DATASETS_ROOT, dataset)


def index_path(dataset: str) -> str:
    return os.path.join(dataset_dir(dataset), INDEX_FILENAME)


def meta_path(dataset: str) -> str:
    return os.path.join(dataset_dir(dataset), META_FILENAME)


def coefs_dir(dataset: str) -> str:
    return os.path.join(dataset_dir(dataset), COEFS_DIRNAME)


def transmit_signal_path(dataset: str) -> str:
    return os.path.join(coefs_dir(dataset), TRANSMIT_FILENAME)


def load_dataset_meta(dataset: str) -> dict:
    """Merged dataset_meta.json (cached). Unknown datasets fall back to defaults."""
    if dataset not in _meta_cache:
        meta = dict(DEFAULT_META)
        loaded = read_json(meta_path(dataset), {})
        if isinstance(loaded, dict):
            meta.update(loaded)
        meta.setdefault("display_name", dataset)
        _meta_cache[dataset] = meta
    return _meta_cache[dataset]


def clear_meta_cache() -> None:
    _meta_cache.clear()


def available_datasets() -> list[str]:
    """Dataset folders that actually have a frame index, in display order."""
    if not os.path.isdir(DATASETS_ROOT):
        return []
    found = []
    for name in os.listdir(DATASETS_ROOT):
        if name == CUSTOM_DATASET or not os.path.isdir(dataset_dir(name)):
            continue
        if not os.path.isfile(index_path(name)):
            continue  # without the index the dataset cannot be shown at all
        found.append((load_dataset_meta(name).get("order", 10_000), name))
    found.sort(key=lambda item: (item[0], item[1]))
    return [name for _, name in found]


def load_dataset_index(dataset: str) -> dict:
    """{sequence: [frames]} -- {} when missing or not valid strict JSON."""
    data = read_json(index_path(dataset), None)
    return data if isinstance(data, dict) else {}


def get_sequences(dataset: str) -> list[str]:
    if dataset == CUSTOM_DATASET:
        return []
    return list(load_dataset_index(dataset).keys())


def get_frames(dataset: str, sequence: str) -> list:
    if dataset == CUSTOM_DATASET or not sequence:
        return []
    return load_dataset_index(dataset).get(sequence, [])


def echo_path(dataset: str, sequence: str, frame: str) -> str:
    return os.path.join(dataset_dir(dataset), sequence, "range_doppler_echo", f"{frame}.npy")


def annotation_path(dataset: str, sequence: str, frame: str) -> str:
    return os.path.join(dataset_dir(dataset), sequence, "annotations", "dense", frame, "range_doppler.npy")


def mtd_amp_path(dataset: str, sequence: str, frame: str) -> str:
    return os.path.join(dataset_dir(dataset), sequence, load_dataset_meta(dataset)["mtd_amp_dir"], f"{frame}.npy")


# --------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------
def runs_dir(dataset: str) -> str:
    return os.path.join(MODELS_ROOT, dataset)


def run_dir(dataset: str, run_folder: str) -> str:
    return os.path.join(runs_dir(dataset), run_folder)


def run_config_path(dataset: str, run_folder: str) -> str:
    return os.path.join(run_dir(dataset, run_folder), "config.json")


def run_weights_path(dataset: str, run_folder: str, filename: str) -> str:
    return os.path.join(run_dir(dataset, run_folder), "results", filename)


def load_run_config(dataset: str, run_folder: str) -> dict:
    data = read_json(run_config_path(dataset, run_folder), None)
    return data if isinstance(data, dict) else {}


def discover_runs(dataset: str) -> dict[str, tuple[str, str]]:
    """
    {model_name: (run_folder, weights_filename)} discovered from disk.

    A run folder qualifies when its config.json declares `"model"` and
    `results/<model>.pt` exists. When several runs train the same model the
    alphabetically first folder wins, unless `run_overrides` in
    dataset_meta.json pins a specific one.
    """
    runs: dict[str, tuple[str, str]] = {}
    if not os.path.isdir(runs_dir(dataset)):
        return runs

    for folder in sorted(os.listdir(runs_dir(dataset))):
        if not os.path.isdir(run_dir(dataset, folder)):
            continue
        cfg = load_run_config(dataset, folder)
        model = cfg.get("model") or cfg.get("model_name")
        if not model:
            continue
        filename = f"{model}.pt"
        if not os.path.isfile(run_weights_path(dataset, folder, filename)):
            continue
        runs.setdefault(model, (folder, filename))

    for model, folder in (load_dataset_meta(dataset).get("run_overrides") or {}).items():
        filename = f"{model}.pt"
        if os.path.isfile(run_weights_path(dataset, folder, filename)):
            runs[model] = (folder, filename)
        else:
            print(f"[warn] run_overrides: '{folder}' for model '{model}' in '{dataset}' is not usable")

    return runs


def model_names(dataset: str) -> list[str]:
    return sorted(discover_runs(dataset))
