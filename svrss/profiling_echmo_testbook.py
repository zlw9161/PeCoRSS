"""Synthetic profiling harness for the EchMo family.

This script measures latency and memory usage for inference and training-like
execution on configurable hardware without involving dataset IO.
"""

import argparse
import json
import numbers
import platform
import time
from pathlib import Path
from statistics import mean, pstdev
from types import SimpleNamespace

import torch
import torch.distributed as dist
import torch.optim as optim

from svrss.infer_echmo import (
    _build_model,
    _get_default_name_exp,
    _load_state_dict,
    _resolve_model_path,
    _resolve_num_classes,
)
from svrss.utils.distributed_utils import get_rank, get_world_size, init_distributed_mode, is_main_process
from svrss.utils.functions import count_params
from svrss.utils.paths import Paths


SUPPORTED_TASKS = {"train", "infer"}
SUPPORTED_LEVELS = {"module-wise", "end2end", "front-back"}
SUPPORTED_DATA_MODES = {"random", "config"}
MODULE_ORDER = [
    "normalization",
    "LMFChains",
    "MLDC",
    "GDF",
    "FeatPack",
    "PKC-Seg Head",
]
FRONT_BACK_ORDER = ["Front-end", "FeatPack", "Back-end"]


class StageAdapter:
    """Expose canonical EchMo stage methods from direct or wrapped models."""

    def __init__(self, owner_model, stage_model, source):
        self.owner_model = owner_model
        self.stage_model = stage_model
        self.source = source

    def normalize_input(self, x):
        return self.stage_model.normalize_input(x)

    def forward_lmf(self, x):
        return self.stage_model.forward_lmf(x)

    def forward_mldc(self, x):
        return self.stage_model.forward_mldc(x)

    def forward_gdf(self, x):
        return self.stage_model.forward_gdf(x)

    def pack_features(self, x):
        return self.stage_model.pack_features(x)

    def forward_head(self, x):
        return self.stage_model.forward_head(x)

    def forward_frontend_core(self, x):
        x_norm = self.normalize_input(x)
        x_k = self.forward_lmf(x_norm)
        x_h = self.forward_mldc(x_k)
        x_d = self.forward_gdf(x_h)
        return x_d


def _parse_csv_set(raw_value, supported_values, field_name):
    items = []
    for token in raw_value.split(","):
        item = token.strip()
        if not item:
            continue
        if item not in supported_values:
            supported = ", ".join(sorted(supported_values))
            raise ValueError(f"Unsupported {field_name}: {item}. Supported values: {supported}.")
        items.append(item)
    if not items:
        raise ValueError(f"At least one {field_name} must be provided.")
    return items


def _cuda_sync_if_needed(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _is_distributed_active():
    return dist.is_available() and dist.is_initialized()


def _distributed_reduce_number(value):
    if not _is_distributed_active() or isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        tensor = torch.tensor(float(value), dtype=torch.float64, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        reduced = tensor.item() / get_world_size()
        return int(round(reduced))
    if isinstance(value, numbers.Real):
        tensor = torch.tensor(float(value), dtype=torch.float64, device="cuda")
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor.item() / get_world_size()
    return value


def _distributed_reduce_nested(value):
    if isinstance(value, dict):
        return {key: _distributed_reduce_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_distributed_reduce_nested(item) for item in value]
    if value is None or isinstance(value, (str, Path, bool)):
        return value
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _distributed_reduce_nested(value.item())
        return value
    if isinstance(value, numbers.Number):
        return _distributed_reduce_number(value)
    return value


def _reset_peak_memory(device):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _capture_peak_memory(device):
    if device.type != "cuda":
        return {
            "peak_allocated_mb": None,
            "peak_reserved_mb": None,
        }
    return {
        "peak_allocated_mb": torch.cuda.max_memory_allocated(device) / (1024.0 ** 2),
        "peak_reserved_mb": torch.cuda.max_memory_reserved(device) / (1024.0 ** 2),
    }


def _maybe_empty_cache(device):
    if device.type == "cuda":
        torch.cuda.empty_cache()


def _measure_call(fn, device):
    _reset_peak_memory(device)
    _cuda_sync_if_needed(device)
    start_time = time.perf_counter()
    result = fn()
    _cuda_sync_if_needed(device)
    elapsed_ms = (time.perf_counter() - start_time) * 1000.0
    metrics = {"latency_ms": elapsed_ms}
    metrics.update(_capture_peak_memory(device))
    return result, metrics


def _summarize_metrics(records):
    if not records:
        return {
            "iterations": 0,
            "latency_ms": None,
            "latency_std_ms": None,
            "latency_min_ms": None,
            "latency_max_ms": None,
            "peak_allocated_mb": None,
            "peak_allocated_max_mb": None,
            "peak_reserved_mb": None,
            "peak_reserved_max_mb": None,
        }

    latencies = [record["latency_ms"] for record in records]
    allocs = [record["peak_allocated_mb"] for record in records if record["peak_allocated_mb"] is not None]
    reserves = [record["peak_reserved_mb"] for record in records if record["peak_reserved_mb"] is not None]
    return {
        "iterations": len(records),
        "latency_ms": mean(latencies),
        "latency_std_ms": pstdev(latencies) if len(latencies) > 1 else 0.0,
        "latency_min_ms": min(latencies),
        "latency_max_ms": max(latencies),
        "peak_allocated_mb": mean(allocs) if allocs else None,
        "peak_allocated_max_mb": max(allocs) if allocs else None,
        "peak_reserved_mb": mean(reserves) if reserves else None,
        "peak_reserved_max_mb": max(reserves) if reserves else None,
    }


def _round_float(value):
    if value is None:
        return None
    return round(float(value), 4)


def _round_nested(value):
    if isinstance(value, dict):
        return {key: _round_nested(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_nested(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return _round_nested(value.item())
        return [_round_nested(item) for item in value.detach().cpu().tolist()]
    if isinstance(value, bool):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        return _round_float(value)
    return value


def _format_number(value, digits=3, suffix=""):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}{suffix}"


def _load_config(cfg_path):
    with open(cfg_path, "r") as fp:
        cfg = json.load(fp)
    cfg["nb_classes"] = _resolve_num_classes(cfg["dataset"])
    return cfg


def _resolve_optional_arg(cli_value, cfg, cfg_keys, fallback):
    if cli_value is not None:
        return cli_value
    for key in cfg_keys:
        if key in cfg and cfg[key] is not None:
            return cfg[key]
    return fallback


def _resolve_device(cfg, requested_device):
    if requested_device:
        return torch.device(requested_device), None

    config_device = cfg.get("device", "cuda")
    if str(config_device).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu"), "Config requested CUDA, but CUDA is unavailable. Falling back to CPU."
    return torch.device(config_device), None


def _build_experiment_path(cfg):
    paths = Paths().get()
    name_exp = cfg.get("name_exp", _get_default_name_exp(cfg))
    version = cfg.get("version", 1)
    return paths["logs"] / cfg["dataset"] / cfg["model"] / f"{name_exp}_{version}"


def _load_model_for_profiling(cfg, args, notes):
    model = _build_model(cfg)
    checkpoint_path = None

    if args.load_mode == "pretrained":
        exp_path = _build_experiment_path(cfg)
        resolve_args = SimpleNamespace(model_path=args.model_path, checkpoint_epoch=args.checkpoint_epoch)
        checkpoint_path = _resolve_model_path(cfg, resolve_args, exp_path)
        _load_state_dict(model, checkpoint_path)
        notes.append(f"Loaded pretrained weights from {checkpoint_path}.")
    elif args.model_path:
        notes.append("Ignored --model-path because load mode is scratch.")

    return model, checkpoint_path


def _create_random_input(cfg, batch_size, device):
    shape = (batch_size, cfg["nb_input_channels"], cfg["w_size"], cfg["h_size"])
    if cfg.get("cmplx_flag", False):
        real = torch.randn(shape, device=device, dtype=torch.float32)
        imag = torch.randn(shape, device=device, dtype=torch.float32)
        return torch.complex(real, imag)
    return torch.randn(shape, device=device, dtype=torch.float32)


def _create_config_input(cfg, batch_size, device):
    shape = (batch_size, cfg["nb_input_channels"], cfg["w_size"], cfg["h_size"])
    total = 1
    for dim in shape:
        total *= int(dim)
    scale = float(cfg.get("profile_input_scale", cfg.get("scale_factor", 1.0)))
    offset = float(cfg.get("profile_input_offset", 0.0))
    base = torch.linspace(-1.0, 1.0, steps=total, device=device, dtype=torch.float32).reshape(shape)
    real = base * scale + offset
    if cfg.get("cmplx_flag", False):
        imag = torch.flip(base, dims=[-1]) * scale
        return torch.complex(real, imag)
    return real


def _create_profile_input(cfg, batch_size, device, data_mode):
    if data_mode == "random":
        return _create_random_input(cfg, batch_size, device)
    if data_mode == "config":
        return _create_config_input(cfg, batch_size, device)
    raise ValueError(f"Unsupported data mode: {data_mode}.")


def _resolve_runtime_settings(cfg, args):
    batch_size = int(_resolve_optional_arg(args.batch_size, cfg, ["profile_batch_size", "batch_size"], 1))
    warmup_iters = int(_resolve_optional_arg(args.warmup_iters, cfg, ["profile_warmup_iters"], 3))
    measure_iters = int(_resolve_optional_arg(args.measure_iters, cfg, ["profile_measure_iters"], 10))
    train_lr = float(_resolve_optional_arg(args.train_lr, cfg, ["profile_train_lr", "lr"], 1e-4))
    data_mode = str(_resolve_optional_arg(args.data_mode, cfg, ["profile_data_mode"], "random")).lower()
    if data_mode not in SUPPORTED_DATA_MODES:
        supported = ", ".join(sorted(SUPPORTED_DATA_MODES))
        raise ValueError(f"Unsupported data mode: {data_mode}. Supported values: {supported}.")
    return {
        "batch_size": batch_size,
        "warmup_iters": warmup_iters,
        "measure_iters": measure_iters,
        "train_lr": train_lr,
        "data_mode": data_mode,
    }


def _scalar_loss(output_tensor):
    tensor = output_tensor.float()
    return tensor.square().mean()


def _unwrap_model(model):
    return model.module if hasattr(model, "module") else model


def _has_stage_interface(model):
    required_methods = [
        "normalize_input",
        "forward_lmf",
        "forward_mldc",
        "forward_gdf",
        "pack_features",
        "forward_head",
    ]
    return all(hasattr(model, method_name) for method_name in required_methods)


def _build_stage_adapter(model):
    root_model = _unwrap_model(model)
    if _has_stage_interface(root_model):
        return StageAdapter(root_model, root_model, "direct")

    if hasattr(root_model, "backbone") and _has_stage_interface(root_model.backbone):
        return StageAdapter(root_model, root_model.backbone, "backbone")

    if hasattr(root_model, "stage1") and _has_stage_interface(root_model.stage1):
        return StageAdapter(root_model, root_model.stage1, "stage1")

    return None


def _prepare_inputs(adapter, model_input):
    with torch.no_grad():
        x_norm = adapter.normalize_input(model_input)
        x_k = adapter.forward_lmf(x_norm)
        x_h = adapter.forward_mldc(x_k)
        x_d = adapter.forward_gdf(x_h)
        packed = adapter.pack_features(x_d)
    return {
        "normalization": model_input,
        "LMFChains": x_norm.detach(),
        "MLDC": x_k.detach(),
        "GDF": x_h.detach(),
        "FeatPack": x_d.detach(),
        "PKC-Seg Head": packed.detach(),
        "front_end_input": model_input,
        "featpack_input": x_d.detach(),
        "back_end_input": packed.detach(),
    }


def _clone_state_dict(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def _run_iterations(warmup_iters, measure_iters, step_fn):
    records = []
    total_iters = warmup_iters + measure_iters
    for iteration in range(total_iters):
        metrics = step_fn(iteration)
        if iteration >= warmup_iters:
            records.append(metrics)
    return records


def _profile_end2end_infer(model, model_input, warmup_iters, measure_iters, device):
    model.eval()

    def _step(_):
        with torch.no_grad():
            _, metrics = _measure_call(lambda: model(model_input), device)
        return metrics

    return _summarize_metrics(_run_iterations(warmup_iters, measure_iters, _step))


def _profile_end2end_train(model, model_input, warmup_iters, measure_iters, device, lr):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    forward_records = []
    backward_records = []
    optimizer_records = []
    total_records = []
    total_iters = warmup_iters + measure_iters

    for iteration in range(total_iters):
        optimizer.zero_grad(set_to_none=True)
        outputs, forward_metrics = _measure_call(lambda: model(model_input), device)
        loss = _scalar_loss(outputs)
        _, backward_metrics = _measure_call(loss.backward, device)
        _, optimizer_metrics = _measure_call(optimizer.step, device)

        total_metrics = {
            "latency_ms": forward_metrics["latency_ms"]
            + backward_metrics["latency_ms"]
            + optimizer_metrics["latency_ms"],
            "peak_allocated_mb": max(
                value
                for value in [
                    forward_metrics["peak_allocated_mb"],
                    backward_metrics["peak_allocated_mb"],
                    optimizer_metrics["peak_allocated_mb"],
                ]
                if value is not None
            )
            if device.type == "cuda"
            else None,
            "peak_reserved_mb": max(
                value
                for value in [
                    forward_metrics["peak_reserved_mb"],
                    backward_metrics["peak_reserved_mb"],
                    optimizer_metrics["peak_reserved_mb"],
                ]
                if value is not None
            )
            if device.type == "cuda"
            else None,
        }

        if iteration >= warmup_iters:
            forward_records.append(forward_metrics)
            backward_records.append(backward_metrics)
            optimizer_records.append(optimizer_metrics)
            total_records.append(total_metrics)

    return {
        "forward": _summarize_metrics(forward_records),
        "backward": _summarize_metrics(backward_records),
        "optimizer": _summarize_metrics(optimizer_records),
        "total": _summarize_metrics(total_records),
        "loss": _round_float(float(loss.detach().item())),
    }


def _profile_module_wise_infer(adapter, model_input, warmup_iters, measure_iters, device):
    adapter.owner_model.eval()
    module_records = {module_name: [] for module_name in MODULE_ORDER}

    def _step(_):
        with torch.no_grad():
            x_norm, metrics_norm = _measure_call(lambda: adapter.normalize_input(model_input), device)
            x_k, metrics_lmf = _measure_call(lambda: adapter.forward_lmf(x_norm), device)
            x_h, metrics_mldc = _measure_call(lambda: adapter.forward_mldc(x_k), device)
            x_d, metrics_gdf = _measure_call(lambda: adapter.forward_gdf(x_h), device)
            packed, metrics_pack = _measure_call(lambda: adapter.pack_features(x_d), device)
            _, metrics_head = _measure_call(lambda: adapter.forward_head(packed), device)
        return {
            "normalization": metrics_norm,
            "LMFChains": metrics_lmf,
            "MLDC": metrics_mldc,
            "GDF": metrics_gdf,
            "FeatPack": metrics_pack,
            "PKC-Seg Head": metrics_head,
        }

    results = _run_iterations(warmup_iters, measure_iters, _step)
    for result in results:
        for module_name, metrics in result.items():
            module_records[module_name].append(metrics)
    return {module_name: {"forward": _summarize_metrics(records)} for module_name, records in module_records.items()}


def _profile_module_wise_train(adapter, model_input, warmup_iters, measure_iters, device):
    adapter.owner_model.train()
    module_results = {}
    stage_methods = {
        "normalization": adapter.normalize_input,
        "LMFChains": adapter.forward_lmf,
        "MLDC": adapter.forward_mldc,
        "GDF": adapter.forward_gdf,
        "FeatPack": adapter.pack_features,
        "PKC-Seg Head": adapter.forward_head,
    }

    boundary_inputs = _prepare_inputs(adapter, model_input)

    for module_name in MODULE_ORDER:
        stage_fn = stage_methods[module_name]
        stage_input = boundary_inputs[module_name]
        forward_records = []
        backward_records = []

        total_iters = warmup_iters + measure_iters
        for iteration in range(total_iters):
            adapter.owner_model.zero_grad(set_to_none=True)
            output_tensor, forward_metrics = _measure_call(lambda: stage_fn(stage_input), device)

            adapter.owner_model.zero_grad(set_to_none=True)
            leaf_input = stage_input.detach().requires_grad_(True)
            stage_output = stage_fn(leaf_input)
            grad_output = torch.ones_like(stage_output)
            _, backward_metrics = _measure_call(lambda: torch.autograd.backward(stage_output, grad_output), device)
            adapter.owner_model.zero_grad(set_to_none=True)

            if iteration >= warmup_iters:
                forward_records.append(forward_metrics)
                backward_records.append(backward_metrics)

        module_results[module_name] = {
            "forward": _summarize_metrics(forward_records),
            "backward": _summarize_metrics(backward_records),
        }

    return module_results


def _profile_front_back_infer(adapter, model_input, warmup_iters, measure_iters, device):
    adapter.owner_model.eval()
    segment_records = {segment_name: [] for segment_name in FRONT_BACK_ORDER}

    def _step(_):
        with torch.no_grad():
            x_d, front_metrics = _measure_call(lambda: adapter.forward_frontend_core(model_input), device)
            packed, pack_metrics = _measure_call(lambda: adapter.pack_features(x_d), device)
            _, back_metrics = _measure_call(lambda: adapter.forward_head(packed), device)
        return {
            "Front-end": front_metrics,
            "FeatPack": pack_metrics,
            "Back-end": back_metrics,
        }

    results = _run_iterations(warmup_iters, measure_iters, _step)
    for result in results:
        for segment_name, metrics in result.items():
            segment_records[segment_name].append(metrics)
    return {segment_name: {"forward": _summarize_metrics(records)} for segment_name, records in segment_records.items()}


def _profile_front_back_train(adapter, model_input, warmup_iters, measure_iters, device):
    adapter.owner_model.train()
    boundary_inputs = _prepare_inputs(adapter, model_input)
    segment_results = {}

    def _frontend_core(x):
        return adapter.forward_frontend_core(x)

    segment_methods = {
        "Front-end": (_frontend_core, boundary_inputs["front_end_input"]),
        "FeatPack": (adapter.pack_features, boundary_inputs["featpack_input"]),
        "Back-end": (adapter.forward_head, boundary_inputs["back_end_input"]),
    }

    for segment_name in FRONT_BACK_ORDER:
        stage_fn, stage_input = segment_methods[segment_name]
        forward_records = []
        backward_records = []
        total_iters = warmup_iters + measure_iters

        for iteration in range(total_iters):
            adapter.owner_model.zero_grad(set_to_none=True)
            _, forward_metrics = _measure_call(lambda: stage_fn(stage_input), device)

            adapter.owner_model.zero_grad(set_to_none=True)
            leaf_input = stage_input.detach().requires_grad_(True)
            stage_output = stage_fn(leaf_input)
            grad_output = torch.ones_like(stage_output)
            _, backward_metrics = _measure_call(lambda: torch.autograd.backward(stage_output, grad_output), device)
            adapter.owner_model.zero_grad(set_to_none=True)

            if iteration >= warmup_iters:
                forward_records.append(forward_metrics)
                backward_records.append(backward_metrics)

        segment_results[segment_name] = {
            "forward": _summarize_metrics(forward_records),
            "backward": _summarize_metrics(backward_records),
        }

    return segment_results


def _markdown_summary_rows(section_data, include_backward):
    rows = [
        "| Name | Forward (ms) | Backward (ms) | Peak Alloc (MB) | Peak Reserved (MB) |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, item in section_data.items():
        forward_stats = item.get("forward", {})
        backward_stats = item.get("backward", {})
        if include_backward:
            alloc_value = backward_stats.get("peak_allocated_mb")
            reserve_value = backward_stats.get("peak_reserved_mb")
        else:
            alloc_value = forward_stats.get("peak_allocated_mb")
            reserve_value = forward_stats.get("peak_reserved_mb")
        rows.append(
            "| {} | {} | {} | {} | {} |".format(
                name,
                _format_number(forward_stats.get("latency_ms")),
                _format_number(backward_stats.get("latency_ms")) if include_backward else "N/A",
                _format_number(alloc_value),
                _format_number(reserve_value),
            )
        )
    return rows


def _build_markdown_report(report):
    metadata = report["metadata"]
    lines = [
        "# EchMo Profiling Testbook",
        "",
        "## Run Metadata",
        "",
        f"- Timestamp: {metadata['timestamp']}",
        f"- Model: {metadata['model_name']}",
        f"- Dataset: {metadata['dataset']}",
        f"- Load Mode: {metadata['load_mode']}",
        f"- Device: {metadata['device']}",
        f"- Device Name: {metadata['device_name']}",
        f"- Distributed: {metadata['distributed']}",
        f"- Rank: {metadata['rank']}",
        f"- World Size: {metadata['world_size']}",
        f"- Data Mode: {metadata['data_mode']}",
        f"- Batch Size: {metadata['batch_size']}",
        f"- Batch Size Per Rank: {metadata['batch_size_per_rank']}",
        f"- Global Batch Size: {metadata['global_batch_size']}",
        f"- Warmup Iters: {metadata['warmup_iters']}",
        f"- Measure Iters: {metadata['measure_iters']}",
        f"- Train LR: {metadata['train_lr']}",
        f"- Trainable Params: {metadata['trainable_params']}",
        f"- Stage Adapter: {metadata['stage_adapter']}",
        "",
    ]

    if report["notes"]:
        lines.extend(["## Notes", ""])
        for note in report["notes"]:
            lines.append(f"- {note}")
        lines.append("")

    for task_name, task_payload in report["results"].items():
        lines.extend([f"## {task_name.title()} Profiling", ""])
        for level_name, level_payload in task_payload.items():
            lines.extend([f"### {level_name}", ""])
            if level_payload.get("status") == "unsupported":
                lines.append(f"- Unsupported: {level_payload['reason']}")
                lines.append("")
                continue

            if level_name == "end2end":
                if task_name == "infer":
                    lines.extend([
                        "| Slice | Latency (ms) | Peak Alloc (MB) | Peak Reserved (MB) |",
                        "| --- | ---: | ---: | ---: |",
                        "| Total | {} | {} | {} |".format(
                            _format_number(level_payload["latency_ms"]),
                            _format_number(level_payload["peak_allocated_mb"]),
                            _format_number(level_payload["peak_reserved_mb"]),
                        ),
                        "",
                    ])
                else:
                    lines.extend([
                        "| Slice | Latency (ms) | Peak Alloc (MB) | Peak Reserved (MB) |",
                        "| --- | ---: | ---: | ---: |",
                    ])
                    for slice_name in ["forward", "backward", "optimizer", "total"]:
                        slice_payload = level_payload.get(slice_name, {})
                        lines.append(
                            "| {} | {} | {} | {} |".format(
                                slice_name,
                                _format_number(slice_payload.get("latency_ms")),
                                _format_number(slice_payload.get("peak_allocated_mb")),
                                _format_number(slice_payload.get("peak_reserved_mb")),
                            )
                        )
                    lines.append("")
                    lines.append(f"- Scalar Loss Proxy: {_format_number(level_payload.get('loss'))}")
                    lines.append("")
            else:
                include_backward = task_name == "train"
                lines.extend(_markdown_summary_rows(level_payload, include_backward=include_backward))
                lines.append("")
    return "\n".join(lines)


def _save_report(report, markdown_output_path, json_output_path):
    rounded_report = _round_nested(report)
    markdown_output_path.parent.mkdir(parents=True, exist_ok=True)
    json_output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(json_output_path, "w") as fp:
        json.dump(rounded_report, fp, indent=2)

    with open(markdown_output_path, "w") as fp:
        fp.write(_build_markdown_report(rounded_report))


def _build_output_paths(args, cfg):
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    root_dir = Path(args.output_dir) if args.output_dir else Path(__file__).resolve().parent / "readme"
    stem = f"profiling_echmo_testbook_{cfg['model']}_{args.load_mode}_{timestamp}"
    markdown_output_path = Path(args.output_markdown) if args.output_markdown else root_dir / f"{stem}.md"
    json_output_path = Path(args.output_json) if args.output_json else root_dir / f"{stem}.json"
    return markdown_output_path, json_output_path, timestamp


def _collect_metadata(cfg, args, device, timestamp, model, checkpoint_path, adapter):
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    else:
        device_name = platform.processor() or "CPU"

    return {
        "timestamp": timestamp,
        "config": str(Path(args.cfg).resolve()),
        "model_name": cfg["model"],
        "dataset": cfg["dataset"],
        "load_mode": args.load_mode,
        "checkpoint_path": str(checkpoint_path) if checkpoint_path else None,
        "device": str(device),
        "device_name": device_name,
        "distributed": bool(getattr(args, "distributed", False)),
        "rank": get_rank(),
        "world_size": get_world_size(),
        "data_mode": args.data_mode,
        "batch_size": args.batch_size,
        "batch_size_per_rank": args.batch_size,
        "global_batch_size": args.batch_size * get_world_size(),
        "warmup_iters": args.warmup_iters,
        "measure_iters": args.measure_iters,
        "train_lr": args.train_lr,
        "tasks": args.tasks,
        "levels": args.levels,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "trainable_params": count_params(model),
        "stage_adapter": adapter.source if adapter else "unsupported",
    }


def _profile_requested_levels(model, adapter, model_input, args, device, lr, initial_state):
    results = {}

    for task_name in args.tasks:
        task_results = {}
        for level_name in args.levels:
            model.load_state_dict(initial_state)
            _maybe_empty_cache(device)
            if level_name == "end2end":
                if task_name == "infer":
                    task_results[level_name] = _profile_end2end_infer(
                        model,
                        model_input,
                        args.warmup_iters,
                        args.measure_iters,
                        device,
                    )
                else:
                    task_results[level_name] = _profile_end2end_train(
                        model,
                        model_input,
                        args.warmup_iters,
                        args.measure_iters,
                        device,
                        lr,
                    )
                continue

            if adapter is None:
                task_results[level_name] = {
                    "status": "unsupported",
                    "reason": "Canonical EchMo stage interface is unavailable for this model variant.",
                }
                continue

            if level_name == "module-wise":
                if task_name == "infer":
                    task_results[level_name] = _profile_module_wise_infer(
                        adapter,
                        model_input,
                        args.warmup_iters,
                        args.measure_iters,
                        device,
                    )
                else:
                    task_results[level_name] = _profile_module_wise_train(
                        adapter,
                        model_input,
                        args.warmup_iters,
                        args.measure_iters,
                        device,
                    )
                continue

            if level_name == "front-back":
                if task_name == "infer":
                    task_results[level_name] = _profile_front_back_infer(
                        adapter,
                        model_input,
                        args.warmup_iters,
                        args.measure_iters,
                        device,
                    )
                else:
                    task_results[level_name] = _profile_front_back_train(
                        adapter,
                        model_input,
                        args.warmup_iters,
                        args.measure_iters,
                        device,
                    )

        results[task_name] = task_results

    return results


def _build_arg_parser():
    parser = argparse.ArgumentParser(description="Profile EchMo family latency and memory usage.")
    parser.add_argument("--cfg", type=str, required=True, help="Path to an EchMo family config JSON.")
    parser.add_argument("--dist-url", default="env://", help="URL used to set up distributed profiling.")
    parser.add_argument(
        "--load-mode",
        type=str,
        default="scratch",
        choices=["scratch", "pretrained"],
        help="Use scratch initialization or load a pretrained checkpoint.",
    )
    parser.add_argument("--model-path", type=str, default="", help="Optional explicit checkpoint path.")
    parser.add_argument(
        "--checkpoint-epoch",
        type=int,
        default=None,
        help="Optional checkpoint epoch under the experiment results folder.",
    )
    parser.add_argument("--device", type=str, default="", help="Optional device override, e.g. cuda or cpu.")
    parser.add_argument(
        "--data-mode",
        type=str,
        default=None,
        help="Input construction mode: random or config. Defaults to profile_data_mode in config, then random.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Profiling batch size. Defaults to profile_batch_size in config, then batch_size in config.",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=None,
        help="Warmup iterations. Defaults to profile_warmup_iters in config, then 3.",
    )
    parser.add_argument(
        "--measure-iters",
        type=int,
        default=None,
        help="Measured iterations. Defaults to profile_measure_iters in config, then 10.",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="train,infer",
        help="Comma-separated list from: train,infer.",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="module-wise,end2end,front-back",
        help="Comma-separated list from: module-wise,end2end,front-back.",
    )
    parser.add_argument(
        "--train-lr",
        type=float,
        default=None,
        help="Train profiling learning rate. Defaults to profile_train_lr in config, then lr in config.",
    )
    parser.add_argument("--output-dir", type=str, default="", help="Directory for result files.")
    parser.add_argument("--output-markdown", type=str, default="", help="Optional explicit markdown output path.")
    parser.add_argument("--output-json", type=str, default="", help="Optional explicit json output path.")
    return parser


def main():
    parser = _build_arg_parser()
    args = parser.parse_args()
    args.tasks = _parse_csv_set(args.tasks, SUPPORTED_TASKS, "task")
    args.levels = _parse_csv_set(args.levels, SUPPORTED_LEVELS, "level")
    init_distributed_mode(args)

    cfg = _load_config(args.cfg)
    runtime_settings = _resolve_runtime_settings(cfg, args)
    args.data_mode = runtime_settings["data_mode"]
    args.batch_size = runtime_settings["batch_size"]
    args.warmup_iters = runtime_settings["warmup_iters"]
    args.measure_iters = runtime_settings["measure_iters"]
    args.train_lr = runtime_settings["train_lr"]

    if getattr(args, "distributed", False):
        device = torch.device("cuda", args.gpu)
        device_note = None
        cfg["device"] = f"cuda:{args.gpu}"
    else:
        device, device_note = _resolve_device(cfg, args.device)
        cfg["device"] = str(device)
    cfg["distributed"] = bool(getattr(args, "distributed", False))

    notes = []
    if device_note:
        notes.append(device_note)
    if getattr(args, "distributed", False):
        notes.append(
            "DDP mode is enabled. Reported latencies and memory numbers are averaged across ranks; batch_size is per-rank and global_batch_size equals batch_size * world_size."
        )

    model, checkpoint_path = _load_model_for_profiling(cfg, args, notes)
    model.to(device)
    if getattr(args, "distributed", False):
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    adapter = _build_stage_adapter(model)
    if adapter is None and any(level in {"module-wise", "front-back"} for level in args.levels):
        notes.append(
            "Module-wise and front-back benchmarking require the canonical EchMo stage interface; unsupported levels are recorded explicitly."
        )
    elif adapter is not None and adapter.source != "direct":
        notes.append(f"Stage-wise profiling is routed through the model's {adapter.source} EchMo backbone.")

    if args.data_mode == "config":
        notes.append(
            "Data mode 'config' builds a deterministic config-driven input tensor from profiling shape and scale settings instead of sampling random noise."
        )

    model_input = _create_profile_input(cfg, args.batch_size, device, args.data_mode)
    lr = args.train_lr
    initial_state = _clone_state_dict(model)

    markdown_output_path, json_output_path, timestamp = _build_output_paths(args, cfg)
    results = _profile_requested_levels(model, adapter, model_input, args, device, lr, initial_state)
    if getattr(args, "distributed", False):
        results = _distributed_reduce_nested(results)

    report = {
        "metadata": _collect_metadata(cfg, args, device, timestamp, model, checkpoint_path, adapter),
        "notes": notes,
        "results": results,
    }
    if is_main_process():
        _save_report(report, markdown_output_path, json_output_path)

    if is_main_process():
        print(f"Markdown report saved to: {markdown_output_path}")
        print(f"JSON report saved to: {json_output_path}")

    if getattr(args, "distributed", False):
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()