import argparse
import json
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader

from svrss.learners.initializer import Initializer
from svrss.loaders.dataloaders import PeCoRSVDataset
from svrss.models import EchMo, EchMo4MoE, EchMoDeep, EchMoMemo, EchMoWide, rEchMo
from svrss.utils.distributed_utils import get_rank, get_world_size, init_distributed_mode
from svrss.utils.functions import define_loss, get_transformations, normalize
from svrss.utils.paths import Paths


def _cuda_sync_if_needed():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _timed_call(fn):
    _cuda_sync_if_needed()
    start = time.perf_counter()
    result = fn()
    _cuda_sync_if_needed()
    return result, time.perf_counter() - start


def _reduce_scalar(value):
    tensor = torch.tensor([value], dtype=torch.float64, device="cuda" if torch.cuda.is_available() else "cpu")
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= get_world_size()
    return float(tensor.item())


def _build_model_cfg(cfg, data_cfg):
    class EchMoCFG:
        def __init__(self):
            self.task_name = "Segmentation"

    model_cfg = EchMoCFG()
    model_cfg.task_name = data_cfg["task_name"]
    model_cfg.coef_path = "coefs"
    if cfg["dataset"] == "LSFJamming":
        model_cfg.dataset_path = str(Paths().get()["LSFJamming"])
    else:
        model_cfg.dataset_path = "/data/PeCoLab/datasets/NuaaSim"
    model_cfg.n_pulse = data_cfg["w_size"]
    model_cfg.n_range = data_cfg["h_size"]
    model_cfg.n_classes = data_cfg["nb_classes"]
    model_cfg.n_frames = data_cfg["nb_input_channels"]
    model_cfg.scale = data_cfg["norm_type"]
    model_cfg.lmf_chains = data_cfg["lmf_chains"]
    model_cfg.lmf_channels = data_cfg["lmf_channels"]
    model_cfg.lmf_layers = data_cfg["lmf_layers"]
    model_cfg.lmf_coefs = data_cfg["lmf_coefs"]
    model_cfg.lmf_impl = data_cfg.get("lmf_impl", "v6")
    model_cfg.lmf_chain_impl = data_cfg.get("lmf_chain_impl", "serial")
    model_cfg.ldc_channels = data_cfg["ldc_channels"]
    model_cfg.ldc_orders = data_cfg["ldc_orders"]
    model_cfg.mldc_impl = data_cfg.get("mldc_impl", "serial")
    model_cfg.df_channels = data_cfg["nb_doppler"]
    model_cfg.gdf_impl = data_cfg.get("gdf_impl", "conv")
    model_cfg.activation = data_cfg["activation"]
    model_cfg.bias = data_cfg["bias"]
    model_cfg.wide_groups = data_cfg.get("wide_groups", 4)
    model_cfg.wide_rank = data_cfg.get("wide_rank", 32)
    model_cfg.wide_share_mldc = data_cfg.get("wide_share_mldc", True)
    model_cfg.wide_fuse_residual = data_cfg.get("wide_fuse_residual", True)
    model_cfg.deep_stage2_enable = data_cfg.get("deep_stage2_enable", True)
    model_cfg.deep_stage2_range_scale = data_cfg.get("deep_stage2_range_scale", 0.5)
    model_cfg.deep_stage2_pulse_scale = data_cfg.get("deep_stage2_pulse_scale", 1.0)
    model_cfg.deep_stage2_chain_scale = data_cfg.get("deep_stage2_chain_scale", 1.0)
    model_cfg.deep_stage2_lmf_channel_scale = data_cfg.get("deep_stage2_lmf_channel_scale", 1.0)
    model_cfg.deep_stage2_ldc_channel_scale = data_cfg.get("deep_stage2_ldc_channel_scale", 1.0)
    model_cfg.memo_enable = data_cfg.get("memo_enable", True)
    model_cfg.memo_beta = data_cfg.get("memo_beta", 0.8)
    model_cfg.memo_gamma = data_cfg.get("memo_gamma", 0.4)
    model_cfg.memo_use_state_proj = data_cfg.get("memo_use_state_proj", True)
    model_cfg.moe_num_experts = data_cfg.get("moe_num_experts", 4)
    model_cfg.moe_topk = data_cfg.get("moe_topk", 2)
    model_cfg.moe_gate_hidden = data_cfg.get("moe_gate_hidden", 32)
    model_cfg.moe_temperature = data_cfg.get("moe_temperature", 1.0)
    model_cfg.moe_dropout = data_cfg.get("moe_dropout", 0.0)
    return model_cfg


def _build_model(cfg, model_cfg):
    if cfg["model"] == "echmo":
        return EchMo(configs=model_cfg)
    if cfg["model"] == "echmo_deep":
        return EchMoDeep(configs=model_cfg)
    if cfg["model"] == "echmo_memo":
        return EchMoMemo(configs=model_cfg)
    if cfg["model"] == "echmo_4moe":
        return EchMo4MoE(configs=model_cfg)
    if cfg["model"] == "echmo_wide":
        return EchMoWide(configs=model_cfg)
    if cfg["model"] == "rechmo":
        return rEchMo(configs=model_cfg)
    raise ValueError(f"Unsupported model for profiling: {cfg['model']}")


def _build_optimizer(cfg, net):
    optimizer = optim.Adam(net.parameters(), lr=cfg["lr"])
    if cfg["schedular"] == "exp":
        scheduler = ExponentialLR(optimizer, gamma=0.9)
    elif cfg["schedular"] == "cos":
        scheduler = CosineAnnealingLR(optimizer, T_max=cfg["Tmax"] - 1, eta_min=1e-7)
    else:
        raise KeyError("we only implement schedular of exp and cos")
    return optimizer, scheduler


def _iter_training_frames(data, cfg, add_temp=False):
    train_loader = data["dataloaders"][0]
    transformations = get_transformations(cfg["transformations"].split(","), sizes=(cfg["w_size"], cfg["h_size"]))
    signal_type = "range_doppler"
    epoch = 0
    while True:
        for _, sequence_data in enumerate(train_loader):
            seq_name, seq = sequence_data
            if cfg["dataset"] == "CWRdata":
                path_to_frames = data["paths"]["CWRdata"] / seq_name[0]
            elif cfg["dataset"] == "PDRdata":
                path_to_frames = data["paths"]["PDRdata"] / seq_name[0]
            elif cfg["dataset"] == "XPDdata":
                path_to_frames = data["paths"]["XPDdata"] / seq_name[0]
            elif cfg["dataset"] == "RDSyn":
                path_to_frames = data["paths"]["RDSyn"] / seq_name[0]
            elif cfg["dataset"] == "NuaaSim":
                path_to_frames = data["paths"]["NuaaSim"] / seq_name[0]
            elif cfg["dataset"] == "LSFJamming":
                path_to_frames = data["paths"]["LSFJamming"] / seq_name[0]
            elif cfg["dataset"] in ["Carrada_RA", "Carrada_RD", "CarradaRAC_RA", "CarradaRAC_RD"]:
                path_to_frames = data["paths"]["Carrada"] / seq_name[0]
            else:
                raise KeyError(f"Dataset {cfg['dataset']} has not been supported yet.")

            dataset = PeCoRSVDataset(
                seq,
                cfg["dataset"],
                cfg["annot_type"],
                signal_type,
                path_to_frames,
                cfg["process_signal"],
                cfg["nb_input_channels"],
                cfg["cmplx_flag"],
                transformations,
                add_temp,
            )

            if cfg["distributed"]:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                sampler.set_epoch(epoch)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            batch_sampler = torch.utils.data.BatchSampler(sampler, cfg["batch_size"], drop_last=True)
            frame_loader = DataLoader(dataset, batch_sampler=batch_sampler, num_workers=cfg["num_workers"])
            for frame in frame_loader:
                yield frame
        epoch += 1


def _compute_loss(cfg, criterion, outputs, rd_mask):
    rd_losses = [c(outputs, torch.argmax(rd_mask, axis=1)) for c in criterion]
    return torch.mean(torch.stack(rd_losses))


def _unwrap_model(net):
    return net.module if hasattr(net, "module") else net


def _supports_split_backward(net):
    base_model = _unwrap_model(net)
    return hasattr(base_model, "forward_frontend") and hasattr(base_model, "forward_head")


def _supports_stagewise_frontend_split(net):
    base_model = _unwrap_model(net)
    required = [
        "normalize_input",
        "forward_lmf",
        "forward_mldc",
        "forward_gdf",
        "pack_features",
        "forward_head",
    ]
    return all(hasattr(base_model, name) for name in required)


def _profile_split_backward(net, cfg, criterion, frame_iter, warmup_iters, measure_iters, device):
    base_model = _unwrap_model(net)
    totals = {
        "frontend_forward": 0.0,
        "head_forward": 0.0,
        "head_backward": 0.0,
        "frontend_backward": 0.0,
        "split_backward_total": 0.0,
    }
    total_iters = warmup_iters + measure_iters
    measured = 0

    for step in range(total_iters):
        frame = next(frame_iter)
        if cfg["cmplx_flag"]:
            rd_data = frame["rd_matrix"].to(device)
        else:
            rd_data = frame["rd_matrix"].to(device).float()
        rd_mask = frame["rd_mask"].to(device).float()
        rd_data = normalize(rd_data, cfg["dataset"], "range_doppler", norm_type=cfg["norm_type"])

        base_model.zero_grad(set_to_none=True)
        features, frontend_forward_time = _timed_call(lambda: base_model.forward_frontend(rd_data))
        leaf_features = features.detach().requires_grad_(True)
        head_outputs, head_forward_time = _timed_call(lambda: base_model.forward_head(leaf_features))
        head_loss = _compute_loss(cfg, criterion, head_outputs, rd_mask)
        _, head_backward_time = _timed_call(head_loss.backward)
        feature_grad = leaf_features.grad.detach()

        base_model.zero_grad(set_to_none=True)
        _, frontend_backward_time = _timed_call(lambda: torch.autograd.backward(features, feature_grad))
        split_total = head_backward_time + frontend_backward_time
        base_model.zero_grad(set_to_none=True)

        if step >= warmup_iters:
            totals["frontend_forward"] += frontend_forward_time
            totals["head_forward"] += head_forward_time
            totals["head_backward"] += head_backward_time
            totals["frontend_backward"] += frontend_backward_time
            totals["split_backward_total"] += split_total
            measured += 1

        if get_rank() == 0:
            print(
                f"[split-backward][iter {step + 1}/{total_iters}] frontend_fwd={frontend_forward_time * 1000:.3f}ms "
                f"head_fwd={head_forward_time * 1000:.3f}ms head_bwd={head_backward_time * 1000:.3f}ms "
                f"frontend_bwd={frontend_backward_time * 1000:.3f}ms split_total={split_total * 1000:.3f}ms"
            )

    summary = {k: (v / max(measured, 1)) * 1000.0 for k, v in totals.items()}
    if dist.is_available() and dist.is_initialized():
        summary = {k: _reduce_scalar(v) for k, v in summary.items()}
    return summary


def _profile_frontend_stagewise_backward(net, cfg, criterion, frame_iter, warmup_iters, measure_iters, device):
    base_model = _unwrap_model(net)
    totals = {
        "lmf_forward": 0.0,
        "mldc_forward": 0.0,
        "gdf_forward": 0.0,
        "pack_forward": 0.0,
        "lmf_backward": 0.0,
        "mldc_backward": 0.0,
        "gdf_backward": 0.0,
        "pack_backward": 0.0,
        "frontend_stagewise_backward_total": 0.0,
    }
    total_iters = warmup_iters + measure_iters
    measured = 0

    for step in range(total_iters):
        frame = next(frame_iter)
        if cfg["cmplx_flag"]:
            rd_data = frame["rd_matrix"].to(device)
        else:
            rd_data = frame["rd_matrix"].to(device).float()
        rd_mask = frame["rd_mask"].to(device).float()

        x_norm = base_model.normalize_input(normalize(rd_data, cfg["dataset"], "range_doppler", norm_type=cfg["norm_type"]))
        x_k, lmf_forward_time = _timed_call(lambda: base_model.forward_lmf(x_norm))
        x_h, mldc_forward_time = _timed_call(lambda: base_model.forward_mldc(x_k))
        x_d, gdf_forward_time = _timed_call(lambda: base_model.forward_gdf(x_h))
        packed, pack_forward_time = _timed_call(lambda: base_model.pack_features(x_d))

        packed_leaf = packed.detach().requires_grad_(True)
        head_outputs = base_model.forward_head(packed_leaf)
        head_loss = _compute_loss(cfg, criterion, head_outputs, rd_mask)
        head_loss.backward()
        grad_packed = packed_leaf.grad.detach()
        base_model.zero_grad(set_to_none=True)

        x_d_leaf = x_d.detach().requires_grad_(True)
        packed_from_x_d, _ = _timed_call(lambda: base_model.pack_features(x_d_leaf))
        _, pack_backward_time = _timed_call(lambda: torch.autograd.backward(packed_from_x_d, grad_packed))
        grad_x_d = x_d_leaf.grad.detach()

        x_h_leaf = x_h.detach().requires_grad_(True)
        x_d_from_x_h, _ = _timed_call(lambda: base_model.forward_gdf(x_h_leaf))
        _, gdf_backward_time = _timed_call(lambda: torch.autograd.backward(x_d_from_x_h, grad_x_d))
        grad_x_h = x_h_leaf.grad.detach()

        x_k_leaf = x_k.detach().requires_grad_(True)
        x_h_from_x_k, _ = _timed_call(lambda: base_model.forward_mldc(x_k_leaf))
        _, mldc_backward_time = _timed_call(lambda: torch.autograd.backward(x_h_from_x_k, grad_x_h))
        grad_x_k = x_k_leaf.grad.detach()

        x_norm_leaf = x_norm.detach().requires_grad_(True)
        x_k_from_x_norm, _ = _timed_call(lambda: base_model.forward_lmf(x_norm_leaf))
        _, lmf_backward_time = _timed_call(lambda: torch.autograd.backward(x_k_from_x_norm, grad_x_k))
        stagewise_total = lmf_backward_time + mldc_backward_time + gdf_backward_time + pack_backward_time
        base_model.zero_grad(set_to_none=True)

        if step >= warmup_iters:
            totals["lmf_forward"] += lmf_forward_time
            totals["mldc_forward"] += mldc_forward_time
            totals["gdf_forward"] += gdf_forward_time
            totals["pack_forward"] += pack_forward_time
            totals["lmf_backward"] += lmf_backward_time
            totals["mldc_backward"] += mldc_backward_time
            totals["gdf_backward"] += gdf_backward_time
            totals["pack_backward"] += pack_backward_time
            totals["frontend_stagewise_backward_total"] += stagewise_total
            measured += 1

        if get_rank() == 0:
            print(
                f"[frontend-stagewise][iter {step + 1}/{total_iters}] "
                f"lmf_bwd={lmf_backward_time * 1000:.3f}ms mldc_bwd={mldc_backward_time * 1000:.3f}ms "
                f"gdf_bwd={gdf_backward_time * 1000:.3f}ms pack_bwd={pack_backward_time * 1000:.3f}ms "
                f"total={stagewise_total * 1000:.3f}ms"
            )

    summary = {k: (v / max(measured, 1)) * 1000.0 for k, v in totals.items()}
    if dist.is_available() and dist.is_initialized():
        summary = {k: _reduce_scalar(v) for k, v in summary.items()}
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--warmup-iters", type=int, default=3)
    parser.add_argument("--measure-iters", type=int, default=10)
    parser.add_argument("--dist-url", default="env://")
    parser.add_argument("--output-json", type=str, default="")
    parser.add_argument("--split-output-json", type=str, default="")
    parser.add_argument("--frontend-stage-output-json", type=str, default="")
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    args = parser.parse_args()

    init_distributed_mode(args)

    with open(args.cfg, "r") as fp:
        cfg = json.load(fp)

    if args.batch_size > 0:
        cfg["batch_size"] = args.batch_size
    if args.num_workers >= 0:
        cfg["num_workers"] = args.num_workers

    cfg["distributed"] = args.distributed
    if cfg["dataset"] in ["CWRdata", "Carrada_RA", "Carrada_RD", "CarradaRAC_RD", "CarradaRAC_RA"]:
        cfg["nb_classes"] = 4
    elif cfg["dataset"] == "PDRdata":
        cfg["nb_classes"] = 5
    elif cfg["dataset"] in ["XPDdata", "RDSyn", "NuaaSim"]:
        cfg["nb_classes"] = 2
    elif cfg["dataset"] == "LSFJamming":
        cfg["nb_classes"] = 7
    else:
        raise KeyError(f"Dataset {cfg['dataset']} has not been supported yet.")

    torch.manual_seed(cfg["torch_seed"])
    device = torch.device(cfg["device"])

    init = Initializer(cfg)
    data = init.get_data()
    model_cfg = _build_model_cfg(cfg, data["cfg"])
    net = _build_model(cfg, model_cfg)
    net.to(device)
    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net)
    net.train()

    optimizer, _ = _build_optimizer(cfg, net)
    criterion = define_loss(cfg["dataset"], "range_doppler", cfg["custom_loss"], cfg["device"], cfg.get("focal_gamma"))
    frame_iter = _iter_training_frames(data, cfg, add_temp=False)

    totals = {
        "data": 0.0,
        "forward": 0.0,
        "loss": 0.0,
        "backward": 0.0,
        "optimizer": 0.0,
        "ddp_sync": 0.0,
        "step_total": 0.0,
    }

    total_iters = args.warmup_iters + args.measure_iters
    measured = 0
    for step in range(total_iters):
        step_start = time.perf_counter()

        def _load_and_prepare():
            frame = next(frame_iter)
            if cfg["cmplx_flag"]:
                rd_data = frame["rd_matrix"].to(device)
            else:
                rd_data = frame["rd_matrix"].to(device).float()
            rd_mask = frame["rd_mask"].to(device).float()
            rd_data_local = normalize(rd_data, cfg["dataset"], "range_doppler", norm_type=cfg["norm_type"])
            return rd_data_local, rd_mask

        (rd_data, rd_mask), data_time = _timed_call(_load_and_prepare)

        optimizer.zero_grad()
        outputs, forward_time = _timed_call(lambda: net(rd_data))
        loss, loss_time = _timed_call(lambda: _compute_loss(cfg, criterion, outputs, rd_mask))
        _, backward_time = _timed_call(loss.backward)
        _, optimizer_time = _timed_call(optimizer.step)
        step_total = time.perf_counter() - step_start

        ddp_sync_time = 0.0
        if args.distributed:
            optimizer.zero_grad()
            probe_outputs = net(rd_data)
            probe_loss = _compute_loss(cfg, criterion, probe_outputs, rd_mask)
            _, backward_no_sync_time = _timed_call(lambda: _backward_no_sync(net, probe_loss))
            ddp_sync_time = max(0.0, backward_time - backward_no_sync_time)

        optimizer.zero_grad()

        if step >= args.warmup_iters:
            totals["data"] += data_time
            totals["forward"] += forward_time
            totals["loss"] += loss_time
            totals["backward"] += backward_time
            totals["optimizer"] += optimizer_time
            totals["ddp_sync"] += ddp_sync_time
            totals["step_total"] += step_total
            measured += 1

        if get_rank() == 0:
            print(
                f"[profile][iter {step + 1}/{total_iters}] data={data_time * 1000:.3f}ms "
                f"forward={forward_time * 1000:.3f}ms loss={loss_time * 1000:.3f}ms "
                f"backward={backward_time * 1000:.3f}ms opt={optimizer_time * 1000:.3f}ms "
                f"ddp_sync_est={ddp_sync_time * 1000:.3f}ms total={step_total * 1000:.3f}ms"
            )

    summary = {k: (v / max(measured, 1)) * 1000.0 for k, v in totals.items()}
    summary = {k: _reduce_scalar(v) for k, v in summary.items()}
    summary["world_size"] = get_world_size()
    summary["warmup_iters"] = args.warmup_iters
    summary["measure_iters"] = args.measure_iters
    summary["config"] = args.cfg
    summary["batch_size_per_rank"] = cfg["batch_size"]
    summary["num_workers"] = cfg["num_workers"]
    summary["notes"] = {
        "data": "includes nested dataset/dataloader creation, batch fetch, host-to-device copy, and normalize()",
        "backward": "normal backward timing on the real training step; in DDP this includes overlapped gradient synchronization",
        "ddp_sync": "estimated as backward_with_sync - backward_no_sync on the same batch; approximate communication overhead, not additive with backward",
    }

    split_summary = None
    if _supports_split_backward(net):
        split_frame_iter = _iter_training_frames(data, cfg, add_temp=False)
        split_summary = _profile_split_backward(
            net,
            cfg,
            criterion,
            split_frame_iter,
            args.warmup_iters,
            args.measure_iters,
            device,
        )
        split_summary["world_size"] = get_world_size()
        split_summary["warmup_iters"] = args.warmup_iters
        split_summary["measure_iters"] = args.measure_iters
        split_summary["config"] = args.cfg
        split_summary["batch_size_per_rank"] = cfg["batch_size"]
        split_summary["num_workers"] = cfg["num_workers"]
        split_summary["notes"] = {
            "frontend_forward": "forward through normalization + LMF + MLDC + GDF + pack",
            "head_forward": "forward through RDTranslator only",
            "head_backward": "backward from loss to RDTranslator params and packed frontend feature tensor only",
            "frontend_backward": "backward from packed frontend feature gradient through EchMo frontend only",
            "split_backward_total": "head_backward + frontend_backward in the decomposed profiling harness",
        }

    frontend_stage_summary = None
    if _supports_stagewise_frontend_split(net):
        stage_frame_iter = _iter_training_frames(data, cfg, add_temp=False)
        frontend_stage_summary = _profile_frontend_stagewise_backward(
            net,
            cfg,
            criterion,
            stage_frame_iter,
            args.warmup_iters,
            args.measure_iters,
            device,
        )
        frontend_stage_summary["world_size"] = get_world_size()
        frontend_stage_summary["warmup_iters"] = args.warmup_iters
        frontend_stage_summary["measure_iters"] = args.measure_iters
        frontend_stage_summary["config"] = args.cfg
        frontend_stage_summary["batch_size_per_rank"] = cfg["batch_size"]
        frontend_stage_summary["num_workers"] = cfg["num_workers"]
        frontend_stage_summary["notes"] = {
            "lmf_backward": "backward through LMF only, using the gradient at the LMF output boundary",
            "mldc_backward": "backward through MLDC only, using the gradient at the MLDC output boundary",
            "gdf_backward": "backward through GDF only, using the gradient at the GDF output boundary",
            "pack_backward": "backward through the pack operation only, from packed real-valued features to complex GDF output",
            "frontend_stagewise_backward_total": "lmf_backward + mldc_backward + gdf_backward + pack_backward in the staged harness",
        }

    if get_rank() == 0:
        print("\n=== EchMo Train Stage Profile Summary (ms/iter) ===")
        for key in ["data", "forward", "loss", "backward", "optimizer", "ddp_sync", "step_total"]:
            print(f"{key:>10}: {summary[key]:.3f}")
        if args.output_json:
            with open(args.output_json, "w") as fp:
                json.dump(summary, fp, indent=2)
            print(f"Saved summary to {args.output_json}")
        if split_summary is not None:
            print("\n=== EchMo Split Backward Summary (ms/iter) ===")
            for key in ["frontend_forward", "head_forward", "head_backward", "frontend_backward", "split_backward_total"]:
                print(f"{key:>18}: {split_summary[key]:.3f}")
            if args.split_output_json:
                with open(args.split_output_json, "w") as fp:
                    json.dump(split_summary, fp, indent=2)
                print(f"Saved split summary to {args.split_output_json}")
        if frontend_stage_summary is not None:
            print("\n=== EchMo Frontend Stagewise Backward Summary (ms/iter) ===")
            for key in ["lmf_forward", "mldc_forward", "gdf_forward", "pack_forward", "lmf_backward", "mldc_backward", "gdf_backward", "pack_backward", "frontend_stagewise_backward_total"]:
                print(f"{key:>30}: {frontend_stage_summary[key]:.3f}")
            if args.frontend_stage_output_json:
                with open(args.frontend_stage_output_json, "w") as fp:
                    json.dump(frontend_stage_summary, fp, indent=2)
                print(f"Saved frontend stage summary to {args.frontend_stage_output_json}")

    if args.distributed:
        dist.barrier()
        dist.destroy_process_group()


def _backward_no_sync(net, loss):
    if hasattr(net, "no_sync"):
        with net.no_sync():
            loss.backward()
    else:
        loss.backward()


if __name__ == "__main__":
    main()