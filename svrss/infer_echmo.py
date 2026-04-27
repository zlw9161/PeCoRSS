"""Inference entrypoint for EchMo model family."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from svrss.learners.tester import Tester
from svrss.loaders.dataloaders import SequenceDataset
from svrss.loaders.dataset import Carrada, CWRdata, LSFJamming, NuaaSim, PDRdata, RDSyn, XPDdata
from svrss.models import EchMo, EchMo4MoE, EchMoDeep, EchMoMemo, EchMoReCK, EchMoWide, rEchMo
from svrss.utils.distributed_utils import init_distributed_mode
from svrss.utils.functions import count_params
from svrss.utils.paths import Paths


def _resolve_num_classes(dataset_name: str) -> int:
    if dataset_name in ["CWRdata", "Carrada_RA", "Carrada_RD", "CarradaRAC_RD", "CarradaRAC_RA"]:
        return 4
    if dataset_name == "PDRdata":
        return 5
    if dataset_name in ["XPDdata", "RDSyn", "NuaaSim"]:
        return 2
    if dataset_name == "LSFJamming":
        return 7
    raise KeyError("Dataset {} has not been supported yet.".format(dataset_name))


def _build_model_cfg(cfg: dict) -> SimpleNamespace:
    model_cfg = SimpleNamespace()
    model_cfg.task_name = cfg["task_name"]
    model_cfg.coef_path = "coefs"
    if cfg["dataset"] == "LSFJamming":
        model_cfg.dataset_path = str(Paths().get()["LSFJamming"])
    else:
        model_cfg.dataset_path = str(Paths().get()["NuaaSim"])

    model_cfg.n_pulse = cfg["w_size"]
    model_cfg.n_range = cfg["h_size"]
    model_cfg.n_classes = cfg["nb_classes"]
    model_cfg.n_frames = cfg["nb_input_channels"]
    model_cfg.scale = cfg["norm_type"]

    model_cfg.lmf_chains = cfg["lmf_chains"]
    model_cfg.lmf_channels = cfg["lmf_channels"]
    model_cfg.lmf_layers = cfg["lmf_layers"]
    model_cfg.lmf_coefs = cfg["lmf_coefs"]
    model_cfg.lmf_impl = cfg.get("lmf_impl", "v6")
    model_cfg.lmf_chain_impl = cfg.get("lmf_chain_impl", "serial")

    model_cfg.ldc_channels = cfg["ldc_channels"]
    model_cfg.ldc_orders = cfg["ldc_orders"]
    model_cfg.mldc_impl = cfg.get("mldc_impl", "serial")

    model_cfg.df_channels = cfg["nb_doppler"]
    model_cfg.gdf_impl = cfg.get("gdf_impl", "conv")
    model_cfg.pack_feat = cfg.get("pack_feat", "apiq")

    model_cfg.activation = cfg["activation"]
    model_cfg.bias = cfg["bias"]

    model_cfg.wide_groups = cfg.get("wide_groups", 4)
    model_cfg.wide_rank = cfg.get("wide_rank", 32)
    model_cfg.wide_share_mldc = cfg.get("wide_share_mldc", True)
    model_cfg.wide_fuse_residual = cfg.get("wide_fuse_residual", True)

    model_cfg.deep_stage2_enable = cfg.get("deep_stage2_enable", True)
    model_cfg.deep_stage2_range_scale = cfg.get("deep_stage2_range_scale", 0.5)
    model_cfg.deep_stage2_pulse_scale = cfg.get("deep_stage2_pulse_scale", 1.0)
    model_cfg.deep_stage2_chain_scale = cfg.get("deep_stage2_chain_scale", 1.0)
    model_cfg.deep_stage2_lmf_channel_scale = cfg.get("deep_stage2_lmf_channel_scale", 1.0)
    model_cfg.deep_stage2_ldc_channel_scale = cfg.get("deep_stage2_ldc_channel_scale", 1.0)

    model_cfg.memo_enable = cfg.get("memo_enable", True)
    model_cfg.memo_beta = cfg.get("memo_beta", 0.8)
    model_cfg.memo_gamma = cfg.get("memo_gamma", 0.4)
    model_cfg.memo_use_state_proj = cfg.get("memo_use_state_proj", True)

    model_cfg.moe_num_experts = cfg.get("moe_num_experts", 4)
    model_cfg.moe_topk = cfg.get("moe_topk", 2)
    model_cfg.moe_gate_hidden = cfg.get("moe_gate_hidden", 32)
    model_cfg.moe_temperature = cfg.get("moe_temperature", 1.0)
    model_cfg.moe_dropout = cfg.get("moe_dropout", 0.0)

    model_cfg.reck_mode = cfg.get("reck_mode", cfg.get("ReCK_MODE", "reck2"))
    model_cfg.reck_modules = cfg.get("reck_modules", cfg.get("ReCK_Module", "all"))
    model_cfg.reck_alpha = cfg.get("reck_alpha", cfg.get("ReCK_Alpha", 0.3))
    model_cfg.reck_alpha_lmf = cfg.get("reck_alpha_lmf", cfg.get("ReCK_Alpha_LMF", model_cfg.reck_alpha))
    model_cfg.reck_alpha_mldc = cfg.get("reck_alpha_mldc", cfg.get("ReCK_Alpha_MLDC", model_cfg.reck_alpha))
    model_cfg.reck_alpha_gdf = cfg.get("reck_alpha_gdf", cfg.get("ReCK_Alpha_GDF", model_cfg.reck_alpha))
    return model_cfg


def _build_model(cfg: dict):
    model_cfg = _build_model_cfg(cfg)
    model_name = cfg["model"]
    if model_name == "echmo":
        return EchMo(configs=model_cfg)
    if model_name == "echmo_deep":
        return EchMoDeep(configs=model_cfg)
    if model_name == "echmo_memo":
        return EchMoMemo(configs=model_cfg)
    if model_name == "echmo_4moe":
        return EchMo4MoE(configs=model_cfg)
    if model_name == "echmo_reck":
        return EchMoReCK(configs=model_cfg)
    if model_name == "echmo_wide":
        return EchMoWide(configs=model_cfg)
    if model_name == "rechmo":
        return rEchMo(configs=model_cfg)
    raise ValueError("model {} is not supported in infer_echmo.py.".format(model_name))


def _normalize_input_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path
    root_candidate = Path("/") / path
    if root_candidate.exists():
        return root_candidate
    return path.resolve()


def _get_default_name_exp(cfg: dict) -> str:
    exp_desc = str(cfg.get("exp_desc", "regular")).strip() or "regular"
    exp_desc = exp_desc.replace(" ", "-")
    return (
        f"{cfg['model']}_"
        f"e{cfg['nb_epochs']}_"
        f"lr{cfg['lr']}_"
        f"{cfg['custom_loss']}_"
        f"{exp_desc}"
    )


def _resolve_model_path(cfg: dict, args, exp_path: Path) -> Path:
    candidates = []
    if args.model_path:
        candidates.append(_normalize_input_path(args.model_path))

    checkpoint_path = cfg.get("checkpoint_path")
    if checkpoint_path:
        candidates.append(_normalize_input_path(checkpoint_path))

    candidates.append(exp_path / "results" / f"{cfg['model']}.pt")

    if args.checkpoint_epoch is not None:
        candidates.append(exp_path / "results" / f"{cfg['model']}_checkpoint_epoch{args.checkpoint_epoch}.pt")

    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate

    checkpoint_files = sorted(
        (exp_path / "results").glob(f"{cfg['model']}_checkpoint_epoch*.pt"),
        key=lambda x: x.stat().st_mtime,
        reverse=True,
    )
    if checkpoint_files:
        return checkpoint_files[0]

    candidate_info = "\n".join(str(p) for p in candidates)
    raise FileNotFoundError(
        "Could not find model weights. Checked:\n{}".format(candidate_info)
    )


def _load_dataset(dataset_name: str):
    if dataset_name in ["Carrada_RA", "Carrada_RD", "CarradaRAC_RA", "CarradaRAC_RD"]:
        return Carrada()
    if dataset_name == "CWRdata":
        return CWRdata()
    if dataset_name == "PDRdata":
        return PDRdata()
    if dataset_name == "RDSyn":
        return RDSyn()
    if dataset_name == "XPDdata":
        return XPDdata()
    if dataset_name == "NuaaSim":
        return NuaaSim()
    if dataset_name == "LSFJamming":
        return LSFJamming()
    raise KeyError("Dataset {} has not been supported yet.".format(dataset_name))


def _load_state_dict(model, model_path: Path):
    saved_model = torch.load(model_path, map_location=torch.device("cpu"))
    if isinstance(saved_model, dict) and "state_dict" in saved_model:
        saved_model = saved_model["state_dict"]

    first_key = next(iter(saved_model))
    if first_key.startswith("module."):
        new_state_dict = {}
        for key, value in saved_model.items():
            name = key[7:] if key.startswith("module.") else key
            new_state_dict[name] = value
        model.load_state_dict(new_state_dict)
        return
    model.load_state_dict(saved_model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Path to config file.", type=str, required=True)
    parser.add_argument("--model-path", default="", help="Optional absolute/relative model checkpoint path.", type=str)
    parser.add_argument(
        "--checkpoint-epoch",
        default=None,
        type=int,
        help="Pick specific checkpoint epoch under results folder (e.g. 40).",
    )
    parser.add_argument("--dist-url", default="env://", help="url used to set up distributed training")
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action="store_true")
    parser.add_argument("--no-quali", action="store_true", help="Disable qualitative result export during testing")
    args = parser.parse_args()

    init_distributed_mode(args)

    with open(args.cfg, "r") as fp:
        cfg = json.load(fp)

    cfg["distributed"] = args.distributed
    cfg["nb_classes"] = _resolve_num_classes(cfg["dataset"])
    device = torch.device(cfg["device"])

    paths = Paths().get()
    name_exp = cfg.get("name_exp", _get_default_name_exp(cfg))
    version = cfg.get("version", 1)
    exp_path = paths["logs"] / cfg["dataset"] / cfg["model"] / f"{name_exp}_{version}"

    model = _build_model(cfg)
    print("Number of trainable parameters in the model: {}".format(count_params(model)))
    print("Model Name: {}".format(cfg["model"]))
    print("DataSet Name: {}".format(cfg["dataset"]))

    model_path = _resolve_model_path(cfg, args, exp_path)
    print("Loading model from: {}".format(model_path))
    _load_state_dict(model, model_path)

    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    tester = Tester(cfg)
    tester.set_annot_type(cfg["annot_type"])

    data = _load_dataset(cfg["dataset"])
    test = data.get("Test")
    testset = SequenceDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)

    test_metrics = tester.predict(model, seq_testloader, get_quali=not args.no_quali, add_temp=False)

    if cfg["dataset"] in ["Carrada_RA", "CarradaRAC_RA"]:
        signal_type = "range_angle"
    else:
        signal_type = "range_doppler"

    print(">>>>Radar Segmentation Results<<<<:")
    print("Test Prec: ({}) = {}".format(signal_type, test_metrics[signal_type]["prec"]))
    print("Test mIoU: ({}) = {}".format(signal_type, test_metrics[signal_type]["miou"]))
    print("Test mIoU by class: ({}) = {}".format(signal_type, test_metrics[signal_type]["miou_by_class"]))
    print("Test Dice: ({}) = {}".format(signal_type, test_metrics[signal_type]["dice"]))
    print("Test Dice by class: ({}) = {}".format(signal_type, test_metrics[signal_type]["dice_by_class"]))
    print("Confusion matrix: ({}) =".format(signal_type))
    for i in range(cfg["nb_classes"]):
        print("{}".format(test_metrics[signal_type]["confusion_matrix"][i]))


if __name__ == "__main__":
    main()