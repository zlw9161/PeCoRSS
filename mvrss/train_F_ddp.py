"""DDP entrypoint to train models (Linux).

Usage (recommended):
  torchrun --standalone --nproc_per_node 4 train_F_ddp.py --cfg ./config_files/tfoc_F.json

Notes:
- Assumes CUDA + NCCL backend.
- Only rank0 prints/saves/validates by default (see model_F_ddp.py).
"""

import argparse
import json
import os

# Reduce allocator fragmentation risk in long-running jobs.
# Users can still override this from the shell environment.
os.environ.setdefault(
    "PYTORCH_CUDA_ALLOC_CONF",
    "max_split_size_mb:128,garbage_collection_threshold:0.8",
)

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Conservative cuDNN settings (helps avoid rare CUDNN_STATUS_* issues around BN3d).
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = False
try:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
except Exception:
    pass

from mvrss.utils.functions import count_params
from learners.initializer_ddp import InitializerDDP
from learners.model_F_ddp import ModelDDP
from mvrss.models import MVNet, MVANet, MVA_DCN, TMVA_DCN, TMVA_TDC, PKCIn, PKCOn, AdaPKC, TFocNet, VitNet, TMVANet


def _setup_dist():
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=os.environ.get("TFOC_DDP_BACKEND", "nccl"))


def _rank() -> int:
    return dist.get_rank() if dist.is_initialized() else 0


def _local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", "0"))


def _is_rank0() -> bool:
    return _rank() == 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Path to config file.", default="config.json")
    args = parser.parse_args()

    cfg_path = args.cfg
    with open(cfg_path, "r") as fp:
        cfg = json.load(fp)

    _setup_dist()
    local_rank = _local_rank()
    torch.cuda.set_device(local_rank)
    if _is_rank0():
        print(f"PYTORCH_CUDA_ALLOC_CONF={os.environ.get('PYTORCH_CUDA_ALLOC_CONF', '')}", flush=True)

    # Force single-process per GPU.
    device = torch.device("cuda", local_rank)
    cfg["device"] = "cuda"
    cfg["gpus"] = 1  # DDP uses 1 GPU per process

    init = InitializerDDP(cfg)
    data = init.get_data()

    # Build model
    if cfg["model"] == "mvnet":
        net = MVNet(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = False
    elif cfg["model"] == "mvanet":
        net = MVANet(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "mva_dcn":
        net = MVA_DCN(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "tmva_dcn":
        net = TMVA_DCN(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "tmva_tdc":
        net = TMVA_TDC(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "pkcin":
        net = PKCIn(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "adapkc":
        net = AdaPKC(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "pkcon":
        net = PKCOn(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True
    elif cfg["model"] == "tfoc":
        tfoc_variant = str(cfg.get("tfoc_model_variant", "")).strip().lower()
        if tfoc_variant in ("att4_v2", "att4-v2", "block", "block_att4_v2", "tfoc_block"):
            from mvrss.models.tfoc_vanilla_maskdraw_light_block_clutter_att4_v2 import TFocNet as TFocBlockNet
            net = TFocBlockNet(
                n_classes=data["cfg"]["nb_classes"],
                n_frames=data["cfg"]["nb_input_channels"],
                search_size=(int(cfg.get("search_size_rd", 3)), int(cfg.get("search_size_ra", 3))),
                zero_doppler_filter=False,
            )
        else:
            net = TFocNet(
                n_classes=data["cfg"]["nb_classes"],
                n_frames=data["cfg"]["nb_input_channels"],
                signal_type=data["cfg"]["signal_type"],
                zero_doppler_filter=False,
            )
        add_temp = True
    elif cfg["model"] == "vit":
        net = VitNet(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"], signal_type=data["cfg"]["signal_type"])
        add_temp = True
    else:
        net = TMVANet(n_classes=data["cfg"]["nb_classes"], n_frames=data["cfg"]["nb_input_channels"])
        add_temp = True

    if _is_rank0():
        print("Number of trainable parameters in the model: %s" % str(count_params(net)), flush=True)

    # Optional SyncBN (recommended when per-GPU batch is small)
    if bool(cfg.get("sync_bn", False)):
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    net = net.to(device)
    net = DDP(net, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=bool(cfg.get("find_unused_parameters", False)))

    ModelDDP(net, data, cfg["mulmask"]).train(add_temp=add_temp)

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

