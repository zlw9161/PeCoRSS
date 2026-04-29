"""Main script to train models with DDP + SyncBN."""

import argparse
import json
import os

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from svrss.learners.initializer_ddp_sv import InitializerDDPSV
from svrss.learners.model_ddp_sv import ModelDDP
from svrss.models import build_model
from svrss.utils.functions import count_params


def _init_distributed(args):
    distributed = (
        "RANK" in os.environ and "WORLD_SIZE" in os.environ and "LOCAL_RANK" in os.environ
    )
    if not distributed:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
        return False

    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ["WORLD_SIZE"])
    args.local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
    dist.barrier()
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", help="Path to config file.", default="config.json")
    parser.add_argument(
        "--dist-url", default="env://", help="url used to set up distributed training"
    )
    parser.add_argument("--dist-backend", default="nccl", help="distributed backend")
    parser.add_argument(
        "--no-sync-bn",
        action="store_true",
        help="Disable SyncBatchNorm conversion before DDP wrap",
    )
    args = parser.parse_args()

    distributed = _init_distributed(args)
    is_main = args.rank == 0
    device = torch.device(
        "cuda", args.local_rank if torch.cuda.is_available() else "cpu"
    )

    with open(args.cfg, "r") as fp:
        cfg = json.load(fp)
    cfg["device"] = str(device)

    init = InitializerDDPSV(
        cfg, distributed=distributed, rank=args.rank, world_size=args.world_size
    )
    data = init.get_data()

    net = build_model(
        model_name=cfg["model"],
        n_classes=data["cfg"]["nb_classes"],
        n_frames=data["cfg"]["nb_input_channels"],
        signal_type=data["cfg"]["signal_type"],
        zero_doppler_filter=False,
        search_size=data["cfg"].get("search_size"),
    )
    if is_main:
        print("Number of trainable parameters in the model: {}".format(count_params(net)))

    net = net.to(device)
    if distributed and not args.no_sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if distributed:
        net = DDP(net, device_ids=[args.local_rank], output_device=args.local_rank)

    if cfg["seq_mode"]:
        ModelDDP(
            net,
            data,
            rank=args.rank,
            world_size=args.world_size,
            distributed=distributed,
        ).train(add_temp=True)
    else:
        ModelDDP(
            net,
            data,
            rank=args.rank,
            world_size=args.world_size,
            distributed=distributed,
        ).train(add_temp=False)

    if distributed:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
