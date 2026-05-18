"""Initializer for DDP training on PeCoR-SV style datasets (CWR/PDR/Carrada, etc.)."""
import json
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from svrss.loaders.dataset import Carrada, CWRdata, PDRdata
from svrss.loaders.dataloaders_sv import SVGlobalWindowTrainDataset, SequenceDataset
from svrss.utils.paths import Paths


def _split_source_for_cfg(cfg):
    """Return a dataset *instance* with .get('Train'|'Validation'|'Test') for cfg['dataset']."""
    raw = (cfg.get("dataset") or "").strip()
    d = raw.lower()

    if d in ("cwr", "cwrdata"):
        return CWRdata()
    if d in ("pdr", "pdrdata"):
        return PDRdata()
    if d == "carrada" or d.startswith("carrada"):
        return Carrada()

    raise ValueError(
        f"initializer_ddp_sv: unsupported cfg['dataset']={raw!r}. "
        "Use cwr/cwrdata, pdr/pdrdata, or carrada/…"
    )


class InitializerDDPSV:
    """Prepare DDP dataloaders and paths for SV pipelines using SequenceDataset."""

    def __init__(self, cfg, distributed=False, rank=0, world_size=1):
        self.cfg = cfg
        self.paths = Paths().get()
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size

    def _get_data(self):
        src = _split_source_for_cfg(self.cfg)
        return [src.get("Train"), src.get("Validation"), src.get("Test")]

    def _get_datasets(self):
        train_d, val_d, test_d = self._get_data()
        if self.cfg.get("seq_shuffle", False):
            trainset = SVGlobalWindowTrainDataset(train_d, self.cfg)
            if len(trainset) == 0:
                raise RuntimeError(
                    "seq_shuffle is True but no training windows were built "
                    "(check nb_input_channels vs sequence lengths)."
                )
        else:
            trainset = SequenceDataset(train_d)
        valset = SequenceDataset(val_d)
        testset = SequenceDataset(test_d)
        return [trainset, valset, testset]

    def _prepare_exp_folders(self):
        signal_type = self.cfg.get("signal_type", "unknown")
        name_exp = (
            self.cfg["model"]
            + "_"
            + signal_type
            + "_"
            + "e"
            + str(self.cfg["nb_epochs"])
            + "_"
            + "lr"
            + str(self.cfg["lr"])
            + "_"
            + "s"
            + str(self.cfg["torch_seed"])
        )
        self.cfg["name_exp"] = name_exp
        folder_root = self.paths["logs"] / str(self.cfg["dataset"]) / self.cfg["model"]

        if self.rank == 0:
            selected_version = int(self.cfg["version"])
            folder_path = folder_root / f"{name_exp}_{selected_version}"
            while folder_path.exists():
                selected_version += 1
                folder_path = folder_root / f"{name_exp}_{selected_version}"
        else:
            selected_version = None
            folder_path = None

        if self.distributed:
            obj_list = [selected_version, str(folder_path) if folder_path is not None else ""]
            dist.broadcast_object_list(obj_list, src=0)
            selected_version = int(obj_list[0])
            folder_path = Path(obj_list[1])

        self.cfg["version"] = selected_version
        self.paths["results"] = folder_path / "results"
        self.paths["writer"] = folder_path / "boards"

        if self.rank == 0:
            self.paths["results"].mkdir(parents=True, exist_ok=True)
            self.paths["writer"].mkdir(parents=True, exist_ok=True)
            config_path = folder_path / "config.json"
            with open(config_path, "w") as fp:
                json.dump(self.cfg, fp)

        if self.distributed:
            dist.barrier()

    def _get_dataloaders(self):
        trainset, valset, testset = self._get_datasets()

        train_sampler = None
        use_seq_shuffle = self.cfg.get("seq_shuffle", False)
        if self.distributed:
            sampler_shuffle = (not use_seq_shuffle) or bool(self.cfg.get("shuffle", True))
            train_sampler = DistributedSampler(
                trainset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=sampler_shuffle,
            )

        nw_train = self.cfg.get("num_workers", 2) if use_seq_shuffle else 0
        bs_train = self.cfg["batch_size"] if use_seq_shuffle else 1
        if use_seq_shuffle:
            shuffle_train = (train_sampler is None) and bool(self.cfg.get("shuffle", True))
        else:
            shuffle_train = train_sampler is None
        trainloader = DataLoader(
            trainset,
            batch_size=bs_train,
            shuffle=shuffle_train,
            sampler=train_sampler,
            num_workers=nw_train,
        )

        if self.rank == 0 or not self.distributed:
            valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=0)
            testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
        else:
            valloader = None
            testloader = None

        return [trainloader, valloader, testloader]

    def _structure_data(self):
        self._prepare_exp_folders()
        return {
            "cfg": self.cfg,
            "paths": self.paths,
            "dataloaders": self._get_dataloaders(),
        }

    def get_data(self):
        return self._structure_data()
