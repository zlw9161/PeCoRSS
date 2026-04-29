"""Initializer class to prepare DDP training."""
import json
from pathlib import Path

import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mvrss.utils.paths import Paths
from mvrss.loaders.dataset import Carrada
from mvrss.loaders.dataloaders import SequenceCarradaDataset


class InitializerDDP:
    """Class to prepare DDP training data and experiment paths."""

    def __init__(self, cfg, distributed=False, rank=0, world_size=1):
        self.cfg = cfg
        self.paths = Paths().get()
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size

    def _get_data(self):
        data = Carrada()
        train = data.get("Train")
        val = data.get("Validation")
        test = data.get("Test")
        return [train, val, test]

    def _get_datasets(self):
        data = self._get_data()
        trainset = SequenceCarradaDataset(data[0])
        valset = SequenceCarradaDataset(data[1])
        testset = SequenceCarradaDataset(data[2])
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
        folder_root = self.paths["logs"] / self.cfg["dataset"] / self.cfg["model"]

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
        if self.distributed:
            train_sampler = DistributedSampler(
                trainset, num_replicas=self.world_size, rank=self.rank, shuffle=True
            )

        trainloader = DataLoader(
            trainset,
            batch_size=1,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            num_workers=0,
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
        data = dict()
        data["cfg"] = self.cfg
        data["paths"] = self.paths
        data["dataloaders"] = self._get_dataloaders()
        return data

    def get_data(self):
        """Return parameters of the DDP training run."""
        return self._structure_data()
