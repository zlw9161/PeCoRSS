"""Initializer class for DDP training (distributed-safe)."""

import json
from torch.utils.data import DataLoader

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

from mvrss.utils.paths import Paths
from mvrss.loaders.dataset import Carrada
from mvrss.loaders.dataloaders_F import SequenceCarradaDataset


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


class InitializerDDP:
    """Prepare dataloaders + experiment folders for DDP.

    Key points:
    - Train loader uses DistributedSampler (each rank sees different sequences).
    - Val/Test loaders are NOT distributed by default (rank0 can run full evaluation).
    - Log folders and config.json are created only on rank0, then synced.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.paths = Paths().get()

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

    def _get_dataloaders(self):
        trainset, valset, testset = self._get_datasets()

        num_workers = int(self.cfg.get("num_workers", 0))

        train_sampler = None
        if _is_dist():
            train_sampler = DistributedSampler(trainset, shuffle=True, drop_last=False)

        trainloader = DataLoader(
            trainset,
            batch_size=1,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=True,
        )

        # Val/Test are full-dataset loaders (typically evaluated on rank0 only).
        valloader = DataLoader(valset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)
        testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True)

        return [trainloader, valloader, testloader]

    def _structure_data(self):
        data = dict()
        dataloaders = self._get_dataloaders()

        name_exp = (
            self.cfg["model"]
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

        # Rank0 chooses experiment folder; others wait and use same folder.
        folder_path = self.paths["logs"] / self.cfg["dataset"] / self.cfg["model"] / name_exp

        if _is_dist():
            rank = dist.get_rank()
        else:
            rank = 0

        if rank == 0:
            temp_folder_name = folder_path.name + "_" + str(self.cfg["version"])
            temp_folder_path = folder_path.parent / temp_folder_name
            while temp_folder_path.exists():
                self.cfg["version"] += 1
                temp_folder_name = folder_path.name + "_" + str(self.cfg["version"])
                temp_folder_path = folder_path.parent / temp_folder_name
            folder_path = temp_folder_path

            self.paths["results"] = folder_path / "results"
            self.paths["writer"] = folder_path / "boards"
            self.paths["results"].mkdir(parents=True, exist_ok=True)
            self.paths["writer"].mkdir(parents=True, exist_ok=True)

            config_path = folder_path / "config.json"
            with open(config_path, "w") as fp:
                json.dump(self.cfg, fp)

            folder_str = str(folder_path)
        else:
            folder_str = ""

        if _is_dist():
            obj_list = [folder_str]
            dist.broadcast_object_list(obj_list, src=0)
            folder_path = type(folder_path)(obj_list[0])  # Path-like

            # Non-rank0 builds paths (without mkdir race).
            self.paths["results"] = folder_path / "results"
            self.paths["writer"] = folder_path / "boards"
            dist.barrier()

        data["cfg"] = self.cfg
        data["paths"] = self.paths
        data["dataloaders"] = dataloaders
        return data

    def get_data(self):
        return self._structure_data()

