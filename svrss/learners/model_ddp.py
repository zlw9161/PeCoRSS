"""Class to train a PyTorch model with DDP."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from svrss.learners.tester import Tester
from svrss.loaders.dataloaders import CarradaDataset
from svrss.loaders.dataset import CWRdata, PDRdata
from svrss.utils.functions import define_loss, get_transformations, normalize
from svrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer


class ModelDDP(nn.Module):
    """Class to train a model with DDP."""

    def __init__(self, net, data, rank=0, world_size=1, distributed=False):
        super().__init__()
        self.net = net
        self.cfg = data["cfg"]
        self.paths = data["paths"]
        self.dataloaders = data["dataloaders"]
        self.model_name = self.cfg["model"]
        self.resume_path = self.cfg["resume"]
        self.process_signal = self.cfg["process_signal"]
        self.signal_type = self.cfg["signal_type"]
        self.annot_type = self.cfg["annot_type"]
        self.w_size = self.cfg["w_size"]
        self.h_size = self.cfg["h_size"]
        self.batch_size = self.cfg["batch_size"]
        self.nb_epochs = self.cfg["nb_epochs"]
        self.lr = self.cfg["lr"]
        self.lr_step = self.cfg["lr_step"]
        self.schedular_type = self.cfg["schedular"]
        self.T_max = self.cfg["T_max"]
        self.loss_step = self.cfg["loss_step"]
        self.val_step = self.cfg["val_step"]
        self.viz_step = self.cfg["viz_step"]
        self.torch_seed = self.cfg["torch_seed"]
        self.numpy_seed = self.cfg["numpy_seed"]
        self.nb_classes = self.cfg["nb_classes"]
        self.device = self.cfg["device"]
        self.custom_loss = self.cfg["custom_loss"]
        self.comments = self.cfg["comments"]
        self.n_frames = self.cfg["nb_input_channels"]
        self.transform_names = self.cfg["transformations"].split(",")
        self.norm_type = self.cfg["norm_type"]
        self.is_shuffled = self.cfg["shuffle"]
        self.num_workers = self.cfg.get("num_workers", 2)

        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_main_process = (self.rank == 0)

        self.writer = SummaryWriter(self.paths["writer"]) if self.is_main_process else None
        self.visualizer = (
            TensorboardMultiLossVisualizer(self.writer) if self.is_main_process else None
        )
        self.tester = Tester(self.cfg, self.visualizer) if self.is_main_process else None
        self.results = dict()
        self._resume_ckpt_path = self._resolve_resume_checkpoint()

    def _resolve_resume_checkpoint(self):
        """Map cfg['resume'] / optional cfg['checkpoint_path'] to a checkpoint file."""
        r = self.cfg.get("resume")
        alt = self.cfg.get("checkpoint_path")

        def _exists(p):
            if p is None:
                return None
            p = Path(p).expanduser()
            return str(p) if p.is_file() else None

        if r in (False, None, "", 0):
            if alt:
                return _exists(alt)
            return None

        if isinstance(r, str):
            s = r.strip()
            if s.lower() in ("false", "0", "no", "none"):
                return _exists(alt)
            if s.lower() in ("true", "1", "yes") or s == "":
                if alt:
                    return _exists(alt)
                return _exists(self.paths["results"] / "model.pt")
            return _exists(s)

        if r is True:
            if alt:
                return _exists(alt)
            return _exists(self.paths["results"] / "model.pt")

        return _exists(alt)

    def train(self, add_temp=False):
        start_epoch = 0
        if self.is_main_process:
            self.writer.add_text("Comments", self.comments)

        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(
            self.transform_names, sizes=(self.w_size, self.h_size)
        )
        self._set_seeds()
        if not self._resume_ckpt_path:
            self.net.apply(self._init_weights)

        criterion = define_loss(self.signal_type, self.custom_loss, self.device)
        nb_losses = len(criterion)
        running_losses = []
        running_ce_losses = []
        running_dice_losses = []
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)

        if self.schedular_type == "exp":
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == "cos":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max - 1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")

        iteration = 0
        best_test_dice = 0.0
        if self.is_main_process and self.cfg.get("resume") not in (False, None, "", 0):
            if not self._resume_ckpt_path:
                print(
                    "Warning: resume is set but no checkpoint file was found "
                    "(use a path string, or checkpoint_path, or true with results/model.pt)."
                )
        if self._resume_ckpt_path:
            if torch.cuda.is_available():
                map_loc = torch.device(f"cuda:{torch.cuda.current_device()}")
            else:
                map_loc = torch.device("cpu")
            checkpoint = torch.load(self._resume_ckpt_path, map_location=map_loc)
            start_epoch = int(checkpoint["epoch"])
            state = checkpoint["state_dict"]
            if any(k.startswith("module.") for k in state.keys()):
                state = {k.replace("module.", "", 1): v for k, v in state.items()}
            self._model_ref().load_state_dict(state)
            optimizer.load_state_dict(checkpoint["optimizer"])
            best_test_dice = float(checkpoint.get("best_dice", 0.0))
            iteration = int(checkpoint.get("iteration", 0))
            if "scheduler" in checkpoint:
                try:
                    scheduler.load_state_dict(checkpoint["scheduler"])
                except Exception:
                    pass
            if self.is_main_process:
                print(
                    "=> Loaded checkpoint from {}".format(self._resume_ckpt_path)
                )
                print(
                    "=> Resume at epoch {} iter {} w/ best_dice {})".format(
                        start_epoch, iteration, best_test_dice
                    )
                )

        for epoch in range(start_epoch, self.nb_epochs):
            if (
                self.distributed
                and hasattr(train_loader, "sampler")
                and hasattr(train_loader.sampler, "set_epoch")
            ):
                train_loader.sampler.set_epoch(epoch)

            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths["carrada"], seq_name[0])
                frame_dataset = CarradaDataset(
                    seq,
                    self.annot_type,
                    path_to_frames,
                    self.process_signal,
                    self.n_frames,
                    transformations,
                    add_temp,
                )
                frame_dataloader = DataLoader(
                    frame_dataset,
                    shuffle=self.is_shuffled,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )

                for _, frame in enumerate(frame_dataloader):
                    if self.signal_type in "range_doppler":
                        sv_data = frame["rd_matrix"].to(self.device).float()
                        sv_mask = frame["rd_mask"].to(self.device).float()
                        sv_data = normalize(
                            sv_data, "range_doppler", norm_type=self.norm_type
                        )
                    elif self.signal_type in "range_angle":
                        sv_data = frame["ra_matrix"].to(self.device).float()
                        sv_mask = frame["ra_mask"].to(self.device).float()
                        sv_data = normalize(
                            sv_data, "range_angle", norm_type=self.norm_type
                        )
                    else:
                        raise ValueError("Unsupported signal_type: {}".format(self.signal_type))

                    optimizer.zero_grad()
                    sv_outputs = self.net(sv_data).to(self.device)
                    sv_losses = [
                        c(sv_outputs, torch.argmax(sv_mask, axis=1)) for c in criterion
                    ]
                    loss = torch.mean(torch.stack(sv_losses))
                    loss.backward()
                    optimizer.step()

                    reduced_loss = self._reduce_scalar(loss.detach())
                    reduced_ce = self._reduce_scalar(sv_losses[0].detach())
                    reduced_dice = self._reduce_scalar(sv_losses[1].detach())

                    running_losses.append(float(reduced_loss.item()))
                    running_ce_losses.append(float(reduced_ce.item()))
                    running_dice_losses.append(float(reduced_dice.item()))

                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        train_ce_loss = np.mean(running_ce_losses)
                        train_dice_loss = np.mean(running_dice_losses)
                        if self.is_main_process:
                            if self.signal_type in "range_doppler":
                                print(
                                    "[Epoch {}/{}, iter {}]: Train loss (RD View) {}".format(
                                        epoch + 1, self.nb_epochs, iteration, train_loss
                                    )
                                )
                            elif self.signal_type in "range_angle":
                                print(
                                    "[Epoch {}/{}, iter {}]: Train loss (RA View) {}".format(
                                        epoch + 1, self.nb_epochs, iteration, train_loss
                                    )
                                )
                            if nb_losses > 1:
                                print(
                                    "[Epoch {}/{}, iter {}]: Train CE loss {}".format(
                                        epoch + 1, self.nb_epochs, iteration, train_ce_loss
                                    )
                                )
                                print(
                                    "[Epoch {}/{}, iter {}]: Train Dice loss {}".format(
                                        epoch + 1, self.nb_epochs, iteration, train_dice_loss
                                    )
                                )
                                self.visualizer.update_sv_train_loss(
                                    self.signal_type,
                                    train_loss,
                                    train_ce_loss,
                                    train_dice_loss,
                                    iteration,
                                )
                            self.visualizer.update_learning_rate(
                                scheduler.get_last_lr()[0], iteration
                            )

                        running_losses = []
                        running_ce_losses = []
                        running_dice_losses = []

                    if iteration % self.val_step == 0 and iteration > 0:
                        if self.is_main_process:
                            model_for_eval = self.net.module if hasattr(self.net, "module") else self.net
                            if iteration % self.viz_step == 0:
                                val_metrics = self.tester.predict(
                                    model_for_eval, val_loader, iteration, add_temp=add_temp
                                )
                            else:
                                val_metrics = self.tester.predict(
                                    model_for_eval, val_loader, add_temp=add_temp
                                )

                            self.visualizer.update_sv_val_metrics(
                                self.signal_type, val_metrics, iteration
                            )
                            vm = (
                                val_metrics["range_doppler"]
                                if self.signal_type in "range_doppler"
                                else val_metrics["range_angle"]
                            )
                            if self.signal_type in "range_doppler":
                                print(
                                    "[Epoch {}/{}]: Val loss (RD View) {}".format(
                                        epoch + 1,
                                        self.nb_epochs,
                                        val_metrics["range_doppler"]["loss"],
                                    )
                                )
                                print(
                                    "[Epoch {}/{}] Val Metrics: Prec={}, mIoU={}, Dice={}".format(
                                        epoch + 1,
                                        self.nb_epochs,
                                        vm["prec"],
                                        vm["miou"],
                                        vm["dice"],
                                    )
                                )
                            elif self.signal_type in "range_angle":
                                print(
                                    "[Epoch {}/{}]: Val loss (RA View) {}".format(
                                        epoch + 1,
                                        self.nb_epochs,
                                        val_metrics["range_angle"]["loss"],
                                    )
                                )
                                print(
                                    "[Epoch {}/{}] Val Metrics: Prec={}, mIoU={}, Dice={}".format(
                                        epoch + 1,
                                        self.nb_epochs,
                                        vm["prec"],
                                        vm["miou"],
                                        vm["dice"],
                                    )
                                )

                            test_metrics = self.tester.predict(
                                model_for_eval, test_loader, add_temp=add_temp
                            )
                            self.visualizer.update_sv_test_metrics(
                                self.signal_type, test_metrics, iteration
                            )
                            metric_key = (
                                "range_doppler"
                                if self.signal_type in "range_doppler"
                                else "range_angle"
                            )
                            cur_dice = test_metrics[metric_key]["dice"]
                            tm = test_metrics[metric_key]
                            if cur_dice > best_test_dice and iteration > 0:
                                best_test_dice = cur_dice
                                if self.signal_type in "range_doppler":
                                    print(
                                        "[Epoch {}/{}]: Test loss (RD View) {}".format(
                                            epoch + 1,
                                            self.nb_epochs,
                                            tm["loss"],
                                        )
                                    )
                                    print(
                                        "[Epoch {}/{}] Test Metrics: Prec={}, mIoU={}, Dice={}".format(
                                            epoch + 1,
                                            self.nb_epochs,
                                            tm["prec"],
                                            tm["miou"],
                                            tm["dice"],
                                        )
                                    )
                                elif self.signal_type in "range_angle":
                                    print(
                                        "[Epoch {}/{}]: Test loss (RA View) {}".format(
                                            epoch + 1,
                                            self.nb_epochs,
                                            tm["loss"],
                                        )
                                    )
                                    print(
                                        "[Epoch {}/{}] Test Metrics: Prec={}, mIoU={}, Dice={}".format(
                                            epoch + 1,
                                            self.nb_epochs,
                                            tm["prec"],
                                            tm["miou"],
                                            tm["dice"],
                                        )
                                    )
                                self.results["train_ce_loss"] = float(train_ce_loss)
                                self.results["train_dice_loss"] = float(train_dice_loss)
                                self.results["train_loss"] = float(train_loss)
                                self.results["val_metrics"] = val_metrics
                                self.results["test_metrics"] = test_metrics
                                save_state = {
                                    "epoch": epoch + 1,
                                    "iteration": iteration + 1,
                                    "arch": str(self._model_ref()),
                                    "state_dict": self._model_ref().state_dict(),
                                    "best_dice": best_test_dice,
                                    "optimizer": optimizer.state_dict(),
                                    "scheduler": scheduler.state_dict(),
                                }
                                self._save_results(save_state)
                            self.net.train()
                        if self.distributed:
                            dist.barrier()

                    iteration += 1

            # After all optimizer updates in this epoch (matches PyTorch LR scheduler order).
            next_ep = epoch + 1
            if self.schedular_type == "exp":
                if next_ep % self.lr_step == 0 and next_ep != 0:
                    scheduler.step()
            else:
                if next_ep != 0 and next_ep < self.T_max:
                    scheduler.step()

        if self.writer is not None:
            self.writer.close()

    def _model_ref(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def _reduce_scalar(self, value):
        if not self.distributed:
            return value
        reduced = value.clone()
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / self.world_size
        return reduced

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0.0, 1.0)
            nn.init.constant_(m.bias, 0.0)

    def _save_results(self, state):
        if not self.is_main_process:
            return
        results_path = self.paths["results"] / "results.json"
        model_path = self.paths["results"] / "model.pt"
        with open(results_path, "w") as fp:
            json.dump(self.results, fp)
        torch.save(state, model_path)

    def _set_seeds(self):
        seed_offset = self.rank if self.distributed else 0
        torch.cuda.manual_seed_all(self.torch_seed + seed_offset)
        torch.manual_seed(self.torch_seed + seed_offset)
        np.random.seed(self.numpy_seed + seed_offset)
