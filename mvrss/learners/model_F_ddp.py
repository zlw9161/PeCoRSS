"""DDP training loop for Model_F.

This is a minimal DDP adaptation:
- Train uses DistributedSampler on the *sequence* loader.
- On val steps in distributed mode: all ranks barrier, then all ranks run val ``predict`` (keeps
  replicas in sync; matches ``model_F_ddp1.py``), then barrier; rank0 additionally runs test
  ``predict``, logging, tensorboard, and checkpoint save.
"""

import os
import sys
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from mvrss.loaders.dataloaders_F import CarradaDataset
from mvrss.learners.tester_F import Tester
from mvrss.utils.functions import normalize, define_loss, get_transformations, build_track_heatmap
from mvrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer


def _is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()


def _rank() -> int:
    return dist.get_rank() if _is_dist() else 0


def _is_rank0() -> bool:
    return _rank() == 0


class ModelDDP(nn.Module):
    def __init__(self, net, data, Mulmask: bool):
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
        self.s = self.cfg["bias_s"]
        self.lamda_flag = self.cfg["lamda_flag"]
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
        self.flag_coherence_track_loss = self.cfg["ct_loss"]
        self.comments = self.cfg["comments"]
        self.n_frames = self.cfg["nb_input_channels"]
        self.transform_names = self.cfg["transformations"].split(",")
        self.norm_type = self.cfg["norm_type"]
        self.is_shuffled = self.cfg["shuffle"]
        self.soft_trackloss = self.cfg["soft_trackloss"]
        self.mulmask = Mulmask
        self.erstp = self.cfg["erstp"]
        self.tgt = self.cfg["tgt"]
        self.rd_weight = self.cfg["rd_weight"]
        self.ra_weight = self.cfg["ra_weight"]
        self.num_workers = int(self.cfg.get("num_workers", 2))
        self.track_gamma = float(self.cfg.get("track_gamma", 0.9))
        self.track_weight = float(self.cfg.get("track_weight", 0.5))
        self.rd_loss_weight = float(self.cfg.get("rd_loss_weight", 1.0))
        self.ra_loss_weight = float(self.cfg.get("ra_loss_weight", 1.0))
        self.use_ema = bool(self.cfg.get("use_ema", False))
        self.ema_decay = float(self.cfg.get("ema_decay", 0.999))
        self.ema_update_after_step = int(self.cfg.get("ema_update_after_step", 100))
        self._ema_state = None

        # Propagate configured target frame to model (controls x_tgt_ori selection).
        model_for_set = self.net.module if hasattr(self.net, "module") else self.net
        if hasattr(model_for_set, "set_tgt_index"):
            try:
                model_for_set.set_tgt_index(self.tgt)
            except Exception:
                pass

        # Only rank0 writes logs/boards.
        if _is_rank0():
            self.writer = SummaryWriter(self.paths["writer"])
            self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        else:
            self.writer = None
            self.visualizer = None

        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()

    def _set_seeds(self):
        torch.manual_seed(self.torch_seed + _rank())
        np.random.seed(self.numpy_seed + _rank())

    def calc_alpha_bar(self, current_epoch, total_epoch, s):
        denominator = np.cos((np.pi / 2) * (s / (1 + s)))
        numerator = np.cos((np.pi / 2) * ((current_epoch / total_epoch) + s) / (1 + s))
        return (numerator / denominator) ** 2

    def _save_results(self, state):
        if not _is_rank0():
            return
        results_path = self.paths["results"] / "results.json"
        model_path = self.paths["results"] / "model.pt"
        with open(results_path, "w") as fp:
            json.dump(self.results, fp)
        torch.save(state, model_path)

    def _ema_model(self):
        return self.net.module if hasattr(self.net, "module") else self.net

    def _init_ema_state(self):
        if not self.use_ema:
            self._ema_state = None
            return
        model = self._ema_model()
        self._ema_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    def _restore_ema_from_checkpoint(self, ema_state):
        if not self.use_ema or ema_state is None:
            return
        model = self._ema_model()
        model_state = model.state_dict()
        restored = {}
        for k, v in model_state.items():
            if k in ema_state:
                restored[k] = ema_state[k].detach().to(device=v.device, dtype=v.dtype)
            else:
                restored[k] = v.detach().clone()
        self._ema_state = restored

    def _update_ema(self, iteration: int):
        if (not self.use_ema) or (self._ema_state is None) or (iteration < self.ema_update_after_step):
            return
        model = self._ema_model()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                if not torch.is_floating_point(v):
                    self._ema_state[k].copy_(v)
                else:
                    self._ema_state[k].mul_(self.ema_decay).add_(v.detach(), alpha=1.0 - self.ema_decay)

    def _swap_in_ema_weights(self):
        if (not self.use_ema) or (self._ema_state is None):
            return None
        model = self._ema_model()
        backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(self._ema_state[k])
        return backup_state

    def _restore_from_backup(self, backup_state):
        if backup_state is None:
            return
        model = self._ema_model()
        with torch.no_grad():
            for k, v in model.state_dict().items():
                v.copy_(backup_state[k])

    def _ema_state_for_save(self):
        if self._ema_state is None:
            return None
        return {k: v.detach().cpu().clone() for k, v in self._ema_state.items()}

    def train(self, add_temp=False):
        start_epoch = 0
        checkpoint = None
        if _is_rank0() and self.writer is not None:
            self.writer.add_text("Comments", self.comments)

        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names, sizes=(self.w_size, self.h_size))
        self._set_seeds()

        rd_criterion = define_loss("range_doppler", self.custom_loss, self.device, view_weight=self.rd_weight)
        ra_criterion = define_loss("range_angle", self.custom_loss, self.device, view_weight=self.ra_weight)
        nb_losses = len(rd_criterion)

        iteration = 0
        best_test_dice = 0.0
        patience_counter = 0
        train_loss = 0.0

        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        if self.schedular_type == "exp":
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == "cos":
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max - 1, eta_min=1e-7)
        elif self.schedular_type == "red":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.5,
                patience=5,
                verbose=_is_rank0(),
                threshold=1e-4,
                min_lr=1e-7,
                cooldown=2,
            )
        else:
            raise KeyError("we only implement schedular of exp, cos and red")

        if self.resume_path:
            map_location = "cpu"
            checkpoint = torch.load(self.resume_path, map_location=map_location)
            start_epoch = checkpoint["epoch"]
            try:
                self.net.load_state_dict(checkpoint["net"])
            except Exception:
                # If checkpoint saved from DP/DDP wrappers, let user handle mapping explicitly.
                self.net.load_state_dict(checkpoint["net"], strict=False)
            optimizer.load_state_dict(checkpoint["optimizer"])
            scheduler.load_state_dict(checkpoint["lr_scheduler"])
            best_test_dice = checkpoint.get("best_dice", 0.0)
            if _is_rank0():
                print(f"=> Loaded checkpoint at epoch {checkpoint['epoch']}", flush=True)

        self._init_ema_state()
        if checkpoint is not None:
            self._restore_ema_from_checkpoint(checkpoint.get("ema_state"))

        # If train_loader uses DistributedSampler, we must set_epoch each epoch for shuffle.
        train_sampler = getattr(train_loader, "sampler", None)
        for epoch in range(start_epoch, self.nb_epochs):
            if _is_dist() and hasattr(train_sampler, "set_epoch"):
                train_sampler.set_epoch(epoch)

            if self.lamda_flag == 1:
                lamda = self.calc_alpha_bar(epoch, self.nb_epochs, self.s)
                if _is_rank0():
                    print(f"lamda = {lamda} in epoch {epoch}\n", flush=True)
            else:
                lamda = None

            if self.schedular_type == "exp":
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            elif self.schedular_type == "cos":
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()

            running_losses = []
            rd_running_losses = []
            ra_running_losses = []
            coherence_running_losses = []
            track_running_losses = []

            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths["carrada"], seq_name[0])
                frame_dataloader = DataLoader(
                    CarradaDataset(
                        seq,
                        self.annot_type,
                        path_to_frames,
                        self.process_signal,
                        self.n_frames,
                        transformations,
                        add_temp,
                        self.mulmask,
                    ),
                    shuffle=self.is_shuffled,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=True,
                )

                for _, frame in enumerate(frame_dataloader):
                    if self.signal_type != "range_doppler_angle":
                        raise NotImplementedError("ModelDDP currently targets range_doppler_angle training.")

                    rd_data = frame["rd_matrix"].to(self.device).float()
                    ra_data = frame["ra_matrix"].to(self.device).float()
                    ad_data = frame["ad_matrix"].to(self.device).float()
                    rd_mask = frame["rd_mask"].to(self.device).float()
                    ra_mask = frame["ra_mask"].to(self.device).float()

                    rd_data = normalize(rd_data, "range_doppler", norm_type=self.norm_type)
                    ra_data = normalize(ra_data, "range_angle", norm_type=self.norm_type)
                    if self.model_name != "mvnet":
                        ad_data = normalize(ad_data, "angle_doppler", norm_type=self.norm_type)

                    optimizer.zero_grad(set_to_none=True)

                    if self.model_name != "mvnet":
                        if self.lamda_flag == 1:
                            rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = self.net(rd_data, ra_data, ad_data, lamda)
                        else:
                            rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = self.net(rd_data, ra_data, ad_data)
                    else:
                        rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = self.net(rd_data, ra_data)

                    rd_tgt_mask = rd_mask[:, :, self.tgt, :, :] if rd_mask.dim() == 5 else rd_mask
                    ra_tgt_mask = ra_mask[:, :, self.tgt, :, :] if ra_mask.dim() == 5 else ra_mask

                    rd_losses = [c(rd_outputs, torch.argmax(rd_tgt_mask, axis=1)) for c in rd_criterion[:2]]
                    ra_losses = [c(ra_outputs, torch.argmax(ra_tgt_mask, axis=1)) for c in ra_criterion[:2]]
                    rd_loss = torch.mean(torch.stack(rd_losses))
                    ra_loss = torch.mean(torch.stack(ra_losses))

                    if nb_losses > 3:
                        coherence_loss = rd_criterion[3](rd_outputs, ra_outputs)
                    else:
                        coherence_loss = torch.tensor(0.0, device=rd_outputs.device)

                    base_loss = self.rd_loss_weight * rd_loss + self.ra_loss_weight * ra_loss
                    if nb_losses > 3:
                        base_loss = base_loss + 0.5 * coherence_loss

                    loss = torch.mean(base_loss)
                    track_loss = torch.tensor(0.0, device=rd_outputs.device)
                    if self.mulmask:
                        # Track heatmap regression (per-class) from history masks.
                        track_gt_rd = build_track_heatmap(rd_mask, gamma=self.track_gamma, use_history=True)
                        track_gt_ra = build_track_heatmap(ra_mask, gamma=self.track_gamma, use_history=True)
                        track_pred_rd = torch.sigmoid(rd_track_logits)
                        track_pred_ra = torch.sigmoid(ra_track_logits)
                        track_loss_rd = F.smooth_l1_loss(track_pred_rd, track_gt_rd)
                        track_loss_ra = F.smooth_l1_loss(track_pred_ra, track_gt_ra)
                        track_loss = torch.mean(track_loss_ra + track_loss_rd)
                        loss = torch.mean(loss + self.track_weight * track_loss)

                    loss.backward()
                    optimizer.step()
                    self._update_ema(iteration)

                    running_losses.append(loss.detach().cpu().numpy().item())
                    rd_running_losses.append(rd_loss.detach().cpu().numpy().item())
                    ra_running_losses.append(ra_loss.detach().cpu().numpy().item())
                    coherence_running_losses.append(coherence_loss.detach().cpu().numpy().item())
                    if self.mulmask:
                        track_running_losses.append(track_loss.detach().cpu().numpy().item())

                    if _is_rank0() and iteration % self.loss_step == 0:
                        train_loss = float(np.mean(running_losses)) if running_losses else 0.0
                        rd_train_loss = float(np.mean(rd_running_losses)) if rd_running_losses else 0.0
                        ra_train_loss = float(np.mean(ra_running_losses)) if ra_running_losses else 0.0
                        print(f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train loss {train_loss}", flush=True)
                        print(
                            f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train losses: RD={rd_train_loss}, RA={ra_train_loss}",
                            flush=True,
                        )
                        if self.mulmask:
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train Track loss {float(np.mean(track_running_losses))}",
                                flush=True,
                            )
                        if nb_losses > 3:
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train Coherence loss {float(np.mean(coherence_running_losses))}",
                                flush=True,
                            )

                        running_losses = []
                        rd_running_losses = []
                        ra_running_losses = []
                        coherence_running_losses = []
                        track_running_losses = []

                    # Validation (model_F_ddp1.py): all ranks sync, all run val predict, then rank0
                    # test/log/save; all ranks set train + final barrier — avoids DDP stalls.
                    do_val = iteration % self.val_step == 0 and iteration > 0
                    if do_val:
                        if _is_dist():
                            dist.barrier()

                        ema_backup = self._swap_in_ema_weights()
                        try:
                            if self.lamda_flag == 1:
                                val_metrics = self.tester.predict(self.net, val_loader, lamda=lamda, add_temp=add_temp)
                            else:
                                val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                            if _is_rank0():
                                if self.visualizer is not None:
                                    self.visualizer.update_multi_val_metrics(val_metrics, iteration)

                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Val losses: RD={val_metrics['range_doppler']['loss']}, RA={val_metrics['range_angle']['loss']}",
                                    flush=True,
                                )
                                if nb_losses > 3 and "coherence_loss" in val_metrics:
                                    print(f"[Epoch {epoch+1}/{self.nb_epochs}] Val Coherence Loss: CoL={val_metrics['coherence_loss']}", flush=True)
                                if self.mulmask and "track_loss" in val_metrics:
                                    print(f"[Epoch {epoch+1}/{self.nb_epochs}] Val Track losses: RDA={val_metrics['track_loss']}", flush=True)
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Val Dice: RD={val_metrics['range_doppler']['dice']}, RA={val_metrics['range_angle']['dice']}",
                                    flush=True,
                                )

                                if self.lamda_flag == 1:
                                    test_metrics = self.tester.predict(self.net, test_loader, lamda=lamda, add_temp=add_temp)
                                else:
                                    test_metrics = self.tester.predict(self.net, test_loader, add_temp=add_temp)
                                print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}", flush=True)
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Test losses: RD={test_metrics['range_doppler']['loss']}, RA={test_metrics['range_angle']['loss']}",
                                    flush=True,
                                )
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Test Prec: RD={test_metrics['range_doppler']['prec']}, RA={test_metrics['range_angle']['prec']}",
                                    flush=True,
                                )
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Test mIoU: RD={test_metrics['range_doppler']['miou']}, RA={test_metrics['range_angle']['miou']}",
                                    flush=True,
                                )
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Test Dice: RD={test_metrics['range_doppler']['dice']}, RA={test_metrics['range_angle']['dice']}",
                                    flush=True,
                                )
                                if "global_prec" in test_metrics:
                                    print(f"[Epoch {epoch+1}/{self.nb_epochs}] Test Global Prec: {test_metrics['global_prec']}", flush=True)
                                if "global_miou" in test_metrics:
                                    print(f"[Epoch {epoch+1}/{self.nb_epochs}] Test Global mIoU: {test_metrics['global_miou']}", flush=True)
                                if "global_dice" in test_metrics:
                                    print(f"[Epoch {epoch+1}/{self.nb_epochs}] Test Global Dice: {test_metrics['global_dice']}", flush=True)

                                current_test_dice = float(test_metrics["global_dice"])
                                if current_test_dice > best_test_dice:
                                    best_test_dice = current_test_dice
                                    print(
                                        f"[Epoch {epoch+1}/{self.nb_epochs}] New best test global_dice: {best_test_dice:.6f}. Saving checkpoint.",
                                        flush=True,
                                    )
                                    self.results["best_epoch"] = epoch + 1
                                    self.results["best_test_dice"] = best_test_dice
                                    self.results["train_loss_at_best"] = train_loss
                                    self.results["best_val_metrics"] = val_metrics
                                    self.results["best_test_metrics"] = test_metrics
                                    state = {
                                        "epoch": epoch + 1,
                                        "arch": str(self.net),
                                        "net": self.net.state_dict(),
                                        "best_dice": best_test_dice,
                                        "optimizer": optimizer.state_dict(),
                                        "lr_scheduler": scheduler.state_dict(),
                                    }
                                    ema_state = self._ema_state_for_save()
                                    if ema_state is not None:
                                        state["ema_state"] = ema_state
                                    self._save_results(state)
                        finally:
                            self._restore_from_backup(ema_backup)

                        self.net.train()

                        # All ranks run validation; clear allocator cache before resuming training.
                        if self.device == "cuda":
                            if _is_rank0():
                                torch.cuda.synchronize()
                            del val_metrics
                            if _is_rank0():
                                try:
                                    del test_metrics
                                except (UnboundLocalError, NameError):
                                    pass
                            torch.cuda.empty_cache()

                        if _is_dist():
                            dist.barrier()

                    iteration += 1

        if _is_rank0() and self.writer is not None:
            self.writer.close()

