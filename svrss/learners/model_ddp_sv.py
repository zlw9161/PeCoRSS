"""Class to train a PyTorch model with DDP."""

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from svrss.learners.tester_sv import TesterSV
from svrss.loaders.dataloaders_sv import PeCoRSVDataset
from svrss.utils.functions import (
    define_loss,
    build_track_heatmap,
    get_transformations,
    normalize,
    resolve_class_weights_dir,
    resolve_tvt_stats_json_path,
    resolved_class_weights_json_path,
)
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
        self.cmplx_flag = self.cfg.get("cmplx_flag", False)
        self.mulmask = bool(self.cfg.get("mulmask", False))
        self.track_gamma = float(self.cfg.get("track_gamma", 0.9))
        self.track_weight = float(self.cfg.get("track_weight", 0.5))
        self.tgt = int(self.cfg.get("tgt", -1))

        self.rank = rank
        self.world_size = world_size
        self.distributed = distributed
        self.is_main_process = (self.rank == 0)

        self.writer = SummaryWriter(self.paths["writer"]) if self.is_main_process else None
        self.visualizer = (
            TensorboardMultiLossVisualizer(self.writer) if self.is_main_process else None
        )
        self.tester = TesterSV(self.cfg, self.visualizer) if self.is_main_process else None
        self.results = dict()
        self._resume_ckpt_path = self._resolve_resume_checkpoint()
        self._debug_first_batch_printed = False
        model_for_set = self._model_ref()
        if hasattr(model_for_set, "set_tgt_index"):
            try:
                model_for_set.set_tgt_index(self.tgt)
            except Exception:
                pass

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

    def _peco_dataset_type(self):
        d = (self.cfg.get("dataset") or "").strip()
        dl = d.lower()
        if dl in ("cwr", "cwrdata"):
            return "CWRdata"
        if dl in ("pdr", "pdrdata"):
            return "PDRdata"
        if dl == "carrada":
            if self.signal_type == "range_doppler":
                return "Carrada_RD"
            if self.signal_type == "range_angle":
                return "Carrada_RA"
            return "Carrada"
        return d

    def _sequence_frames_root(self, seq_name):
        seq0 = seq_name[0]
        paths = self.paths
        dl = (self.cfg.get("dataset") or "").lower()
        if dl in ("cwr", "cwrdata"):
            return paths["CWRdata"] / seq0
        if dl in ("pdr", "pdrdata"):
            return paths["PDRdata"] / seq0
        if dl == "carrada" or dl.startswith("carrada"):
            return paths["carrada"] / seq0
        if dl in ("xpddata", "xpd"):
            return paths["XPDdata"] / seq0
        if dl in ("rdsyn",):
            return paths["RDSyn"] / seq0
        if dl in ("nuaasim",):
            return paths["NuaaSim"] / seq0
        raise KeyError(
            f"model_ddp_sv: unsupported cfg['dataset']={self.cfg.get('dataset')!r} for path resolution."
        )

    @staticmethod
    def _resolve_tgt_index(t: int, tgt_index: int) -> int:
        if t <= 0:
            return 0
        idx = int(tgt_index)
        if idx < 0:
            idx = t + idx
        if idx < 0:
            idx = 0
        if idx > t - 1:
            idx = t - 1
        return idx

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

        _w_dir = resolve_class_weights_dir(self.cfg, self.paths)
        _tvt_stats = resolve_tvt_stats_json_path(
            self.signal_type, self.cfg, self.paths
        )
        if self.is_main_process:
            cl = self.custom_loss
            if cl == "wce" or (isinstance(cl, str) and cl.startswith("wce_")):
                print(
                    "[svrss] class weights JSON:",
                    resolved_class_weights_json_path(
                        self.signal_type, self.cfg, self.paths
                    ),
                    flush=True,
                )
            if self.norm_type == "tvt" and _tvt_stats is not None:
                print(
                    "[svrss] tvt norm stats JSON:",
                    _tvt_stats.resolve(),
                    flush=True,
                )
        criterion = define_loss(
            self.signal_type,
            self.custom_loss,
            self.device,
            weights_dir=_w_dir,
            cfg=self.cfg,
        )
        # define_loss may return a single Module (e.g. wce, sdice) or a list (compound losses).
        if not isinstance(criterion, (list, tuple)):
            criterion = [criterion]
        nb_losses = len(criterion)
        running_losses = []
        running_ce_losses = []
        running_dice_losses = []
        running_track_losses = []
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
                print(f"=> Loaded checkpoint at epoch {checkpoint['epoch']}", flush=True)
                print(
                    "=> Resume at epoch {} iter {} w/ best_dice {})".format(
                        start_epoch, iteration, best_test_dice
                    ),
                    flush=True,
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
                path_to_frames = self._sequence_frames_root(seq_name)
                pe_type = self._peco_dataset_type()
                _dataset = PeCoRSVDataset(
                    seq,
                    pe_type,
                    self.annot_type,
                    self.signal_type,
                    path_to_frames,
                    self.process_signal,
                    self.n_frames,
                    self.cmplx_flag,
                    transformations,
                    add_temp,
                    self.mulmask,
                )

                frame_dataloader = DataLoader(
                     _dataset,
                    shuffle=self.is_shuffled,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                )

                for _, frame in enumerate(frame_dataloader):
                    if self.signal_type in "range_doppler":
                        sv_data = frame["rd_matrix"].to(self.device).float()
                        sv_mask = frame["rd_mask"].to(self.device).float()
                        sv_data = normalize(
                            sv_data,
                            "range_doppler",
                            norm_type=self.norm_type,
                            stats_json_path=_tvt_stats,
                        )
                    elif self.signal_type in "range_angle":
                        sv_data = frame["ra_matrix"].to(self.device).float()
                        sv_mask = frame["ra_mask"].to(self.device).float()
                        sv_data = normalize(
                            sv_data,
                            "range_angle",
                            norm_type=self.norm_type,
                            stats_json_path=_tvt_stats,
                        )
                    else:
                        raise ValueError("Unsupported signal_type: {}".format(self.signal_type))

                    # Debug: verify data loading (shape + class distribution) once.
                    if self.is_main_process and (not self._debug_first_batch_printed):
                        try:
                            y = torch.argmax(sv_mask, dim=1).detach().cpu()
                            u, c = torch.unique(y, return_counts=True)
                            cls_dist = {int(k): int(v) for k, v in zip(u.tolist(), c.tolist())}
                        except Exception as exc:
                            cls_dist = {"error": str(exc)}
                        print(
                            "[debug:first_batch] seq={} sv_data.shape={} sv_mask.shape={} mask_class_counts={}".format(
                                seq_name[0],
                                tuple(sv_data.shape),
                                tuple(sv_mask.shape),
                                cls_dist,
                            )
                        )
                        self._debug_first_batch_printed = True

                    optimizer.zero_grad()
                    net_outputs = self.net(sv_data)
                    if isinstance(net_outputs, (tuple, list)):
                        sv_outputs = net_outputs[0]
                        track_logits = net_outputs[1] if len(net_outputs) > 1 else None
                    else:
                        sv_outputs = net_outputs
                        track_logits = None
                    sv_outputs = sv_outputs.to(self.device)

                    if sv_mask.dim() == 5:
                        tgt_idx = self._resolve_tgt_index(sv_mask.size(2), self.tgt)
                        sv_tgt_mask = sv_mask[:, :, tgt_idx, :, :]
                    else:
                        sv_tgt_mask = sv_mask
                    sv_losses = [
                        c(sv_outputs, torch.argmax(sv_tgt_mask, axis=1)) for c in criterion
                    ]
                    loss = torch.mean(torch.stack(sv_losses))

                    track_loss = None
                    if self.mulmask and sv_mask.dim() == 5 and track_logits is not None:
                        track_gt = build_track_heatmap(
                            sv_mask, gamma=self.track_gamma, use_history=True
                        )
                        track_pred = torch.sigmoid(track_logits)
                        if track_pred.shape == track_gt.shape:
                            track_loss = F.smooth_l1_loss(track_pred, track_gt)
                            loss = loss + self.track_weight * track_loss

                    loss.backward()
                    optimizer.step()

                    reduced_loss = self._reduce_scalar(loss.detach())
                    running_losses.append(float(reduced_loss.item()))
                    if nb_losses > 1:
                        reduced_ce = self._reduce_scalar(sv_losses[0].detach())
                        reduced_dice = self._reduce_scalar(sv_losses[1].detach())
                        running_ce_losses.append(float(reduced_ce.item()))
                        running_dice_losses.append(float(reduced_dice.item()))
                    if track_loss is not None:
                        reduced_track = self._reduce_scalar(track_loss.detach())
                        running_track_losses.append(float(reduced_track.item()))

                    if iteration % self.loss_step == 0:
                        train_loss = float(np.mean(running_losses)) if running_losses else 0.0
                        if self.is_main_process:
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train loss {train_loss}",
                                flush=True,
                            )
                            if nb_losses > 1:
                                train_ce_loss = float(np.mean(running_ce_losses)) if running_ce_losses else 0.0
                                train_dice_loss = float(np.mean(running_dice_losses)) if running_dice_losses else 0.0
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train losses: CE={train_ce_loss}, Dice={train_dice_loss}",
                                    flush=True,
                                )
                                self.visualizer.update_sv_train_loss(
                                    self.signal_type,
                                    train_loss,
                                    train_ce_loss,
                                    train_dice_loss,
                                    iteration,
                                )
                            else:
                                train_ce_loss = train_loss
                                self.visualizer.update_sv_train_global_only(
                                    self.signal_type, train_loss, iteration
                                )
                            if running_track_losses:
                                train_track_loss = float(np.mean(running_track_losses))
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}, iter {iteration}]: Train Track loss {train_track_loss}",
                                    flush=True,
                                )
                            self.visualizer.update_learning_rate(
                                scheduler.get_last_lr()[0], iteration
                            )

                        running_losses = []
                        if nb_losses > 1:
                            running_ce_losses = []
                            running_dice_losses = []
                        running_track_losses = []

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
                            view_tag = "RD" if self.signal_type in "range_doppler" else "RA"
                            metric_key = "range_doppler" if self.signal_type in "range_doppler" else "range_angle"
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}] Val losses: {view_tag}={val_metrics[metric_key]['loss']}",
                                flush=True,
                            )
                            if "track_loss" in val_metrics:
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Val Track losses: {view_tag}={val_metrics['track_loss']}",
                                    flush=True,
                                )
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}] Val Dice: {view_tag}={vm['dice']}",
                                flush=True,
                            )

                            test_metrics = self.tester.predict(
                                model_for_eval, test_loader, add_temp=add_temp
                            )
                            self.visualizer.update_sv_test_metrics(
                                self.signal_type, test_metrics, iteration
                            )
                            cur_dice = test_metrics[metric_key]["dice"]
                            tm = test_metrics[metric_key]
                            print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.8f}", flush=True)
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}] Test losses: {view_tag}={tm['loss']}",
                                flush=True,
                            )
                            if "track_loss" in test_metrics:
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] Test Track losses: {view_tag}={test_metrics['track_loss']}",
                                    flush=True,
                                )
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}] Test Prec: {view_tag}={tm['prec']}",
                                flush=True,
                            )
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}] Test mIoU: {view_tag}={tm['miou']}",
                                flush=True,
                            )
                            print(
                                f"[Epoch {epoch+1}/{self.nb_epochs}] Test Dice: {view_tag}={tm['dice']}",
                                flush=True,
                            )
                            if cur_dice > best_test_dice and iteration > 0:
                                best_test_dice = cur_dice
                                print(
                                    f"[Epoch {epoch+1}/{self.nb_epochs}] New best test {view_tag} dice: {best_test_dice:.6f}. Saving checkpoint.",
                                    flush=True,
                                )
                                self.results["train_loss"] = float(train_loss)
                                self.results["train_ce_loss"] = float(train_ce_loss)
                                if nb_losses > 1:
                                    self.results["train_dice_loss"] = float(
                                        train_dice_loss
                                    )
                                if running_track_losses:
                                    self.results["train_track_loss"] = float(
                                        np.mean(running_track_losses)
                                    )
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
