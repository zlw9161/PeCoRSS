"""Validation / test loop for PeCoR-SV style datasets (PeCoRSVDataset)."""
import json

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from svrss.loaders.dataloaders_sv import PeCoRSVDataset
from svrss.utils.functions import (
    transform_masks_viz,
    get_metrics,
    normalize,
    define_loss,
    get_transformations,
    get_qualitatives,
    build_track_heatmap,
    resolve_class_weights_dir,
    resolve_tvt_stats_json_path,
)
from svrss.utils.metrics import Evaluator
from svrss.utils.paths import Paths


class TesterSV:
    """Evaluate a model on sequence loaders built with PeCoRSVDataset."""

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.nb_classes = self.cfg["nb_classes"]
        self.annot_type = self.cfg["annot_type"]
        self.signal_type = self.cfg["signal_type"]
        self.process_signal = self.cfg["process_signal"]
        self.w_size = self.cfg["w_size"]
        self.h_size = self.cfg["h_size"]
        self.n_frames = self.cfg["nb_input_channels"]
        self.batch_size = self.cfg["batch_size"]
        self.device = self.cfg["device"]
        self.custom_loss = self.cfg["custom_loss"]
        self.transform_names = self.cfg["transformations"].split(",")
        self.norm_type = self.cfg["norm_type"]
        self.paths = Paths().get()
        self.cmplx_flag = self.cfg.get("cmplx_flag", False)
        self.mulmask = bool(self.cfg.get("mulmask", False))
        self.track_gamma = float(self.cfg.get("track_gamma", 0.9))
        self.tgt = int(self.cfg.get("tgt", -1))
        self.test_results = dict()
        self.visualizer = visualizer

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

    def _sequence_frames_root(self, seq_name_0):
        paths = self.paths
        dl = (self.cfg.get("dataset") or "").lower()
        if dl in ("cwr", "cwrdata"):
            return paths["CWRdata"] / seq_name_0
        if dl in ("pdr", "pdrdata"):
            return paths["PDRdata"] / seq_name_0
        if dl == "carrada" or dl.startswith("carrada"):
            return paths["carrada"] / seq_name_0
        if dl in ("xpddata", "xpd"):
            return paths["XPDdata"] / seq_name_0
        if dl in ("rdsyn",):
            return paths["RDSyn"] / seq_name_0
        if dl in ("nuaasim",):
            return paths["NuaaSim"] / seq_name_0
        raise KeyError(
            f"tester_sv: unsupported cfg['dataset']={self.cfg.get('dataset')!r} for path resolution."
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

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=True):
        net.eval()
        transformations = get_transformations(
            self.transform_names, split="test", sizes=(self.w_size, self.h_size)
        )
        pe_type = self._peco_dataset_type()
        _w_dir = resolve_class_weights_dir(self.cfg, self.paths)

        if self.signal_type in "range_doppler":
            sv_criterion = define_loss(
                "range_doppler",
                self.custom_loss,
                self.device,
                weights_dir=_w_dir,
                cfg=self.cfg,
            )
        elif self.signal_type in "range_angle":
            sv_criterion = define_loss(
                "range_angle",
                self.custom_loss,
                self.device,
                weights_dir=_w_dir,
                cfg=self.cfg,
            )
        else:
            raise ValueError(self.signal_type)

        if not isinstance(sv_criterion, (list, tuple)):
            sv_criterion = [sv_criterion]
        nb_losses = len(sv_criterion)
        tvt_stats = resolve_tvt_stats_json_path(
            self.signal_type, self.cfg, self.paths
        )
        running_losses = []
        running_ce_losses = []
        running_dice_losses = []
        running_track_losses = []
        sv_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self._sequence_frames_root(seq_name[0])
                frame_dataloader = DataLoader(
                    PeCoRSVDataset(
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
                    ),
                    shuffle=False,
                    batch_size=min(self.batch_size, 8),
                    num_workers=min(self.cfg.get("num_workers", 4), 4),
                )
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_sv = self.n_frames - 1
                for j, frame in enumerate(frame_dataloader):
                    if self.signal_type in "range_doppler":
                        sv_data = frame["rd_matrix"].to(self.device).float()
                        sv_mask = frame["rd_mask"].to(self.device).float()
                        sv_data = normalize(
                            sv_data,
                            "range_doppler",
                            norm_type=self.norm_type,
                            stats_json_path=tvt_stats,
                        )
                    elif self.signal_type in "range_angle":
                        sv_data = frame["ra_matrix"].to(self.device).float()
                        sv_mask = frame["ra_mask"].to(self.device).float()
                        sv_data = normalize(
                            sv_data,
                            "range_angle",
                            norm_type=self.norm_type,
                            stats_json_path=tvt_stats,
                        )
                    else:
                        raise ValueError(self.signal_type)

                    net_outputs = net(sv_data)
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
                    if get_quali:
                        quali_iter_sv = get_qualitatives(
                            sv_outputs,
                            sv_tgt_mask,
                            self.paths,
                            seq_name,
                            quali_iter_sv,
                            self.signal_type,
                        )
                    sv_metrics.add_batch(
                        torch.argmax(sv_tgt_mask, axis=1).cpu(),
                        torch.argmax(sv_outputs, axis=1).cpu(),
                    )

                    sv_losses = [
                        c(sv_outputs, torch.argmax(sv_tgt_mask, axis=1)) for c in sv_criterion
                    ]
                    loss = torch.mean(torch.stack(sv_losses))
                    if self.mulmask and sv_mask.dim() == 5 and track_logits is not None:
                        track_gt = build_track_heatmap(
                            sv_mask, gamma=self.track_gamma, use_history=True
                        )
                        track_pred = torch.sigmoid(track_logits)
                        if track_pred.shape == track_gt.shape:
                            track_loss = F.smooth_l1_loss(track_pred, track_gt)
                            running_track_losses.append(float(track_loss.detach().cpu()))

                    running_losses.append(float(loss.detach().cpu()))
                    if nb_losses > 1:
                        running_ce_losses.append(float(sv_losses[0].detach().cpu()))
                        running_dice_losses.append(float(sv_losses[1].detach().cpu()))

                    if iteration and i == rand_seq and j == rand_frame:
                        sv_pred_masks = torch.argmax(sv_outputs, axis=1)[:5]
                        sv_gt_masks = torch.argmax(sv_tgt_mask, axis=1)[:5]
                        sv_pred_grid = make_grid(
                            transform_masks_viz(sv_pred_masks, self.nb_classes)
                        )
                        sv_gt_grid = make_grid(
                            transform_masks_viz(sv_gt_masks, self.nb_classes)
                        )
                        if self.visualizer is not None:
                            self.visualizer.update_img_masks(
                                sv_pred_grid, sv_gt_grid, iteration
                            )

            self.test_results = dict()
            mean_loss = float(np.mean(running_losses)) if running_losses else 0.0
            loss_tensor = torch.tensor(mean_loss)
            if nb_losses > 1:
                mean_ce = float(np.mean(running_ce_losses)) if running_ce_losses else 0.0
                mean_dice = float(np.mean(running_dice_losses)) if running_dice_losses else 0.0
                loss_parts = [torch.tensor(mean_ce), torch.tensor(mean_dice)]
            else:
                loss_parts = None
            if self.signal_type in "range_doppler":
                self.test_results["range_doppler"] = get_metrics(
                    sv_metrics,
                    loss_tensor,
                    loss_parts,
                )
            elif self.signal_type in "range_angle":
                self.test_results["range_angle"] = get_metrics(
                    sv_metrics,
                    loss_tensor,
                    loss_parts,
                )
            if running_track_losses:
                self.test_results["track_loss"] = float(np.mean(running_track_losses))
            sv_metrics.reset()
        return self.test_results

    def write_params(self, path):
        with open(path, "w") as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        self.device = device

    def set_annot_type(self, annot_type):
        self.annot_type = annot_type
