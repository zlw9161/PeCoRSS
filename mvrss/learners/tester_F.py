"""Class to test a model"""
# single view version
# merge model resume & cos lr strategy
# zlw@20230818
import gc
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from mvrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, \
    get_qualitatives, soft_mask_pdf, soft_mask_gaus, build_track_heatmap
from mvrss.utils.paths import Paths
from mvrss.utils.metrics import Evaluator
from mvrss.loaders.dataloaders_F import CarradaDataset


class Tester:
    """
    Class to test a model

    PARAMETERS
    ----------
    cfg: dict
        Configuration parameters used for train/test
    visualizer: object or None
        Add a visulization during testing
        Default: None
    """

    def __init__(self, cfg, visualizer=None):
        self.cfg = cfg
        self.model = self.cfg['model']
        self.nb_classes = self.cfg['nb_classes']
        self.annot_type = self.cfg['annot_type']
        self.signal_type = self.cfg['signal_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.soft_trackloss = self.cfg['soft_trackloss']
        self.flag_coherence_track_loss = self.cfg['ct_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()
        self.visualizer = visualizer
        self.mulmask = self.cfg['mulmask']
        self.tgt = self.cfg['tgt']
        self.rd_weight = self.cfg['rd_weight']
        self.ra_weight = self.cfg['ra_weight']
        self.lamda_flag = self.cfg["lamda_flag"]
        self.num_workers = int(self.cfg.get('num_workers', 2))
        self.track_gamma = float(self.cfg.get('track_gamma', 0.9))
        # Optional: keep validation/test runs short for debugging on Windows.
        # Set in cfg: diag_max_seqs (int), diag_max_frames (int)
        self.diag_max_seqs = self.cfg.get('diag_max_seqs', None)
        self.diag_max_frames = self.cfg.get('diag_max_frames', None)
        self.diag_max_seqs = int(self.diag_max_seqs) if self.diag_max_seqs is not None else None
        self.diag_max_frames = int(self.diag_max_frames) if self.diag_max_frames is not None else None
        # Optional: progress printing to avoid "silent" validation.
        # Set in cfg: progress_every_seqs (int), progress_every_frames (int)
        self.progress_every_seqs = self.cfg.get('progress_every_seqs', None)
        self.progress_every_frames = self.cfg.get('progress_every_frames', None)
        self.progress_every_seqs = int(self.progress_every_seqs) if self.progress_every_seqs is not None else None
        self.progress_every_frames = int(self.progress_every_frames) if self.progress_every_frames is not None else None

    def _progress(self, msg: str) -> None:
        if self.progress_every_seqs is not None or self.progress_every_frames is not None:
            print(msg, flush=True)

    def _mask_tgt_frame(self, mask_seq: torch.Tensor) -> torch.Tensor:
        """(B,C,T,H,W) -> (B,C,H,W) at cfg tgt; passthrough if already (B,C,H,W)."""
        if mask_seq.dim() == 5:
            return mask_seq[:, :, self.tgt, :, :]
        return mask_seq

    def predict(self, net, seq_loader, lamda=None, iteration=None, get_quali=False, add_temp=True):
        """
        Method to predict on a given dataset using a fixed model

        PARAMETERS
        ----------
        net: PyTorch Model
            Network to test
        seq_loader: DataLoader
            Specific to the dataset used for test
        iteration: int
            Iteration used to display visualization
            Default: None
        get_quali: boolean
            If you want to save qualitative results
            Default: False
        add_temp: boolean
            Is the data are considered as a sequence
            Default: False
        """
        net.eval()
        transformations = get_transformations(self.transform_names, split='test',
                                              sizes=(self.w_size, self.h_size))
        if self.signal_type == 'range_doppler':
            rd_criterion = define_loss('range_doppler', self.custom_loss, self.device)
            nb_losses = len(rd_criterion)
        elif self.signal_type == 'range_angle':
            ra_criterion = define_loss('range_angle', self.custom_loss, self.device, view_weight=1.)
            nb_losses = len(ra_criterion)
        elif self.signal_type == 'range_doppler_angle':
            rd_criterion = define_loss('range_doppler', self.custom_loss, self.device, view_weight=self.rd_weight)
            ra_criterion = define_loss('range_angle', self.custom_loss, self.device, view_weight=self.ra_weight)
            nb_losses = len(rd_criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        ra_running_losses = list()
        ra_running_global_losses = [list(), list()]
        coherence_running_losses = list()
        track_running_losses = list()
        rd_metrics = Evaluator(num_class=self.nb_classes)
        ra_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.inference_mode():
            for i, sequence_data in enumerate(seq_loader):
                if self.diag_max_seqs is not None and i >= self.diag_max_seqs:
                    break
                if self.progress_every_seqs is not None and i % max(self.progress_every_seqs, 1) == 0:
                    try:
                        total = len(seq_loader)
                        self._progress(f"[Predict] seq {i}/{total} ...")
                    except Exception:
                        self._progress(f"[Predict] seq {i} ...")
                seq_name, seq = sequence_data
                path_to_frames = self.paths['carrada'] / seq_name[0]
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp,
                                                             self.mulmask),
                                              shuffle=False,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers)
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_sv = self.n_frames - 1
                for j, frame in enumerate(frame_dataloader):
                    if self.diag_max_frames is not None and j >= self.diag_max_frames:
                        break
                    if self.progress_every_frames is not None and j % max(self.progress_every_frames, 1) == 0:
                        self._progress(f"[Predict]   frame {j} ...")
                    if self.signal_type == 'range_doppler':
                        rd_data = frame['rd_matrix'].to(self.device).float()
                        rd_mask = frame['rd_mask'].to(self.device).float()
                        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                        # Use the model passed as argument; `Tester` does not own `self.net`.
                        rd_outputs, x_pre = net(rd_data)
                        rd_outputs = rd_outputs.to(self.device)
                        if get_quali:
                            quali_iter_rd = get_qualitatives(
                                rd_outputs, self._mask_tgt_frame(rd_mask), self.paths,
                                seq_name, quali_iter_rd, 'range_doppler')
                        rd_gt = self._mask_tgt_frame(rd_mask)
                        if self.mulmask:
                            rd_metrics.add_batch(torch.argmax(rd_gt, axis=1).cpu(),
                                                 torch.argmax(rd_outputs, axis=1).cpu())
                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion]
                                loss = torch.mean(torch.stack(rd_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion[:2]]
                                loss = torch.mean(torch.stack(rd_losses))
                            x_mask = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask[x_mask > 1] = 1  # (b,4,h,w)
                            track_loss = rd_criterion[0](x_pre, torch.argmax(x_mask, axis=1))
                            loss = torch.mean(loss + track_loss)
                        else:
                            rd_metrics.add_batch(torch.argmax(rd_gt, axis=1).cpu(),
                                                 torch.argmax(rd_outputs, axis=1).cpu())

                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion]
                                loss = torch.mean(torch.stack(rd_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion[:2]]
                                loss = torch.mean(torch.stack(rd_losses))

                        running_losses.append(loss.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])

                    elif self.signal_type == 'range_angle':
                        ra_data = frame['ra_matrix'].to(self.device).float()
                        ra_mask = frame['ra_mask'].to(self.device).float()
                        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                        ra_outputs, x_pre = net(ra_data)
                        ra_outputs = ra_outputs.to(self.device)
                        if get_quali:
                            quali_iter_rd = get_qualitatives(
                                ra_outputs, self._mask_tgt_frame(ra_mask), self.paths,
                                seq_name, quali_iter_rd, 'range_angle')
                        ra_gt = self._mask_tgt_frame(ra_mask)
                        if self.mulmask:
                            ra_metrics.add_batch(torch.argmax(ra_gt, axis=1).cpu(),
                                                 torch.argmax(ra_outputs, axis=1).cpu())
                            if nb_losses < 3:
                                # Case without the CoL
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion]
                                loss = torch.mean(torch.stack(ra_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion[:2]]
                                loss = torch.mean(torch.stack(ra_losses))
                            x_mask = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                            x_mask[x_mask > 1] = 1  # (b,4,h,w)
                            track_loss = ra_criterion[0](x_pre, torch.argmax(x_mask, axis=1))
                            loss = torch.mean(loss + track_loss)
                        else:
                            ra_metrics.add_batch(torch.argmax(ra_gt, axis=1).cpu(),
                                                 torch.argmax(ra_outputs, axis=1).cpu())

                            if nb_losses < 3:
                                # Case without the CoL
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion]
                                loss = torch.mean(torch.stack(ra_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion[:2]]
                                loss = torch.mean(torch.stack(ra_losses))

                        running_losses.append(loss.data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                    elif self.signal_type == 'range_doppler_angle':
                        rd_data = frame['rd_matrix'].to(self.device).float()
                        ra_data = frame['ra_matrix'].to(self.device).float()
                        ad_data = frame['ad_matrix'].to(self.device).float()
                        rd_mask = frame['rd_mask'].to(self.device).float()
                        ra_mask = frame['ra_mask'].to(self.device).float()
                        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                        # zlw@20220325
                        if self.model != 'mvnet':
                            ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                            if self.lamda_flag == 1:
                                rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = net(rd_data, ra_data, ad_data, lamda)
                            else:
                                rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = net(rd_data, ra_data, ad_data)
                        else:
                            rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = net(rd_data, ra_data)
                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)
                        rd_gt = self._mask_tgt_frame(rd_mask)
                        ra_gt = self._mask_tgt_frame(ra_mask)
                        if get_quali:
                            quali_iter_rd = get_qualitatives(rd_outputs, rd_gt, self.paths,
                                                             seq_name, quali_iter_rd, 'range_doppler')
                            quali_iter_ra = get_qualitatives(ra_outputs, ra_gt, self.paths,
                                                             seq_name, quali_iter_ra, 'range_angle')
                        if self.mulmask:
                            rd_metrics.add_batch(torch.argmax(rd_gt, axis=1).cpu(),
                                                 torch.argmax(rd_outputs, axis=1).cpu())
                            ra_metrics.add_batch(torch.argmax(ra_gt, axis=1).cpu(),
                                                 torch.argmax(ra_outputs, axis=1).cpu())
                            if nb_losses < 4:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                loss = torch.mean(rd_loss + ra_loss)
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                # Coherence loss
                                coherence_loss = rd_criterion[3](rd_outputs, ra_outputs)
                                if self.flag_coherence_track_loss:
                                    coherence_track_loss = rd_criterion[2](rd_track_logits, ra_track_logits)
                                    coherence_loss = torch.mean(coherence_track_loss + coherence_track_loss)
                                loss = torch.mean(20.*rd_loss + 15.*ra_loss + coherence_loss)
                            x_mask_rd = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask_rd[:, 0, ...][x_mask_rd[:, 0, ...] < self.n_frames] = 0
                            x_mask_rd[x_mask_rd > 0] = 1  # (b,4,h,w)
                            #track_loss_rd = rd_criterion[0](x_pre_rd, torch.argmax(x_mask_rd, axis=1))
                            x_mask_ra = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                            x_mask_ra[:, 0, ...][x_mask_ra[:, 0, ...] < self.n_frames] = 0
                            x_mask_ra[x_mask_ra > 0] = 1  # (b,4,h,w)
                            #track_loss_ra = ra_criterion[0](x_pre_ra, torch.argmax(x_mask_ra, axis=1))
                            # Track heatmap regression for evaluation (matches training semantics)
                            track_gt_rd = build_track_heatmap(rd_mask, gamma=self.track_gamma, use_history=True)
                            track_gt_ra = build_track_heatmap(ra_mask, gamma=self.track_gamma, use_history=True)
                            track_pred_rd = torch.sigmoid(rd_track_logits)
                            track_pred_ra = torch.sigmoid(ra_track_logits)
                            track_loss_rd = torch.nn.functional.smooth_l1_loss(track_pred_rd, track_gt_rd)
                            track_loss_ra = torch.nn.functional.smooth_l1_loss(track_pred_ra, track_gt_ra)
                            track_loss = torch.mean(track_loss_ra + track_loss_rd)
                            loss = torch.mean(loss + track_loss)
                        else:
                            rd_metrics.add_batch(torch.argmax(rd_gt, axis=1).cpu(),
                                                 torch.argmax(rd_outputs, axis=1).cpu())
                            ra_metrics.add_batch(torch.argmax(ra_gt, axis=1).cpu(),
                                                 torch.argmax(ra_outputs, axis=1).cpu())
                            if nb_losses < 4:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                loss = torch.mean(rd_loss + ra_loss)
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_gt, axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_gt, axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                # Coherence loss
                                coherence_loss = rd_criterion[3](rd_outputs, ra_outputs)
                                loss = torch.mean(rd_loss + ra_loss + coherence_loss)

                        running_losses.append(loss.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])

                        if nb_losses > 3:
                            coherence_running_losses.append(coherence_loss.data.cpu().numpy()[()])
                    if self.mulmask:
                        track_running_losses.append(track_loss.data.cpu().numpy()[()])


                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            if self.signal_type == 'range_doppler':
                                rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                                rd_gt_masks = torch.argmax(self._mask_tgt_frame(rd_mask), axis=1)[:5]
                                rd_pred_grid = make_grid(transform_masks_viz(rd_pred_masks,
                                                                             self.nb_classes))
                                rd_gt_grid = make_grid(transform_masks_viz(rd_gt_masks,
                                                                           self.nb_classes))
                                self.visualizer.update_img_masks(rd_pred_grid, rd_gt_grid, iteration)
                            elif self.signal_type == 'range_angle':
                                ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                                ra_gt_masks = torch.argmax(self._mask_tgt_frame(ra_mask), axis=1)[:5]
                                ra_pred_grid = make_grid(transform_masks_viz(ra_pred_masks,
                                                                             self.nb_classes))
                                ra_gt_grid = make_grid(transform_masks_viz(ra_gt_masks,
                                                                           self.nb_classes))
                                self.visualizer.update_img_masks(ra_pred_grid, ra_gt_grid, iteration)
                            elif self.signal_type == 'range_doppler_angle':
                                rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                                ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                                rd_gt_masks = torch.argmax(self._mask_tgt_frame(rd_mask), axis=1)[:5]
                                ra_gt_masks = torch.argmax(self._mask_tgt_frame(ra_mask), axis=1)[:5]

                                rd_pred_grid = make_grid(transform_masks_viz(rd_pred_masks,
                                                                             self.nb_classes))
                                ra_pred_grid = make_grid(transform_masks_viz(ra_pred_masks,
                                                                             self.nb_classes))
                                rd_gt_grid = make_grid(transform_masks_viz(rd_gt_masks,
                                                                           self.nb_classes))
                                ra_gt_grid = make_grid(transform_masks_viz(ra_gt_masks,
                                                                           self.nb_classes))
                                self.visualizer.update_multi_img_masks(rd_pred_grid, rd_gt_grid,
                                                                       ra_pred_grid, ra_gt_grid,
                                                                       iteration)
                # Release batch-scoped references promptly to keep validation/test memory stable.
                try:
                    del loss
                except (UnboundLocalError, NameError):
                    pass
                if self.signal_type == 'range_doppler_angle':
                    try:
                        del (
                            rd_outputs, ra_outputs, rd_track_logits, ra_track_logits,
                            rd_losses, rd_loss, ra_losses, ra_loss,
                            track_gt_rd, track_gt_ra, track_pred_rd, track_pred_ra,
                            track_loss_rd, track_loss_ra, track_loss,
                            rd_data, ra_data, ad_data, rd_mask, ra_mask
                        )
                    except (UnboundLocalError, NameError):
                        pass
                del frame_dataloader

            self.test_results = dict()
            if self.signal_type == 'range_doppler':
                self.test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                                 [np.mean(sub_loss) for sub_loss
                                                                  in rd_running_global_losses])
                rd_metrics.reset()
            elif self.signal_type == 'range_angle':
                self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
                                                               [np.mean(sub_loss) for sub_loss
                                                                in ra_running_global_losses])
                ra_metrics.reset()
            elif self.signal_type == 'range_doppler_angle':
                self.test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                                 [np.mean(sub_loss) for sub_loss
                                                                  in rd_running_global_losses])
                self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
                                                               [np.mean(sub_loss) for sub_loss
                                                                in ra_running_global_losses])

                if nb_losses > 3:
                    self.test_results['coherence_loss'] = np.mean(coherence_running_losses).item()
                self.test_results['track_loss'] = np.mean(track_running_losses).item()
                self.test_results['global_acc'] = (1 / 2) * (self.test_results['range_doppler']['acc'] +
                                                             self.test_results['range_angle']['acc'])
                self.test_results['global_prec'] = (1 / 2) * (self.test_results['range_doppler']['prec'] +
                                                              self.test_results['range_angle']['prec'])
                self.test_results['global_dice'] = (1 / 2) * (self.test_results['range_doppler']['dice'] +
                                                              self.test_results['range_angle']['dice'])
                # self.test_results['global_dice'] = (self.rd_weight *(self.test_results['range_doppler']['dice'])+
                #                                 self.ra_weight * self.test_results['range_angle']['dice'])/(self.rd_weight+self.ra_weight)
                # add global recall and mIoU zlw@20220704
                self.test_results['global_recall'] = (1 / 2) * (self.test_results['range_doppler']['recall'] +
                                                                self.test_results['range_angle']['recall'])
                self.test_results['global_miou'] = (1 / 2) * (self.test_results['range_doppler']['miou'] +
                                                              self.test_results['range_angle']['miou'])
                rd_metrics.reset()
                ra_metrics.reset()

        if self.signal_type == 'range_doppler_angle':
            try:
                del rd_data, ra_data, ad_data, rd_mask, ra_mask
            except (UnboundLocalError, NameError):
                pass
        gc.collect()
        if self.device == 'cuda':
            torch.cuda.empty_cache()
        return self.test_results

    def write_params(self, path):
        """Write quantitative results of the Test"""
        with open(path, 'w') as fp:
            json.dump(self.test_results, fp)

    def set_device(self, device):
        """Set device used for test (supported: 'cuda', 'cpu')"""
        self.device = device

    def set_annot_type(self, annot_type):
        """Set annotation type to test on (specific to CARRADA)"""
        self.annot_type = annot_type
