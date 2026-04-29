"""Class to train a PyTorch model"""
import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from mvrss.loaders.dataloaders_F import CarradaDataset
from mvrss.learners.tester_F import Tester
from mvrss.utils.functions import normalize, define_loss, get_transformations, soft_mask_pdf, soft_mask_gaus, build_track_heatmap
from mvrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer

from thop import profile
from thop import clever_format
import time
# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"



def _tfg_dbg(msg: str) -> None:
    """Set TFOC_DEBUG=1 for step prints (see train_F.py)."""
    if os.environ.get("TFOC_DEBUG", "").strip() in ("1", "true", "True", "yes", "YES"):
        print(f"[TFOC_DEBUG] {msg}", flush=True)


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data, Mulmask):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.model_name = self.cfg['model']
        self.resume_path = self.cfg['resume']
        self.process_signal = self.cfg['process_signal']
        self.signal_type = self.cfg['signal_type']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.s = self.cfg["bias_s"]
        self.lamda_flag = self.cfg["lamda_flag"]
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.schedular_type = self.cfg['schedular']
        self.T_max = self.cfg['T_max']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.flag_coherence_track_loss = self.cfg['ct_loss']
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.soft_trackloss = self.cfg['soft_trackloss']
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.mulmask = Mulmask
        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()
        self.erstp = self.cfg['erstp']
        self.tgt = self.cfg['tgt']
        self.rd_weight = self.cfg['rd_weight']
        self.ra_weight = self.cfg['ra_weight']
        self.num_workers = int(self.cfg.get('num_workers', 2))
        self.track_gamma = float(self.cfg.get('track_gamma', 0.9))
        self.track_weight = float(self.cfg.get('track_weight', 0.5))
        self.rd_loss_weight = float(self.cfg.get('rd_loss_weight', 1.0))
        self.ra_loss_weight = float(self.cfg.get('ra_loss_weight', 1.0))
        self.warmup = bool(self.cfg.get('warmup', False))
        self.warmup_factor = float(self.cfg.get('warmup_factor', 0.001))
        self.warmup_iters = int(self.cfg.get('warmup_iters', 2000))
        self.use_ema = bool(self.cfg.get('use_ema', False))
        self.ema_decay = float(self.cfg.get('ema_decay', 0.999))
        self.ema_update_after_step = int(self.cfg.get('ema_update_after_step', 100))
        self._ema_state = None

    def calc_alpha_bar(self, current_epoch, total_epoch, s):
        """
        实现ᾱₜ随epoch变化的余弦调度闭式公式
        :param current_epoch: 当前epoch数（标量，或批量的epoch数组），支持t从0到total_epoch
        :param total_epoch: 训练总epoch数 T
        :param s: 全局偏移超参数
        :return: 当前epoch对应的ᾱₜ值（标量/数组，和输入格式一致）
        """
        # 分母项：cos( (π/2) * s/(1+s) )
        denominator = np.cos((np.pi / 2) * (s / (1 + s)))
        # 分子项：cos( (π/2) * ( (t/T) + s )/(1+s) )
        numerator = np.cos((np.pi / 2) * ((current_epoch / total_epoch) + s) / (1 + s))
        # 最终ᾱₜ = (分子/分母)²
        alpha_bar_t = (numerator / denominator) ** 2

        return alpha_bar_t

    def _apply_warmup_lr(self, optimizer, iteration):
        """Linear warmup from warmup_factor -> 1.0 over warmup_iters."""
        if not self.warmup or self.warmup_iters <= 0:
            return
        if iteration >= self.warmup_iters:
            return

        alpha = float(iteration) / float(self.warmup_iters)
        warm_mult = self.warmup_factor * (1.0 - alpha) + alpha
        for param_group in optimizer.param_groups:
            param_group['lr'] = self.lr * warm_mult

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

    def _update_ema(self, iteration):
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
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """
        start_epoch = 0
        _tfg_dbg("Model.train: enter")
        self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        _tfg_dbg("Model.train: SummaryWriter + dataloaders ready")
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        self.net.apply(self._init_weights)
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
        iteration = 0
        best_val_prec = 0
        # saving results w/ best recall zlw@20220704
        best_val_recall = 0
        best_test_dice = 0
        patience_counter = 0
        checkpoint = None
        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr) #weight_decay=0.02
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max - 1, eta_min=1e-7)
        elif self.schedular_type == 'red':
            scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True, threshold=1e-4, min_lr=1e-7, cooldown=2)
        else:
            raise KeyError("we only implement schedular of exp, cos and red")
        if self.resume_path:
            checkpoint = torch.load(self.resume_path)
            start_epoch = checkpoint['epoch']
            # net = load_my_state_dict(net, checkpoint['state_dict'])
            self.net.load_state_dict(checkpoint['net'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['lr_scheduler'])
            best_dice = checkpoint['best_dice']
            print("=> Loaded checkpoint at epoch {}\n)".format(checkpoint['epoch']))
            print("=> Initial learning rate is {}\n)".format(checkpoint['lr_scheduler']))
            print("=> Best dice is {})".format(checkpoint['best_dice']))

        self._init_ema_state()
        if checkpoint is not None:
            self._restore_ema_from_checkpoint(checkpoint.get('ema_state'))

        for epoch in range(start_epoch, self.nb_epochs):
            if self.lamda_flag == 1:
                lamda = self.calc_alpha_bar(epoch, self.nb_epochs, self.s)
                print("lamda = {} in epoch {}\n".format(lamda, epoch))
            else:
                lamda = None
            _tfg_dbg(f"epoch {epoch}: about to iterate outer train_loader (sequences)")
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            elif self.schedular_type == 'cos':
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            for si, sequence_data in enumerate(train_loader):
                if si == 0:
                    _tfg_dbg(f"epoch {epoch}: first sequence yielded from train_loader")
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['carrada'], seq_name[0])
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp,
                                                             self.mulmask),
                                              shuffle=self.is_shuffled,
                                              batch_size=self.batch_size,
                                              num_workers=self.num_workers)

                for fi, frame in enumerate(frame_dataloader):
                    if si == 0 and fi == 0:
                        _tfg_dbg(f"epoch {epoch}: first frame batch from CarradaDataset (inner DataLoader)")
                    self._apply_warmup_lr(optimizer, iteration)

                    if self.signal_type == 'range_doppler':
                        rd_data = frame['rd_matrix'].to(self.device).float()
                        rd_mask = frame['rd_mask'].to(self.device).float()
                        rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                        optimizer.zero_grad()
                        rd_outputs, x_pre = self.net(rd_data)
                        rd_outputs = rd_outputs.to(self.device)
                        if self.mulmask:
                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                             for c in rd_criterion]
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                             for c in rd_criterion[:2]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            x_mask = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask[x_mask > 1] = 1  # (b,4,h,w)
                            track_loss = rd_criterion[0](x_pre, torch.argmax(x_mask, axis=1))
                            loss = torch.mean(rd_loss + track_loss)
                        else:
                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion]
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion[:2]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            loss = rd_loss
                        loss.backward()
                        optimizer.step()
                        self._update_ema(iteration)
                        running_losses.append(loss.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                    elif self.signal_type == 'range_angle':
                        ra_data = frame['ra_matrix'].to(self.device).float()
                        ra_mask = frame['ra_mask'].to(self.device).float()
                        ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                        optimizer.zero_grad()
                        ra_outputs, x_pre = self.net(ra_data)
                        ra_outputs = ra_outputs.to(self.device)
                        if self.mulmask:
                            if nb_losses < 3:
                                # Case without the CoL
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, -1, :, :], axis=1))
                                             for c in ra_criterion]
                                loss = torch.mean(torch.stack(ra_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, -1, :, :], axis=1))
                                             for c in ra_criterion[:2]]
                                loss = torch.mean(torch.stack(ra_losses))
                            x_mask = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                            x_mask[x_mask > 1] = 1  # (b,4,h,w)
                            track_loss = ra_criterion[0](x_pre, torch.argmax(x_mask, axis=1))
                            loss = torch.mean(loss + track_loss)
                        else:
                            if nb_losses < 3:
                                # Case without the CoL
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                             for c in ra_criterion]
                                loss = torch.mean(torch.stack(ra_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                             for c in ra_criterion[:2]]
                                loss = torch.mean(torch.stack(ra_losses))
                        loss.backward()
                        optimizer.step()
                        self._update_ema(iteration)
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
                        if self.model_name != 'mvnet':
                            ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                        optimizer.zero_grad()
                        if self.model_name != 'mvnet':

                            # ##### FLOPs calculation #######
                            # flops, params = profile(self.net, inputs=(rd_data, ra_data, ad_data,))
                            # macs, params = clever_format([flops, params], "%.3f")
                            # print("Macs:{},params:{}".format(macs, params))
                            # #########################################################
                            # time_start = time.time()
                            ##tfoc###
                            if self.lamda_flag == 1:
                                rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = self.net(rd_data, ra_data, ad_data, lamda)
                            else:
                                rd_outputs, ra_outputs, rd_track_logits, ra_track_logits = self.net(rd_data, ra_data, ad_data)
                            ##other model##
                            # rd_outputs, ra_outputs = self.net(rd_data, ra_data, ad_data)
                            #########
                            # torch.cuda.synchronize()
                            # time_end = time.time()
                            # print("Time:{}".format(time_end - time_start))
                            # ##########################################################
                            # total_time += time_end - time_start
                            # count += 1
                            # if count == 200:
                            #     print('#####################################')
                            #     print('mean_time:{}'.format(total_time / count))
                            #     print('#####################################')
                        else:
                            rd_outputs, ra_outputs, rd_probs, ra_probs = self.net(rd_data, ra_data)

                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)
                        if self.mulmask:
                            x_mask_rd = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask_rd[:,0,...][x_mask_rd[:,0,...] < self.n_frames] = 0
                            x_mask_rd[x_mask_rd > 0] = 1  # (b,4,h,w)
                            x_mask_ra = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                            x_mask_ra[:,0,...][x_mask_ra[:, 0, ...] < self.n_frames] = 0
                            x_mask_ra[x_mask_ra > 0] = 1  # (b,4,h,w)
                            # Track heatmap regression (per-class), supervised from history masks.
                            # Predict: sigmoid(track_logits) in [0,1]
                            # Target: temporally-decayed heatmap built from frames [0..T-2]
                            track_gt_rd = build_track_heatmap(rd_mask, gamma=self.track_gamma, use_history=True)
                            track_gt_ra = build_track_heatmap(ra_mask, gamma=self.track_gamma, use_history=True)
                            track_pred_rd = torch.sigmoid(rd_track_logits)
                            track_pred_ra = torch.sigmoid(ra_track_logits)
                            track_loss_rd = F.smooth_l1_loss(track_pred_rd, track_gt_rd)
                            track_loss_ra = F.smooth_l1_loss(track_pred_ra, track_gt_ra)
                            track_loss = torch.mean(track_loss_ra + track_loss_rd)
                            if nb_losses < 4:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, self.tgt, :, :], axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, self.tgt, :, :], axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                base_loss = self.rd_loss_weight * rd_loss + self.ra_loss_weight * ra_loss
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, self.tgt, :, :], axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, self.tgt, :, :], axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                # Coherence loss
                                coherence_loss = rd_criterion[3](rd_outputs, ra_outputs)
                                if self.flag_coherence_track_loss:
                                    coherence_track_loss = rd_criterion[3](rd_track_logits, ra_track_logits)
                                    coherence_loss = torch.mean(coherence_loss + coherence_track_loss)
                                base_loss = (
                                    self.rd_loss_weight * rd_loss
                                    + self.ra_loss_weight * ra_loss
                                    + 0.5 * coherence_loss
                                )
                            # Add track loss with configurable weight
                            loss = torch.mean(base_loss + self.track_weight * track_loss)
                            # del rd_outputs, ra_outputs, rd_probs, ra_probs
                            # torch.cuda.empty_cache()  # 清理显存
                        else:
                            if nb_losses < 4:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                loss = torch.mean(
                                    self.rd_loss_weight * rd_loss + self.ra_loss_weight * ra_loss
                                )
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                # Coherence loss
                                coherence_loss = rd_criterion[3](rd_outputs, ra_outputs)
                                loss = torch.mean(
                                    self.rd_loss_weight * rd_loss
                                    + self.ra_loss_weight * ra_loss
                                    + coherence_loss
                                )
                            # del rd_outputs, ra_outputs, rd_probs, ra_probs
                            # torch.cuda.empty_cache()  # 清理显存
                        loss.backward()
                        optimizer.step()
                        self._update_ema(iteration)
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

                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        if self.signal_type == 'range_doppler':
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train loss {}'.format(epoch + 1,
                                                         self.nb_epochs,
                                                         iteration,
                                                         train_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train losses: RD={}'.format(epoch + 1,
                                                                      self.nb_epochs,
                                                                      iteration,
                                                                      rd_train_loss))

                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, iteration)

                            running_losses = list()
                            rd_running_losses = list()
                            rd_running_global_losses = [list(), list()]
                        elif self.signal_type == 'range_angle':
                            ra_train_loss = np.mean(ra_running_losses)
                            ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train loss {}'.format(epoch + 1,
                                                         self.nb_epochs,
                                                         iteration,
                                                         train_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train losses: RA={}'.format(epoch + 1,
                                                                      self.nb_epochs,
                                                                      iteration,
                                                                      ra_train_loss))

                            self.visualizer.update_multi_train_loss(train_loss, ra_train_loss,
                                                                        ra_train_losses, iteration)
                            running_losses = list()
                            ra_running_losses = list()
                            ra_running_global_losses = [list(), list()]
                        elif self.signal_type == 'range_doppler_angle':
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            ra_train_loss = np.mean(ra_running_losses)
                            ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                            print('[Epoch {}/{}, iter {}]: '
                                'Train loss {}'.format(epoch+1,
                                                        self.nb_epochs,
                                                         iteration,
                                                        train_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                'Train losses: RD={}, RA={}'.format(epoch+1,
                                                                  self.nb_epochs,
                                                                  iteration,
                                                                  rd_train_loss,
                                                                  ra_train_loss))

                            if self.mulmask:
                                track_train_loss = np.mean(track_running_losses)
                                print('[Epoch {}/{}, iter {}]: '
                                      'Train Track loss {}'.format(epoch + 1,
                                                                       self.nb_epochs,
                                                                       iteration,
                                                                       track_train_loss))
                            if nb_losses > 3:
                                coherence_train_loss = np.mean(coherence_running_losses)
                                print('[Epoch {}/{}, iter {}]: '
                                      'Train Coherence loss {}'.format(epoch + 1,
                                                                       self.nb_epochs,
                                                                       iteration,
                                                                       coherence_train_loss))
                                self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, ra_train_loss,
                                                                        ra_train_losses, iteration,
                                                                        coherence_train_loss)
                            else:
                                self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, ra_train_loss,
                                                                        ra_train_losses, iteration)

                            running_losses = list()
                            rd_running_losses = list()
                            ra_running_losses = list()
                            rd_running_global_losses = [list(), list()]
                            ra_running_global_losses = [list(), list()]
                            coherence_running_losses = list()
                            track_running_losses = list()
                        # zlw@20220107 get_lr() --> get_last_lr()
                        #self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        ema_backup = self._swap_in_ema_weights()
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp)
                        else:
                            if self.lamda_flag == 1:
                                val_metrics = self.tester.predict(self.net, val_loader, lamda=lamda, add_temp=add_temp)
                            else:
                                val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                        if self.signal_type == 'range_doppler':
                            print('[Epoch {}/{}] Val losses: '
                                  'RD={}'.format(epoch + 1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['loss']))
                            if self.mulmask:
                                print('[Epoch {}/{}] Val Track losses: '
                                      'RD={}'.format(epoch + 1,
                                                     self.nb_epochs,
                                                     val_metrics['track_loss']))
                            print('[Epoch {}/{}] Val Pixel Prec: '
                                  'RD={}'.format(epoch + 1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['prec']))
                            print('[Epoch {}/{}] Val mIoU: '
                                  'RD={}'.format(epoch + 1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['miou']))
                            print('[Epoch {}/{}] Val Dice: '
                                  'RD={}'.format(epoch + 1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['dice']))
                        elif self.signal_type == 'range_angle':
                            print('[Epoch {}/{}] Val losses: '
                                  'RA={}'.format(epoch + 1,
                                                 self.nb_epochs,
                                                 val_metrics['range_angle']['loss'], ))
                            if self.mulmask:
                                print('[Epoch {}/{}] Val Track losses: '
                                      'RA={}'.format(epoch + 1,
                                                     self.nb_epochs,
                                                     val_metrics['track_loss']))
                            print('[Epoch {}/{}] Val Pixel Prec: '
                                  'RA={}'.format(epoch + 1,
                                                    self.nb_epochs,
                                                    val_metrics['range_angle']['prec']))
                            print('[Epoch {}/{}] Val mIoU: '
                                  ' RA={}'.format(epoch + 1,
                                                    self.nb_epochs,
                                                    val_metrics['range_angle']['miou']))
                            print('[Epoch {}/{}] Val Dice: '
                                  'RA={}'.format(epoch + 1,
                                                    self.nb_epochs,
                                                    val_metrics['range_angle']['dice']))
                        elif self.signal_type == 'range_doppler_angle':
                            print('[Epoch {}/{}] Val losses: '
                                'RD={}, RA{}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['range_doppler']['loss'],
                                                    val_metrics['range_angle']['loss'],))
                            if nb_losses > 3:
                                print('[Epoch {}/{}] Val Coherence Loss: '
                                'CoL={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['coherence_loss']))
                            if self.mulmask:
                                print('[Epoch {}/{}] Val Track losses: '
                                      'RDA={}'.format(epoch + 1,
                                                     self.nb_epochs,
                                                     val_metrics['track_loss']))
                            print('[Epoch {}/{}] Val Pixel Prec: '
                                'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['range_doppler']['prec'],
                                                    val_metrics['range_angle']['prec']))
                            print('[Epoch {}/{}] Val mIoU: '
                                'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['range_doppler']['miou'],
                                                    val_metrics['range_angle']['miou']))
                            print('[Epoch {}/{}] Val Dice: '
                                'RD={}, RA={}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['range_doppler']['dice'],
                                                    val_metrics['range_angle']['dice']))
                            #scheduler.step(val_metrics['global_dice'])

                        if self.device == 'cuda':
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()

                        '''if val_metrics['global_prec'] > best_val_prec and iteration > 0:
                            best_val_prec = val_metrics['global_prec']
                            '''
                        # saving results w/ best dice zlw@20220704
                        test_metrics  = self.tester.predict(self.net, test_loader, lamda=lamda,
                                                               add_temp=add_temp)

                        # print learning rate
                        print('Learning Rate: {:.8f}'.format(optimizer.param_groups[0]['lr']))

                        if test_metrics['global_dice'] > best_test_dice and iteration > 0:
                            best_test_dice = test_metrics['global_dice']
                            if self.signal_type == 'range_doppler':
                                print('[Epoch {}/{}] Test losses: '
                                      'RD={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_doppler']['loss']))
                                print('[Epoch {}/{}] Test Prec: '
                                      'RD={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_doppler']['prec']))
                                print('[Epoch {}/{}] Test mIoU: '
                                      'RD={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_doppler']['miou']))
                                print('[Epoch {}/{}] Test Dice: '
                                      'RD={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_doppler']['dice']))

                                self.results['rd_train_loss'] = rd_train_loss.item()

                            elif self.signal_type == 'range_angle':
                                print('[Epoch {}/{}] Test losses: '
                                      'RA={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_angle']['loss']))
                                print('[Epoch {}/{}] Test Prec: '
                                      'RA={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_angle']['prec']))
                                print('[Epoch {}/{}] Test mIoU: '
                                      'RA={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_angle']['miou']))
                                print('[Epoch {}/{}] Test Dice: '
                                      'RA={}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            test_metrics['range_angle']['dice']))
                                self.results['ra_train_loss'] = ra_train_loss.item()

                            elif self.signal_type == 'range_doppler_angle':
                                print('[Epoch {}/{}] Test losses: '
                                    'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['loss'],
                                                        test_metrics['range_angle']['loss']))
                                print('[Epoch {}/{}] Test Prec: '
                                    'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['prec'],
                                                        test_metrics['range_angle']['prec']))
                                print('[Epoch {}/{}] Test mIoU: '
                                    'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['miou'],
                                                        test_metrics['range_angle']['miou']))
                                print('[Epoch {}/{}] Test Dice: '
                                    'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['dice'],
                                                        test_metrics['range_angle']['dice']))

                                self.results['rd_train_loss'] = rd_train_loss.item()
                                self.results['ra_train_loss'] = ra_train_loss.item()

                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics
                            if nb_losses > 3:
                                self.results['coherence_train_loss'] = coherence_train_loss.item()
                            state = {
                                'epoch': epoch + 1,
                                'arch': str(self.net),
                                'net': self.net.state_dict(),
                                'best_dice': best_test_dice,
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': scheduler.state_dict(),
                            }
                            ema_state = self._ema_state_for_save()
                            if ema_state is not None:
                                state['ema_state'] = ema_state
                            self._save_results(state)
                            patience_counter = 0
                        else:
                            patience_counter += 1
                            if patience_counter >= self.erstp:
                                print(f"早停触发：第{epoch + 1}个epoch，验证性能连续{self.erstp}个epoch未提升")
                                sys.exit(0)
                        self._restore_from_backup(ema_backup)
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
                    # del loss, track_loss # 删除当前batch的中间变量
                    # torch.cuda.empty_cache()  # 清理未使用的显存缓存
        self.writer.close()

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.uniform_(m.weight, 0., 1.)
                nn.init.constant_(m.bias, 0.)

    def _save_results(self, state):
        results_path = self.paths['results'] / 'results.json'
        model_path = self.paths['results'] / 'model.pt'
        with open(results_path, "w") as fp:
            json.dump(self.results, fp)
        torch.save(state, model_path)


    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)

