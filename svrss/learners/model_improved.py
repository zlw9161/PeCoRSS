"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, ConcatDataset

from svrss.loaders.dataloaders import PeCoRSVDataset
from svrss.learners.tester import Tester
from svrss.utils.functions import normalize, define_loss, get_transformations
from svrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
from svrss.utils.distributed_utils import get_rank, reduce_value
import random
import time
# from tqdm import tqdm


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.model_name = self.cfg['model']
        self.dataset = self.cfg['dataset']
        self.process_signal = self.cfg['process_signal']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.schedular_type = self.cfg['schedular']
        self.T_max = self.cfg['Tmax']
        self.loss_step = self.cfg['loss_step']
        self.val_step = self.cfg['val_step']
        self.viz_step = self.cfg['viz_step']
        self.torch_seed = self.cfg['torch_seed']
        self.numpy_seed = self.cfg['numpy_seed']
        self.nb_classes = self.cfg['nb_classes']
        self.custom_loss = self.cfg['custom_loss']
        self.focal_gamma = self.cfg.get('focal_gamma', None)
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.device = self.cfg['device']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.cmplx_flag = self.cfg['cmplx_flag']
        self.pin_mem = bool(self.cfg.get('pin_mem', False))
        self.reck = self.cfg.get('ReCK_MODE', 'none')
        self.use_global_dataset = bool(self.cfg.get('use_global_dataset', False))
        self.rank = get_rank()
        self.lmf_v6_curriculum_enable = self.cfg.get('lmf_v6_curriculum_enable', False)
        self.lmf_v6_freeze_ratio = float(self.cfg.get('lmf_v6_freeze_ratio', 0.1))
        self.lmf_v6_transition_ratio = float(self.cfg.get('lmf_v6_transition_ratio', 0.4))
        self.lmf_v6_prior_reg_start = float(self.cfg.get('lmf_v6_prior_reg_start', 1e-2))
        self.lmf_v6_prior_reg_end = float(self.cfg.get('lmf_v6_prior_reg_end', 1e-4))
        self.lmf_v6_prior_lr_scale = float(self.cfg.get('lmf_v6_prior_lr_scale', 0.1))
        self.lmf_v6_residual_lr_scale_start = float(self.cfg.get('lmf_v6_residual_lr_scale_start', 0.3))
        self.lmf_v6_residual_lr_scale_end = float(self.cfg.get('lmf_v6_residual_lr_scale_end', 1.0))
        self.lmf_v6_residual_alpha_start = float(self.cfg.get('lmf_v6_residual_alpha_start', 0.1))
        self.lmf_v6_residual_alpha_end = float(self.cfg.get('lmf_v6_residual_alpha_end', 1.0))
        self._lmf_prior_params = []
        self._lmf_residual_params = []
        self._lmf_prior_modules = []
        self._lmf_residual_modules = []
        self._lmf_prior_reg_lambda = 0.0
        if self.lmf_v6_curriculum_enable:
            self._setup_lmf_v6_curriculum()
        if self.rank == 0:
            self.writer = SummaryWriter(self.paths['writer'])
            self.visualizer = TensorboardMultiLossVisualizer(self.writer)
            self.tester = Tester(self.cfg, self.visualizer)
        else:
            self.tester = Tester(self.cfg)
        self.results = dict()
        self.start_epoch = 0
        self.start_iteration = 0
        self.best_test_dop = 0.0

    def train(self, add_temp=False, checkpoint_path=None, use_global_dataset=None):
        """
        Method to train a network

        PARAMETERS
        ----------
        add_temp: boolean
            Add a temporal dimension during training?
            Considering the input as a sequence.
            Default: False
        """
        if use_global_dataset is None:
            use_global_dataset = self.use_global_dataset
        if checkpoint_path is not None or use_global_dataset:
            return self._train_with_global_dataset(
                add_temp=add_temp,
                checkpoint_path=checkpoint_path,
                use_global_dataset=use_global_dataset,
            )

        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        # self.net.module.apply(self._init_weights)
        if self.dataset in ['CWRdata', 'PDRdata', 'Carrada_RD', 'CarradaRAC_RD', 'XPDdata', 'RDSyn', 'NuaaSim', 'LSFJamming']:
            signal_type = 'range_doppler'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device, self.focal_gamma)
        elif self.dataset in ['Carrada_RA', 'CarradaRAC_RA']:
            signal_type = 'range_angle'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device, self.focal_gamma)
        else:
            raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
        nb_losses = len(criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        if self.lmf_v6_curriculum_enable and (self._lmf_prior_params or self._lmf_residual_params):
            optimizer = self._build_lmf_v6_optimizer()
        else:
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
        
        iteration = 0
        flag_save = False
        name_result = []
        best_test_dop = 0
        model_vis_flag = True

        for epoch in range(self.nb_epochs):
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            if self.lmf_v6_curriculum_enable and (self._lmf_prior_params or self._lmf_residual_params):
                self._apply_lmf_v6_curriculum(epoch, optimizer)

            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                if self.dataset == 'CWRdata':
                    path_to_frames = self.paths['CWRdata'] / seq_name[0]
                elif self.dataset == 'PDRdata':
                    path_to_frames = self.paths['PDRdata'] / seq_name[0]
                elif self.dataset == 'XPDdata':
                    path_to_frames = self.paths['XPDdata'] / seq_name[0]
                elif self.dataset == 'RDSyn':
                    path_to_frames = self.paths['RDSyn'] / seq_name[0]
                elif self.dataset == 'NuaaSim':
                    path_to_frames = self.paths['NuaaSim'] / seq_name[0]
                elif self.dataset == 'LSFJamming':
                    path_to_frames = self.paths['LSFJamming'] / seq_name[0]
                elif self.dataset in ['Carrada_RA', 'Carrada_RD', 'CarradaRAC_RA', 'CarradaRAC_RD']:
                    path_to_frames = self.paths['Carrada'] / seq_name[0]
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
                
                _dataset = PeCoRSVDataset(seq,
                                          self.dataset,
                                          self.annot_type,
                                          signal_type,
                                          path_to_frames,
                                          self.process_signal,
                                          self.n_frames,
                                          self.cmplx_flag,
                                          transformations,
                                          add_temp)
                
                if self.distributed:
                    sampler_train = torch.utils.data.distributed.DistributedSampler(_dataset)
                    sampler_train.set_epoch(epoch)
                else:
                    sampler_train = torch.utils.data.RandomSampler(_dataset)
                train_batch_sampler = torch.utils.data.BatchSampler(sampler_train, self.batch_size, drop_last=True)
                frame_dataloader = DataLoader(_dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=self.num_workers)
                for _, frame in enumerate(frame_dataloader):
                    if self.cmplx_flag:
                        rd_data = frame['rd_matrix'].to(self.device)
                    else:
                        rd_data = frame['rd_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, self.dataset, signal_type, norm_type=self.norm_type)
                    optimizer.zero_grad()
                    if model_vis_flag and self.rank == 0 and not self.distributed:
                        vis_net = self.net.module if self.distributed or isinstance(self.net, nn.parallel.DataParallel) else self.net
                        self.visualizer.model_visualization(vis_net, rd_data)
                        model_vis_flag = False
                    elif model_vis_flag and self.rank == 0:
                        model_vis_flag = False
                    rd_outputs = self.net(rd_data) # svrss:rd_data=[6,1,5,124,2048]
                    rd_outputs = rd_outputs.to(self.device)

                    rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                for c in criterion]

                    rd_loss = torch.mean(torch.stack(rd_losses))
                    loss = rd_loss
                    if self.lmf_v6_curriculum_enable and self._lmf_prior_modules:
                        prior_penalty = self._compute_lmf_prior_penalty()
                        if prior_penalty is not None and self._lmf_prior_reg_lambda > 0:
                            loss = loss + self._lmf_prior_reg_lambda * prior_penalty

                    rd_losses_reduced = reduce_value(rd_losses)
                    rd_loss_reduced = reduce_value(rd_loss)
                    loss_reduced = reduce_value(loss)

                    loss.backward()
                    optimizer.step()
                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss_reduced.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses_reduced[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses_reduced[1].data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            if signal_type == 'range_doppler':
                                print('[{}][Epoch {}/{}, iter {}]: '
                                    'Train loss {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                            epoch+1,
                                                            self.nb_epochs,
                                                            iteration,
                                                            train_loss))
                                print('[Epoch {}/{}, iter {}]: '
                                    'Train losses: RD={}'.format(epoch+1,
                                                                        self.nb_epochs,
                                                                        iteration,
                                                                        rd_train_loss))
                            else:
                                print('[{}][Epoch {}/{}, iter {}]: '
                                    'Train loss {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                                            epoch+1,
                                                            self.nb_epochs,
                                                            iteration,
                                                            train_loss))
                                print('[Epoch {}/{}, iter {}]: '
                                    'Train losses: RA={}'.format(epoch+1,
                                                                        self.nb_epochs,
                                                                        iteration,
                                                                        rd_train_loss))
                            
                    
                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                        rd_train_losses, iteration)
                            running_losses = list()
                            rd_running_losses = list()
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        if self.rank == 0:
                            self.visualizer.update_multi_val_metrics(signal_type, val_metrics, iteration)
                            if signal_type == 'range_doppler':
                                print('[Epoch {}/{}] Val losses: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['loss'],
                                                            ))
                                
                                print('[Epoch {}/{}] Val Pixel Prec: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['prec'],
                                                            ))
                                print('[Epoch {}/{}] Val mIoU: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['miou'],
                                                            ))
                                print('[Epoch {}/{}] Val Dice: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['dice'],
                                                            ))
                            else:
                                print('[Epoch {}/{}] Val losses: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['loss'],
                                                            ))
                                
                                print('[Epoch {}/{}] Val Pixel Prec: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['prec'],
                                                            ))
                                print('[Epoch {}/{}] Val mIoU: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['miou'],
                                                            ))
                                print('[Epoch {}/{}] Val Dice: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['dice'],
                                                            ))

                            print('[Epoch {}/{}] Val Recall by class: '
                                  '{}'.format(epoch+1,
                                              self.nb_epochs,
                                              val_metrics[signal_type]['recall_by_class']))
                            print('[Epoch {}/{}] Val Confusion matrix:\n{}'.format(
                                epoch + 1,
                                self.nb_epochs,
                                self._format_confusion_matrix(val_metrics[signal_type]['confusion_matrix'])
                            ))

                        test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp)
                        if self.rank == 0:
                            self.visualizer.update_multi_test_metrics(signal_type, test_metrics, iteration)
                            if signal_type == 'range_doppler':
                                print('[Epoch {}/{}] Test losses: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['loss']))
                                print('[Epoch {}/{}] Test Prec: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['prec'],
                                                            ))
                                print('[Epoch {}/{}] Test mIoU: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['miou'],
                                                            ))
                                print('[Epoch {}/{}] Test Dice: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['dice'],
                                                            ))
                            else:
                                print('[Epoch {}/{}] Test losses: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['loss']))
                                print('[Epoch {}/{}] Test Prec: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['prec'],
                                                            ))
                                print('[Epoch {}/{}] Test mIoU: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['miou'],
                                                            ))
                                print('[Epoch {}/{}] Test Dice: '
                                    'RA={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            test_metrics[signal_type]['dice'],
                                                            ))

                            print('[Epoch {}/{}] Test Recall by class: '
                                  '{}'.format(epoch+1,
                                              self.nb_epochs,
                                              test_metrics[signal_type]['recall_by_class']))
                            print('[Epoch {}/{}] Test Confusion matrix:\n{}'.format(
                                epoch + 1,
                                self.nb_epochs,
                                self._format_confusion_matrix(test_metrics[signal_type]['confusion_matrix'])
                            ))

                            self.results['epoch'] = epoch + 1
                            self.results['rd_train_loss'] = rd_train_loss.item()
                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics
                    
                            
                            if test_metrics[signal_type]['dice'] > best_test_dop and iteration > 0:
                                best_test_dop = test_metrics[signal_type]['dice']
                                flag_save = True
                                name_result.append(self.model_name)
                            
                            if flag_save:
                                self._save_results(name_result)
                                flag_save = False
                                name_result = []
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
        if self.rank == 0:
            self.writer.close()

    def _train_with_global_dataset(self, add_temp=False, checkpoint_path=None, use_global_dataset=True):
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)

        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names, sizes=(self.w_size, self.h_size))
        self._set_seeds()

        if self.dataset in ['CWRdata', 'PDRdata', 'Carrada_RD', 'CarradaRAC_RD', 'XPDdata', 'RDSyn', 'NuaaSim', 'LSFJamming']:
            signal_type = 'range_doppler'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device, self.focal_gamma)
        elif self.dataset in ['Carrada_RA', 'CarradaRAC_RA']:
            signal_type = 'range_angle'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device, self.focal_gamma)
        else:
            raise KeyError(f'Dataset {self.dataset} has not been supported yet.')

        running_losses = []
        rd_running_losses = []
        rd_running_global_losses = [[], []]

        if self.lmf_v6_curriculum_enable and (self._lmf_prior_params or self._lmf_residual_params):
            optimizer = self._build_lmf_v6_optimizer()
        else:
            optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)

        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max - 1, eta_min=1e-7)
        else:
            raise KeyError('we only implement schedular of exp and cos')

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            loaded = self._load_checkpoint(checkpoint_path)
            self.start_epoch, self.start_iteration, self.best_test_dop, opt_state, sch_state = loaded
            if opt_state is not None:
                optimizer.load_state_dict(opt_state)
            if sch_state is not None:
                scheduler.load_state_dict(sch_state)
            model_vis_flag = True
        else:
            self.start_epoch = 0
            self.start_iteration = 0
            self.best_test_dop = 0.0
            model_vis_flag = True

        iteration = self.start_iteration
        flag_save = False
        name_result = []

        combined_dataset = None
        if use_global_dataset:
            combined_dataset = self._build_combined_dataset(train_loader, transformations, signal_type, add_temp)

        for epoch in range(self.start_epoch, self.nb_epochs):
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()

            if self.lmf_v6_curriculum_enable and (self._lmf_prior_params or self._lmf_residual_params):
                self._apply_lmf_v6_curriculum(epoch, optimizer)

            if use_global_dataset:
                frame_loaders = [self._build_global_frame_loader(combined_dataset, epoch)]
            else:
                frame_loaders = self._build_sequence_frame_loaders(train_loader, transformations, signal_type, add_temp, epoch)

            for frame_loader in frame_loaders:
                for _, frame in enumerate(frame_loader):
                    if self.cmplx_flag:
                        rd_data = frame['rd_matrix'].to(self.device)
                    else:
                        rd_data = frame['rd_matrix'].to(self.device).float()

                    rd_mask = frame['rd_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, self.dataset, signal_type, norm_type=self.norm_type)

                    optimizer.zero_grad()
                    if model_vis_flag and self.rank == 0:
                        vis_net = self.net.module if self.distributed or isinstance(self.net, nn.parallel.DataParallel) else self.net
                        self.visualizer.model_visualization(vis_net, rd_data)
                        model_vis_flag = False

                    rd_outputs = self.net(rd_data).to(self.device)
                    rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1)) for c in criterion]
                    rd_loss = torch.mean(torch.stack(rd_losses))
                    loss = rd_loss

                    if self.lmf_v6_curriculum_enable and self._lmf_prior_modules:
                        prior_penalty = self._compute_lmf_prior_penalty()
                        if prior_penalty is not None and self._lmf_prior_reg_lambda > 0:
                            loss = loss + self._lmf_prior_reg_lambda * prior_penalty

                    rd_losses_reduced = reduce_value(rd_losses)
                    rd_loss_reduced = reduce_value(rd_loss)
                    loss_reduced = reduce_value(loss)

                    loss.backward()
                    optimizer.step()

                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss_reduced.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses_reduced[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses_reduced[1].data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            print('[{}][Epoch {}/{}, iter {}]: Train loss {}'.format(
                                time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
                                epoch + 1,
                                self.nb_epochs,
                                iteration,
                                train_loss,
                            ))
                            print('[Epoch {}/{}, iter {}]: Train losses: {}={}'.format(
                                epoch + 1,
                                self.nb_epochs,
                                iteration,
                                'RD' if signal_type == 'range_doppler' else 'RA',
                                rd_train_loss,
                            ))
                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss, rd_train_losses, iteration)
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)
                            running_losses = []
                            rd_running_losses = []

                    if iteration % self.val_step == 0 and iteration > 0:
                        if self.reck == 'ReCK-1' and hasattr(self.net, 'remove_hooks'):
                            self.net.remove_hooks()

                        if iteration % self.viz_step == 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration, add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        test_metrics = self.tester.predict(self.net, test_loader, add_temp=add_temp)

                        if self.rank == 0:
                            self.visualizer.update_multi_val_metrics(signal_type, val_metrics, iteration)
                            self.visualizer.update_multi_test_metrics(signal_type, test_metrics, iteration)
                            print('[Epoch {}/{}] Val Dice: {}={}'.format(epoch + 1, self.nb_epochs,
                                                                         'RD' if signal_type == 'range_doppler' else 'RA',
                                                                         val_metrics[signal_type]['dice']))
                            print('[Epoch {}/{}] Test Dice: {}={}'.format(epoch + 1, self.nb_epochs,
                                                                          'RD' if signal_type == 'range_doppler' else 'RA',
                                                                          test_metrics[signal_type]['dice']))
                            self.results['epoch'] = epoch + 1
                            self.results['rd_train_loss'] = float(rd_train_loss) if 'rd_train_loss' in locals() else None
                            self.results['train_loss'] = float(train_loss) if 'train_loss' in locals() else None
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics

                            if test_metrics[signal_type]['dice'] > self.best_test_dop:
                                self.best_test_dop = test_metrics[signal_type]['dice']
                                flag_save = True
                                name_result.append(self.model_name)

                            if flag_save:
                                self._save_results(
                                    name_result,
                                    save_full_checkpoint=True,
                                    current_epoch=epoch,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    current_iteration=iteration,
                                    best_dice=self.best_test_dop,
                                )
                                flag_save = False
                                name_result = []

                        self.net.train()
                        if self.reck == 'ReCK-1' and hasattr(self.net, 'register_hooks'):
                            self.net.register_hooks()

                    iteration += 1

            if epoch % 10 == 0 and self.rank == 0:
                self._save_results(
                    [f'{self.model_name}_epoch_{epoch:04d}'],
                    save_full_checkpoint=True,
                    current_epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    current_iteration=iteration,
                    best_dice=self.best_test_dop,
                )

        if self.rank == 0:
            self.writer.close()
            if self.reck == 'ReCK-1' and hasattr(self.net, 'remove_hooks'):
                self.net.remove_hooks()

    def _resolve_path_to_frames(self, seq_name):
        if self.dataset == 'CWRdata':
            return self.paths['CWRdata'] / seq_name
        if self.dataset == 'PDRdata':
            return self.paths['PDRdata'] / seq_name
        if self.dataset == 'XPDdata':
            return self.paths['XPDdata'] / seq_name
        if self.dataset == 'RDSyn':
            return self.paths['RDSyn'] / seq_name
        if self.dataset == 'NuaaSim':
            return self.paths['NuaaSim'] / seq_name
        if self.dataset == 'LSFJamming':
            return self.paths['LSFJamming'] / seq_name
        if self.dataset in ['Carrada_RA', 'Carrada_RD', 'CarradaRAC_RA', 'CarradaRAC_RD']:
            return self.paths['Carrada'] / seq_name
        raise KeyError(f'Dataset {self.dataset} has not been supported yet.')

    def _build_combined_dataset(self, train_loader, transformations, signal_type, add_temp):
        datasets = []
        for seq_name, seq in train_loader:
            path_to_frames = self._resolve_path_to_frames(seq_name[0])
            datasets.append(
                PeCoRSVDataset(
                    seq,
                    self.dataset,
                    self.annot_type,
                    signal_type,
                    path_to_frames,
                    self.process_signal,
                    self.n_frames,
                    self.cmplx_flag,
                    transformations,
                    add_temp,
                )
            )
        return ConcatDataset(datasets)

    def _build_global_frame_loader(self, combined_dataset, epoch):
        if self.distributed:
            sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset)
            sampler.set_epoch(epoch)
        else:
            sampler = torch.utils.data.RandomSampler(combined_dataset)

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_mem,
            persistent_workers=self.num_workers > 0,
            drop_last=True,
        )

    def _build_sequence_frame_loaders(self, train_loader, transformations, signal_type, add_temp, epoch):
        loaders = []
        for seq_name, seq in train_loader:
            path_to_frames = self._resolve_path_to_frames(seq_name[0])
            dataset = PeCoRSVDataset(
                seq,
                self.dataset,
                self.annot_type,
                signal_type,
                path_to_frames,
                self.process_signal,
                self.n_frames,
                self.cmplx_flag,
                transformations,
                add_temp,
            )

            if self.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(dataset)
                sampler.set_epoch(epoch)
            else:
                sampler = torch.utils.data.RandomSampler(dataset)

            batch_sampler = torch.utils.data.BatchSampler(sampler, self.batch_size, drop_last=True)
            loaders.append(
                DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_mem,
                    persistent_workers=self.num_workers > 0,
                )
            )
        return loaders

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

    def _format_confusion_matrix(self, matrix, precision=4):
        cm = np.asarray(matrix)
        if cm.ndim != 2:
            return str(cm)

        n_rows, n_cols = cm.shape
        if np.issubdtype(cm.dtype, np.integer) or np.allclose(cm, np.rint(cm), atol=1e-6):
            formatter = lambda v: str(int(round(float(v))))
        else:
            formatter = lambda v: f'{float(v):.{precision}f}'

        str_cells = [[formatter(cm[i, j]) for j in range(n_cols)] for i in range(n_rows)]
        col_headers = [f'c{j}' for j in range(n_cols)]
        col_widths = []
        for j in range(n_cols):
            max_cell = max(len(str_cells[i][j]) for i in range(n_rows))
            col_widths.append(max(len(col_headers[j]), max_cell))

        row_label_width = max(len(f'r{n_rows - 1}'), 2)
        sep = ' | '
        header = ' ' * (row_label_width + len(sep)) + sep.join(
            f'{col_headers[j]:>{col_widths[j]}}' for j in range(n_cols)
        )
        divider = '-' * len(header)

        lines = [header, divider]
        for i in range(n_rows):
            row_label = f'r{i:>{row_label_width - 1}}'
            row = sep.join(f'{str_cells[i][j]:>{col_widths[j]}}' for j in range(n_cols))
            lines.append(f'{row_label}{sep}{row}')

        return '\n'.join(lines)

    def _save_results(self, names, save_full_checkpoint=False, current_epoch=None,
                      optimizer=None, scheduler=None, current_iteration=None, best_dice=None):
        for name in names:
            save_dir = self.paths['results']
            save_dir.mkdir(parents=True, exist_ok=True)
            net_state = self.net.module.state_dict() if (
                self.distributed or isinstance(self.net, nn.parallel.DataParallel)
            ) else self.net.state_dict()

            if 'epoch' not in name:
                results_path = save_dir / (name + '_' + 'results.json')
                model_path = save_dir / (name + '.pt')
                with open(results_path, 'w') as fp:
                    json.dump(self.results, fp)
                torch.save(net_state, model_path)
                if save_full_checkpoint:
                    ckpt_path = save_dir / f'{name}_checkpoint_epoch{current_epoch}.pt'
                    checkpoint = {
                        'epoch': current_epoch,
                        'iteration': current_iteration,
                        'net_state_dict': net_state,
                        'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                        'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                        'best_dice': best_dice,
                        'results': self.results,
                        'cfg': self.cfg,
                    }
                    torch.save(checkpoint, ckpt_path)
            elif save_full_checkpoint:
                ckpt_path = save_dir / f'{name}_checkpoint_epoch{current_epoch}.pt'
                checkpoint = {
                    'epoch': current_epoch,
                    'iteration': current_iteration,
                    'net_state_dict': net_state,
                    'optimizer_state_dict': optimizer.state_dict() if optimizer is not None else None,
                    'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
                    'best_dice': best_dice,
                    'results': self.results,
                    'cfg': self.cfg,
                }
                torch.save(checkpoint, ckpt_path)

    def _load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            if self.rank == 0:
                print(f'Checkpoint does not exist: {checkpoint_path}, starting from scratch.')
            return 0, 0, 0.0, None, None

        if self.rank == 0:
            print(f'Resuming training from checkpoint: {checkpoint_path}')

        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        except Exception as exc:
            if self.rank == 0:
                print(f'Failed to load checkpoint: {exc}, starting from scratch.')
            return 0, 0, 0.0, None, None

        required_keys = ['net_state_dict', 'epoch', 'iteration', 'best_dice']
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            if self.rank == 0:
                print(f'Invalid checkpoint format, missing keys: {missing_keys}, starting from scratch.')
            return 0, 0, 0.0, None, None

        net_state_dict = checkpoint['net_state_dict']
        if self.distributed or isinstance(self.net, nn.parallel.DataParallel):
            if not list(net_state_dict.keys())[0].startswith('module.'):
                net_state_dict = {f'module.{k}': v for k, v in net_state_dict.items()}
        else:
            net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

        try:
            self.net.load_state_dict(net_state_dict, strict=True)
        except Exception as exc:
            if self.rank == 0:
                print(f'Failed to load model weights: {exc}, starting from scratch.')
            return 0, 0, 0.0, None, None

        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration']
        best_dice = checkpoint['best_dice']
        self.results = checkpoint.get('results', {})
        opt_state = checkpoint.get('optimizer_state_dict')
        sch_state = checkpoint.get('scheduler_state_dict')
        if self.rank == 0:
            print('Checkpoint loaded successfully:')
            print(f'   - Resumed starting epoch: {start_epoch}')
            print(f'   - Resumed global iteration: {start_iteration}')
            print(f'   - Restored best Test Dice: {best_dice:.4f}')
        return start_epoch, start_iteration, best_dice, opt_state, sch_state

    def _set_seeds(self):
        # each processor has its own seed
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
        random.seed(self.numpy_seed)

    def _get_model_core(self):
        return self.net.module if hasattr(self.net, 'module') else self.net

    def _setup_lmf_v6_curriculum(self):
        model_core = self._get_model_core()
        for module in model_core.modules():
            if module.__class__.__name__ not in (
                'Learnable_Match_FilterV6',
                'Learnable_Match_FilterV7',
                'LMF_Layer_Parallel',
            ):
                continue
            if getattr(module, 'residual_mode', False):
                self._lmf_residual_modules.append(module)
                self._lmf_residual_params.extend([p for p in module.parameters() if p.requires_grad])
            else:
                self._lmf_prior_modules.append(module)
                self._lmf_prior_params.extend([p for p in module.parameters() if p.requires_grad])

    def _build_lmf_v6_optimizer(self):
        prior_ids = {id(p) for p in self._lmf_prior_params}
        residual_ids = {id(p) for p in self._lmf_residual_params}
        model_core = self._get_model_core()
        base_params = [p for p in model_core.parameters()
                       if p.requires_grad and id(p) not in prior_ids and id(p) not in residual_ids]
        param_groups = []
        if base_params:
            param_groups.append({'params': base_params, 'lr': self.lr, 'group_name': 'base'})
        if self._lmf_prior_params:
            param_groups.append({'params': self._lmf_prior_params,
                                 'lr': self.lr * self.lmf_v6_prior_lr_scale,
                                 'group_name': 'lmf_prior'})
        if self._lmf_residual_params:
            param_groups.append({'params': self._lmf_residual_params,
                                 'lr': self.lr * self.lmf_v6_residual_lr_scale_start,
                                 'group_name': 'lmf_residual'})
        return optim.Adam(param_groups, lr=self.lr)

    def _apply_lmf_v6_curriculum(self, epoch, optimizer):
        if self.nb_epochs <= 1:
            progress = 1.0
        else:
            progress = epoch / float(self.nb_epochs - 1)

        freeze_end = max(0.0, min(self.lmf_v6_freeze_ratio, 1.0))
        transition_end = max(freeze_end, min(self.lmf_v6_transition_ratio, 1.0))

        if progress < freeze_end:
            stage = 0
            stage_t = 0.0
        elif progress < transition_end:
            stage = 1
            stage_t = (progress - freeze_end) / max(transition_end - freeze_end, 1e-8)
        else:
            stage = 2
            stage_t = (progress - transition_end) / max(1.0 - transition_end, 1e-8)

        freeze_prior = stage == 0

        if stage == 0:
            prior_lambda = self.lmf_v6_prior_reg_start
            residual_alpha = self.lmf_v6_residual_alpha_start
            residual_lr_scale = self.lmf_v6_residual_lr_scale_start
        elif stage == 1:
            prior_lambda = self.lmf_v6_prior_reg_start + stage_t * (self.lmf_v6_prior_reg_end - self.lmf_v6_prior_reg_start)
            residual_alpha = self.lmf_v6_residual_alpha_start + stage_t * (0.5 - self.lmf_v6_residual_alpha_start)
            residual_lr_scale = self.lmf_v6_residual_lr_scale_start + stage_t * (0.7 - self.lmf_v6_residual_lr_scale_start)
        else:
            prior_lambda = self.lmf_v6_prior_reg_end * (1.0 - stage_t)
            residual_alpha = 0.5 + stage_t * (self.lmf_v6_residual_alpha_end - 0.5)
            residual_lr_scale = 0.7 + stage_t * (self.lmf_v6_residual_lr_scale_end - 0.7)

        self._lmf_prior_reg_lambda = max(prior_lambda, 0.0)
        for module in self._lmf_residual_modules:
            module.residual_scale = float(max(residual_alpha, 0.0))

        base_lr = None
        for group in optimizer.param_groups:
            if group.get('group_name') == 'base':
                base_lr = group['lr']
                break
        if base_lr is None and optimizer.param_groups:
            base_lr = optimizer.param_groups[0]['lr']
        if base_lr is None:
            return

        prior_lr_scale = 0.0 if freeze_prior else self.lmf_v6_prior_lr_scale

        for group in optimizer.param_groups:
            gname = group.get('group_name')
            if gname == 'lmf_prior':
                group['lr'] = base_lr * prior_lr_scale
            elif gname == 'lmf_residual':
                group['lr'] = base_lr * residual_lr_scale

    def _compute_lmf_prior_penalty(self):
        if not self._lmf_prior_modules:
            return None
        penalty = None
        for module in self._lmf_prior_modules:
            if module.prior_kernel_real is None or module.prior_kernel_imag is None:
                continue
            loss_real = torch.mean((module.kernel_real - module.prior_kernel_real) ** 2)
            loss_imag = torch.mean((module.kernel_imag - module.prior_kernel_imag) ** 2)
            item = loss_real + loss_imag
            penalty = item if penalty is None else (penalty + item)
        return penalty
