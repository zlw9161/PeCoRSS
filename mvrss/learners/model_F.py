"""Class to train a PyTorch model"""
import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from mvrss.loaders.dataloaders_F import CarradaDataset
from mvrss.learners.tester_F import Tester
from mvrss.utils.functions import normalize, define_loss, get_transformations, soft_mask_pdf, soft_mask_gaus
from mvrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
from mvrss.utils.distributed_utils import get_rank, reduce_value
from mvrss.utils import MVRSS_HOME
import random
import time

#from thop import profile
#from thop import clever_format
import time


class Model(nn.Module):
    """Class to train a model

    PARAMETERS
    ----------
    net: PyTorch Model
        Network to train
    data: dict
        Parameters and configurations for training
    """

    def __init__(self, net, data, trackloss):
        super().__init__()
        self.net = net
        self.cfg = data['cfg']
        self.paths = data['paths']
        self.dataloaders = data['dataloaders']
        self.dataset_type = self.cfg['dataset']
        self.model_name = self.cfg['model']
        self.resume_path = self.cfg['resume']
        self.process_signal = self.cfg['process_signal']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
        self.lr = self.cfg['lr']
        self.lr_step = self.cfg['lr_step']
        self.schedular_type = self.cfg['schedular']
        self.T_max = self.cfg['T_max']
        self.warmup = self.cfg['warmup']
        self.warmup_factor = self.cfg['warmup_factor']
        self.warmup_iters = self.cfg['warmup_iters']
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
        self.device = self.cfg['device']
        self.trackloss = trackloss
        if self.trackloss:
            self.soft_trackloss = self.cfg['soft_trackloss']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.rank = get_rank()
        if self.rank == 0:
            self.writer = SummaryWriter(self.paths['writer'])
            self.visualizer = TensorboardMultiLossVisualizer(self.writer)
            self.tester = Tester(self.cfg, self.trackloss, self.visualizer)
        else:
            self.tester = Tester(self.cfg, self.trackloss)
        self.results = dict()

    def _get_class_weights(self, signal_type):
        """Load class weights for custom loss

        PARAMETERS
        ----------
        signal_type: str
            Supported: 'range_doppler', 'range_angle'

        RETURNS
        -------
        weights: numpy array
        """
        weight_path = MVRSS_HOME / 'config_files'
        if signal_type in ('range_angle'):
            file_name = 'ra_weights.json'
        elif signal_type in ('range_doppler'):
            file_name = 'rd_weights.json'
        else:
            raise ValueError('Signal type {} is not supported.'.format(signal_type))
        file_path = weight_path / file_name
        with open(file_path, 'r') as fp:
            weights = json.load(fp)
        weights = np.array([weights['background'], weights['pedestrian'],
                            weights['cyclist'], weights['car']])
        weights = torch.from_numpy(weights)
        return weights

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
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        self.net.apply(self._init_weights)
        rd_criterion = define_loss('range_doppler', self.custom_loss, self.device, view_weight=20.)
        ra_criterion = define_loss('range_angle', self.custom_loss, self.device, view_weight=15.)
        if self.trackloss:
            weights_rd = self._get_class_weights('range_doppler')
            weights_ra = self._get_class_weights('range_angle')
            ce_loss_rd = nn.CrossEntropyLoss(weight=weights_rd.to(self.device).float())
            ce_loss_ra = nn.CrossEntropyLoss(weight=weights_ra.to(self.device).float())
            rd_criterion.append(ce_loss_rd)
            ra_criterion.append(ce_loss_ra)
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
        model_vis_flag = True
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)

        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max - 1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
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

        # lt @20230416, warmup strategy
        if self.warmup:
            warm_scheduler = warmup_lr_scheduler(optimizer, self.warmup_iters, self.warmup_factor)

        for epoch in range(start_epoch, self.nb_epochs):
            if self.warmup and iteration < self.warmup_iters:
                warm_scheduler.step()
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['Carrada'], seq_name[0])
                carrada_dataset =CarradaDataset(seq, self.dataset_type, self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp,
                                                             self.trackloss)
                if self.distributed:
                    sampler_train = torch.utils.data.distributed.DistributedSampler(carrada_dataset)
                    sampler_train.set_epoch(epoch)
                else:
                    sampler_train = torch.utils.data.RandomSampler(carrada_dataset)
                train_batch_sampler = torch.utils.data.BatchSampler(sampler_train, self.batch_size, drop_last=True)
                frame_dataloader = DataLoader(carrada_dataset,
                                              batch_sampler=train_batch_sampler,
                                              num_workers=self.num_workers)
                total_time = 0
                count = 0
                for _, frame in enumerate(frame_dataloader):
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
                    if model_vis_flag and self.rank == 0:
                        vis_net = self.net.module if self.distributed or isinstance(self.net, nn.parallel.DataParallel) else self.net
                        self.visualizer.model_visualization(vis_net, rd_data)
                        model_vis_flag = False
                    
                    if self.model_name == 'tfoc':
                        if self.model_name != 'mvnet':
                            rd_outputs, ra_outputs, rd_probs, ra_probs = self.net(rd_data, ra_data, ad_data)
                        else:
                            rd_outputs, ra_outputs, rd_probs, ra_probs = self.net(rd_data, ra_data)
                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)
                        # trackloss is the father point of soft_tarckloss, which are specific to t-foc net.
                        if self.trackloss:
                            rd_probs = rd_probs.to(self.device)
                            ra_probs = ra_probs.to(self.device)
                            x_mask_rd = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask_rd[x_mask_rd > 1] = 1  # (b,4,h,w)
                            x_mask_ra = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                            x_mask_ra[x_mask_ra > 1] = 1  # (b,4,h,w)
                            if self.soft_trackloss:
                                x_mask_rd = soft_mask_pdf(x_mask_rd, signal_type='range_doppler')  # [b,4,h,w]
                                x_mask_ra = soft_mask_pdf(x_mask_ra, signal_type='range_angle')
                                track_loss_rd = rd_criterion[-1](rd_probs, x_mask_rd)
                                track_loss_ra = ra_criterion[-1](ra_probs, x_mask_ra)
                            else:
                                track_loss_rd = rd_criterion[-1](rd_probs, torch.argmax(x_mask_rd, axis=1))
                                track_loss_ra = ra_criterion[-1](ra_probs, torch.argmax(x_mask_ra, axis=1))
                            track_loss = torch.mean(track_loss_ra + track_loss_rd)
                            loss = track_loss[...]
                        else:
                            loss = 0
                    else:
                        if self.model_name != 'mvnet':
                            rd_outputs, ra_outputs = self.net(rd_data, ra_data, ad_data)
                        else:
                            rd_outputs, ra_outputs = self.net(rd_data, ra_data)
                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)
                        loss = 0

                    # !!default set coherence_loss in the last position of self.cfg["custom_loss"]
                    if "col" in self.cfg["custom_loss"]:
                        # Case with the CoL
                        # Select the wCE and wSDice
                        if self.trackloss:
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                     for c in rd_criterion[:-2]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, -1, :, :], axis=1))
                                     for c in ra_criterion[:-2]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            coherence_loss = rd_criterion[-2](rd_outputs, ra_outputs)
                            if self.flag_coherence_track_loss:
                                coherence_track_loss = rd_criterion[-2](rd_probs, ra_probs)
                                coherence_loss = torch.mean(coherence_loss + coherence_track_loss)
                            loss += torch.mean(rd_loss + ra_loss + coherence_loss)
                        else:
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                         for c in rd_criterion[:-1]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                         for c in ra_criterion[:-1]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            coherence_loss = rd_criterion[-1](rd_outputs, ra_outputs)
                            loss += torch.mean(rd_loss + ra_loss + coherence_loss)
                    else:
                        if self.trackloss:
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                         for c in rd_criterion[:-1]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, -1, :, :], axis=1))
                                         for c in ra_criterion[:-1]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            loss += torch.mean(rd_loss + ra_loss)
                        else:
                            # Case without the CoL
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                         for c in rd_criterion]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                         for c in ra_criterion]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            loss += torch.mean(rd_loss + ra_loss)
                    loss.backward()
                    optimizer.step()
                    if self.rank == 0:
                        running_losses.append(loss.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                        if "col" in self.cfg["custom_loss"]:
                            coherence_running_losses.append(coherence_loss.data.cpu().numpy()[()])
                        if self.trackloss:
                            track_running_losses.append(track_loss.data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
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
                            if self.trackloss:
                                track_train_loss = np.mean(track_running_losses)
                                print('[Epoch {}/{}, iter {}]: '
                                  'Train Track loss {}'.format(epoch + 1,
                                                                   self.nb_epochs,
                                                                   iteration,
                                                                   track_train_loss))
                            else:
                                track_train_loss = None
                            if "col" in self.cfg["custom_loss"]:
                                coherence_train_loss = np.mean(coherence_running_losses)
                                print('[Epoch {}/{}, iter {}]: '
                                  'Train Coherence loss {}'.format(epoch + 1,
                                                                   self.nb_epochs,
                                                                   iteration,
                                                                   coherence_train_loss))
                            else:
                                coherence_train_loss = None
                            self.visualizer.update_multi_train_loss(train_loss, rd_train_loss,
                                                                    rd_train_losses, ra_train_loss,
                                                                    ra_train_losses, iteration,
                                                                    coherence_train_loss, track_train_loss)
                            running_losses = list()
                            rd_running_losses = list()
                            ra_running_losses = list()
                            # zlw@20220107 get_lr() --> get_last_lr()
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                          add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        if self.rank == 0:
                            self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                        
                            print('[Epoch {}/{}] Val losses: '
                                'RD={}, RA={}'.format(epoch+1,
                                                self.nb_epochs,
                                                    val_metrics['range_doppler']['loss'],
                                                    val_metrics['range_angle']['loss'],))
                            if "col" in self.cfg["custom_loss"]:
                                print('[Epoch {}/{}] Val Coherence Loss: '
                                'CoL={}'.format(epoch+1,
                                            self.nb_epochs,
                                            val_metrics['coherence_loss']))
                            if self.trackloss:
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

                        '''if val_metrics['global_prec'] > best_val_prec and iteration > 0:
                            best_val_prec = val_metrics['global_prec']
                            '''
                        # saving results w/ best dice zlw@20220704
                        test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp)
                        if self.rank == 0:
                            self.visualizer.update_multi_test_metrics(test_metrics, iteration)


                            print('[Epoch {}/{}] Test losses: '
                                'RD={}, RA={}'.format(epoch+1,
                                                        self.nb_epochs,
                                                        test_metrics['range_doppler']['loss'],
                                                        test_metrics['range_angle']['loss']))
                            if "col" in self.cfg["custom_loss"]:
                                print('[Epoch {}/{}] Test Coherence Loss: '
                                'CoL={}'.format(epoch+1,
                                            self.nb_epochs,
                                            test_metrics['coherence_loss']))

                            if self.trackloss:
                                print('[Epoch {}/{}] Test Track Loss: '
                                'Track={}'.format(epoch+1,
                                            self.nb_epochs,
                                            test_metrics['track_loss']))

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
                            if "col" in self.cfg["custom_loss"]:
                                self.results['coherence_train_loss'] = coherence_train_loss.item()
                            self._save_results({
                                'epoch': epoch + 1,
                                'arch': str(self.net),
                                'net': self.net.state_dict(),
                                'best_dice': best_test_dice,
                                'optimizer': optimizer.state_dict(),
                                'lr_scheduler': scheduler.state_dict(),
                            })
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
        if self.rank == 0:
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
        # 每个进程有自己的seed
        torch.cuda.manual_seed_all(self.torch_seed+get_rank())
        torch.manual_seed(self.torch_seed+get_rank())
        np.random.seed(self.numpy_seed+get_rank())
        random.seed(self.numpy_seed+get_rank())

