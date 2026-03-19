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

from svrss.loaders.dataloaders import PeCoRSVDataset
from svrss.learners.tester import Tester
from svrss.utils.functions import normalize, define_loss, get_transformations
from svrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer
from svrss.utils.distributed_utils import get_rank, reduce_value
import random
import time
import torch.distributed as dist  



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
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.device = self.cfg['device']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.cmplx_flag = self.cfg['cmplx_flag']
        self.pin_mem = self.cfg['pin_mem']
        self.rank = get_rank()
        if self.rank == 0:
            self.writer = SummaryWriter(self.paths['writer'])
            self.visualizer = TensorboardMultiLossVisualizer(self.writer)
            self.tester = Tester(self.cfg, self.visualizer)
        else:
            self.tester = Tester(self.cfg)
        self.results = dict()

        self.reck = self.cfg['ReCK_MODE']
        

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
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        # self.net.module.apply(self._init_weights)
        if self.dataset in ['CWRdata', 'PDRdata', 'Carrada_RD', 'CarradaRAC_RD', 'XPDdata', 'RDSyn', 'NuaaSim', "LSFJamming"]:
            signal_type = 'range_doppler'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device)
        elif self.dataset in ['Carrada_RA', 'CarradaRAC_RA']:
            signal_type = 'range_angle'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device)
        else:
            raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
        # ra_criterion = define_loss('range_angle', self.custom_loss, self.device)
        nb_losses = len(criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        # ra_running_losses = list()
        # ra_running_global_losses = [list(), list()]
        # coherence_running_losses = list()

        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)

        # lt @20230417
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
        
        iteration = 0
        best_val_prec = 0
        # saving results w/ best recall zlw@20220704
        best_val_recall = 0
        # lt @20230522
        flag_save = False
        name_result = []
        best_val_dop = 0
        best_val_ang = 0
        best_val_dop_plus_ang = 0
        best_test_dop = 0
        best_test_ang = 0
        best_test_dop_plus_ang = 0
        model_vis_flag = True
        

        for epoch in range(self.nb_epochs):
            # lt @20230417
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            # for _, sequence_data in enumerate(tqdm(train_loader)):
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
                                              num_workers=self.num_workers,
                                              pin_memory=self.pin_mem)
                for _, frame in enumerate(frame_dataloader):
                    if self.cmplx_flag:
                        rd_data = frame['rd_matrix'].to(self.device)
                    else:
                        rd_data = frame['rd_matrix'].to(self.device).float()
                    # ra_data = frame['ra_matrix'].to(self.device).float()
                    # ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    # ra_mask = frame['ra_mask'].to(self.device).float()

                    rd_data = normalize(rd_data, self.dataset, signal_type, norm_type=self.norm_type)
                    # ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    # zlw@20220325
                    # if self.model_name != 'mvnet':
                    #     ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                    optimizer.zero_grad()
                    if model_vis_flag and self.rank == 0:
                        vis_net = self.net.module if self.distributed or isinstance(self.net, nn.parallel.DataParallel) else self.net
                        self.visualizer.model_visualization(vis_net, rd_data)
                        model_vis_flag = False
                    rd_outputs = self.net(rd_data) # svrss:rd_data=[6,1,5,124,2048]
                    rd_outputs = rd_outputs.to(self.device)
                    #print('rd_outputs.shape', rd_outputs.shape)
                    #print('rd_mask.shape', rd_mask.shape)

                    # Case without the CoL
                    rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                    for c in criterion]
                    
                    rd_loss = torch.mean(torch.stack(rd_losses))
                    # ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                    #              for c in ra_criterion]
                    # ra_loss = torch.mean(torch.stack(ra_losses))
                    loss = rd_loss
                    rd_losses_reduced = reduce_value(rd_losses)
                    rd_loss_reduced = reduce_value(rd_loss)
                    # ra_losses_reduced = reduce_value(ra_losses)
                    # ra_loss_reduced = reduce_value(ra_loss)
                    loss_reduced = reduce_value(loss)

                    loss.backward()
                    # grad_monitor.step()
                    optimizer.step()

                    # print in main proccessor
                    if self.rank == 0:
                        running_losses.append(loss_reduced.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss_reduced.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses_reduced[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses_reduced[1].data.cpu().numpy()[()])
                        # ra_running_losses.append(ra_loss_reduced.data.cpu().numpy()[()])
                        # ra_running_global_losses[0].append(ra_losses_reduced[0].data.cpu().numpy()[()])
                        # ra_running_global_losses[1].append(ra_losses_reduced[1].data.cpu().numpy()[()])
                        # if nb_losses > 2:
                        #     coherence_running_losses.append(coherence_loss_reduced.data.cpu().numpy()[()])

                        if iteration % self.loss_step == 0:
                            train_loss = np.mean(running_losses)
                            rd_train_loss = np.mean(rd_running_losses)
                            rd_train_losses = [np.mean(sub_loss) for sub_loss in rd_running_global_losses]
                            # ra_train_loss = np.mean(ra_running_losses)
                            # ra_train_losses = [np.mean(sub_loss) for sub_loss in ra_running_global_losses]
                            # zlw@20220302

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
                            ra_running_losses = list()
                            # zlw@20220107 get_lr() --> get_last_lr()
                            self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if self.reck == "ReCK-1":
                            self.net.remove_hooks()
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
                                # zlw@20220227
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
                                # fxy20251030 lsf
                                print('[Epoch {}/{}] Val miou_by_class: '
                                    'RD={}'.format(epoch+1,
                                                            self.nb_epochs,
                                                            val_metrics[signal_type]['miou_by_class'],
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
                                # zlw@20220227
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

                            # lt @20230522
                            # if val_metrics[signal_type]['dice'] > best_val_dop and iteration > 0:
                            #     best_val_dop = val_metrics[signal_type]['dice']
                            #     flag_save = True
                            #     name_result.append(signal_type)
                            
                        # saving results w/ best dice zlw@20220704
                        test_metrics = self.tester.predict(self.net, test_loader,
                                                               add_temp=add_temp)
                        if self.rank == 0:
                            self.visualizer.update_multi_test_metrics(signal_type, test_metrics, iteration)
                            # if test_metrics['global_dice'] > best_test_dice and iteration > 0:
                            #     best_test_dice = test_metrics['global_dice']
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
                                # zlw@20220227
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
                                # zlw@20220227
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

                            self.results['epoch'] = epoch + 1
                            self.results['rd_train_loss'] = rd_train_loss.item()
                            self.results['train_loss'] = train_loss.item()
                            self.results['val_metrics'] = val_metrics
                            self.results['test_metrics'] = test_metrics
                    
                            
                            if test_metrics[signal_type]['dice'] > best_test_dop and iteration > 0:
                                best_test_dop = test_metrics[signal_type]['dice']
                                # if best_test_dop > 0.67:
                                #     exit(0)
                                flag_save = True
                                name_result.append(self.model_name)
                            
                            if flag_save:
                                self._save_results(name_result)
                                flag_save = False
                                name_result = []
                            if self.reck == "ReCK-1":
                                self.net.register_hooks()
                                # dist.barrier() 
                        self.net.train()  # Train mode after evaluation process

                    iteration += 1
        if self.rank == 0:
            self.writer.close()
            if self.reck == "ReCK-1":
                self.net.remove_hooks()
            

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

    def _save_results(self, names):
        for name in names:
            results_path = self.paths['results'] / (name + '_' + 'results.json')
            model_path = self.paths['results'] / (name + '.pt')
            with open(results_path, "w") as fp:
                json.dump(self.results, fp)
            # zlw@20240424 self.net.module.state_dict for higher torch version
            torch.save(self.net.state_dict(), model_path)

    def _set_seeds(self):
        # each processor has its own seed
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
        random.seed(self.numpy_seed)


            


