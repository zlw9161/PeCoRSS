"""Class to train a PyTorch model"""
# single view version
# merge model resume & cos lr strategy
# zlw@20230818

import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from thop import profile

from svrss.loaders.dataloaders import CarradaDataset
from svrss.learners.tester import Tester
from svrss.utils.functions import normalize, define_loss, get_transformations
from svrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer


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
        self.resume_path = self.cfg['resume']
        self.process_signal = self.cfg['process_signal']
        self.signal_type = self.cfg['signal_type']
        self.annot_type = self.cfg['annot_type']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.batch_size = self.cfg['batch_size']
        self.nb_epochs = self.cfg['nb_epochs']
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
        self.comments = self.cfg['comments']
        self.n_frames = self.cfg['nb_input_channels']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.is_shuffled = self.cfg['shuffle']
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.tester = Tester(self.cfg, self.visualizer)
        self.results = dict()

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
        self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names,
                                              sizes=(self.w_size, self.h_size))
        self._set_seeds()
        self.net.apply(self._init_weights)
        criterion = define_loss(self.signal_type, self.custom_loss, self.device)
        nb_losses = len(criterion)  # [0]: ce; [1]: dice
        running_losses = list()
        running_ce_losses = list()
        running_dice_losses = list()
        optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
        # scheduler = ExponentialLR(optimizer, gamma=0.9) # epoch = 300, lr_step = 20
        # lt @20230416 add cos lr strategy
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")
        iteration = 0
        # saving results w/ best dice zlw@20220704
        best_test_dice = 0
        if self.resume_path:
            checkpoint = torch.load(self.resume_path)
            start_epoch = checkpoint['epoch']
            self.net.load_state_dict(checkpoint['state_dict'])
            optimizer = optim.Adam(self.net.parameters(), lr=self.lr)
            optimizer.load_state_dict(checkpoint['optimizer'])
            best_dice = checkpoint['best_dice']
            print("=> Loaded checkpoint at epoch {} w/ dice {})".format(checkpoint['epoch'], best_dice))
            # zyc@20230320
            # if netdown == False:
            #     # zlw@20211012
            #     if torch.cuda.device_count() > 1:
            #         self.net = nn.DataParallel(self.net, device_ids=[0, 1])
            #     #self.net = nn.DataParallel(self.net, device_ids=[3])
            #     self.net.to(self.device)

        for epoch in range(start_epoch, self.nb_epochs):
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()
            for _, sequence_data in enumerate(train_loader):
                seq_name, seq = sequence_data
                path_to_frames = os.path.join(self.paths['carrada'], seq_name[0])
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=self.is_shuffled,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                for _, frame in enumerate(frame_dataloader):
                    if self.signal_type in 'range_doppler':
                        sv_data = frame['rd_matrix'].to(self.device).float()
                        sv_mask = frame['rd_mask'].to(self.device).float()
                        sv_data = normalize(sv_data, 'range_doppler', norm_type=self.norm_type)
                    elif self.signal_type in 'range_angle':
                        sv_data = frame['ra_matrix'].to(self.device).float()
                        sv_mask = frame['ra_mask'].to(self.device).float()
                        #zyc@20230821
                        sv_data = normalize(sv_data, 'range_angle', norm_type=self.norm_type)

                    optimizer.zero_grad()
                    # zyc@20230920
                    # #####################
                    # flops, params = profile(self.net.module, inputs = (sv_data,))
                    # print("**********************************")
                    # print("svrss-1frame", ":")
                    # print("FLOPs=", str(flops / (1000 ** 3)) + '{}'.format("G"))
                    # print("params=", str(params / (1000 ** 2)) + '{}'.format("M"))
                    # print("**********************************")


                    sv_outputs = self.net(sv_data)
                    sv_outputs = sv_outputs.to(self.device)


                    # # print('net', net)


                    # sv_losses = [c(sv_outputs[i], torch.argmax(sv_mask, axis=1))
                    #              for i, c in enumerate(criterion)]
                    sv_losses = [c(sv_outputs, torch.argmax(sv_mask, axis=1))
                                 for c in criterion]

                    loss = torch.mean(torch.stack(sv_losses))  # total loss?

                    loss.backward()
                    optimizer.step()
                    running_losses.append(loss.data.cpu().numpy()[()])
                    running_ce_losses.append(sv_losses[0].data.cpu().numpy()[()])
                    running_dice_losses.append(sv_losses[1].data.cpu().numpy()[()])

                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        train_ce_loss = np.mean(running_ce_losses)
                        train_dice_loss = np.mean(running_dice_losses)
                        if self.signal_type in 'range_doppler':
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train loss (RD View) {}'.format(epoch + 1,
                                                                   self.nb_epochs,
                                                                   iteration,
                                                                   train_loss))
                        elif self.signal_type in 'range_angle':
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train loss (RA View) {}'.format(epoch + 1,
                                                                   self.nb_epochs,
                                                                   iteration,
                                                                   train_loss))

                        if nb_losses > 1:
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train CE loss {}'.format(epoch + 1,
                                                            self.nb_epochs,
                                                            iteration,
                                                            train_ce_loss))
                            print('[Epoch {}/{}, iter {}]: '
                                  'Train Dice loss {}'.format(epoch + 1,
                                                              self.nb_epochs,
                                                              iteration,
                                                              train_dice_loss))
                            self.visualizer.update_sv_train_loss(self.signal_type,
                                                                 train_loss,
                                                                 train_ce_loss,
                                                                 train_dice_loss,
                                                                 iteration)
                        running_losses = list()
                        running_ce_losses = list()
                        running_dice_losses = list()

                        # zlw@20220107 get_lr() --> get_last_lr()
                        self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        self.visualizer.update_sv_val_metrics(self.signal_type,
                                                              val_metrics,
                                                              iteration)
                        if self.signal_type in 'range_doppler':
                            print('[Epoch {}/{}]: '
                                  'Val loss (RD View) {}'.format(epoch + 1,
                                                                 self.nb_epochs,
                                                                 val_metrics['range_doppler']['loss']))
                            print('[Epoch {}/{}] Val Metrics: '
                                  'Prec={}, mIoU={}, Dice={}'.format(epoch + 1,
                                                                     self.nb_epochs,
                                                                     val_metrics['range_doppler']['prec'],
                                                                     val_metrics['range_doppler']['miou'],
                                                                     val_metrics['range_doppler']['dice']))
                        elif self.signal_type in 'range_angle':
                            print('[Epoch {}/{}]: '
                                  'Val loss (RA View) {}'.format(epoch + 1,
                                                                 self.nb_epochs,
                                                                 val_metrics['range_angle']['loss']))
                            print('[Epoch {}/{}] Val Metrics: '
                                  'Prec={}, mIoU={}, Dice={}'.format(epoch + 1,
                                                                     self.nb_epochs,
                                                                     val_metrics['range_angle']['prec'],
                                                                     val_metrics['range_angle']['miou'],
                                                                     val_metrics['range_angle']['dice']))

                        '''if val_metrics['global_prec'] > best_val_prec and iteration > 0:
                            best_val_prec = val_metrics['global_prec']
                            '''
                        # saving results w/ best dice zlw@20220704
                        test_metrics = self.tester.predict(self.net, test_loader,
                                                           add_temp=add_temp)
                        self.visualizer.update_sv_test_metrics(self.signal_type,
                                                               test_metrics,
                                                               iteration)
                        if self.signal_type in 'range_doppler':
                            if test_metrics['range_doppler']['dice'] > best_test_dice and iteration > 0:
                                best_test_dice = test_metrics['range_doppler']['dice']
                                print('[Epoch {}/{}]: '
                                        'Test loss (RD View) {}'.format(epoch + 1,
                                                                          self.nb_epochs,
                                                                          test_metrics['range_doppler']['loss']))
                                print('[Epoch {}/{}] Test Metrics: '
                                          'Prec={}, mIoU={}, Dice={}'.format(epoch + 1,
                                                                             self.nb_epochs,
                                                                             test_metrics['range_doppler']['prec'],
                                                                             test_metrics['range_doppler']['miou'],
                                                                             test_metrics['range_doppler']['dice']))
                                self.results['train_ce_loss'] = train_ce_loss.item()
                                self.results['train_dice_loss'] = train_dice_loss.item()
                                self.results['train_loss'] = train_loss.item()
                                self.results['val_metrics'] = val_metrics
                                self.results['test_metrics'] = test_metrics
                                self._save_results({
                                    'epoch': epoch + 1,
                                    'arch': str(self.net),
                                    'state_dict': self.net.state_dict(),
                                    'best_dice': best_test_dice,
                                    'optimizer': optimizer.state_dict(),
                                })
                        elif self.signal_type in 'range_angle':
                            if test_metrics['range_angle']['dice'] > best_test_dice and iteration > 0:
                                best_test_dice = test_metrics['range_angle']['dice']
                                print('[Epoch {}/{}]: '
                                      'Test loss (RA View) {}'.format(epoch + 1,
                                                                      self.nb_epochs,
                                                                      test_metrics['range_angle']['loss']))
                                print('[Epoch {}/{}] Test Metrics: '
                                      'Prec={}, mIoU={}, Dice={}'.format(epoch + 1,
                                                                         self.nb_epochs,
                                                                         test_metrics['range_angle']['prec'],
                                                                         test_metrics['range_angle']['miou'],
                                                                         test_metrics['range_angle']['dice']))

                                self.results['train_ce_loss'] = train_ce_loss.item()
                                self.results['train_dice_loss'] = train_dice_loss.item()
                                self.results['train_loss'] = train_loss.item()
                                self.results['val_metrics'] = val_metrics
                                self.results['test_metrics'] = test_metrics
                                self._save_results({
                                    'epoch': epoch + 1,
                                    'arch': str(self.net),
                                    'state_dict': self.net.state_dict(),
                                    'best_dice': best_test_dice,
                                    'optimizer': optimizer.state_dict(),
                                })
                        self.net.train()  # Train mode after evaluation process
                    iteration += 1
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
