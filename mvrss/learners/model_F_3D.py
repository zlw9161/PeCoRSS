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
from mvrss.learners.tester_F_3D import Tester
from mvrss.utils.functions import normalize, define_loss, get_transformations
from mvrss.utils.tensorboard_visualizer import TensorboardMultiLossVisualizer



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
        self.writer = SummaryWriter(self.paths['writer'])
        self.visualizer = TensorboardMultiLossVisualizer(self.writer)
        self.mulmask = Mulmask
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
        rd_criterion = define_loss('range_doppler', self.custom_loss, self.device, view_weight=20.)
        ra_criterion = define_loss('range_angle', self.custom_loss, self.device, view_weight=10.)
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
                                                             add_temp,
                                                             self.mulmask),
                                              shuffle=self.is_shuffled,
                                              batch_size=self.batch_size,
                                              num_workers=4)
                for _, frame in enumerate(frame_dataloader):

                    if self.signal_type in 'range_doppler':
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
                                loss = torch.mean(torch.stack(rd_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                             for c in rd_criterion[:2]]
                                loss = torch.mean(torch.stack(rd_losses))
                            x_mask = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask[x_mask > 1] = 1  # (b,4,h,w)
                            track_loss = rd_criterion[0](x_pre, torch.argmax(x_mask, axis=1))
                            loss = torch.mean(loss + track_loss)
                        else:
                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion]
                                loss = torch.mean(torch.stack(rd_losses))

                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion[:2]]
                                loss = torch.mean(torch.stack(rd_losses))
                        loss.backward()
                        optimizer.step()
                        running_losses.append(loss.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                    elif self.signal_type in 'range_angle':
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
                            track_lo
                            ss = ra_criterion[0](x_pre, torch.argmax(x_mask, axis=1))
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
                        running_losses.append(loss.data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                    elif self.signal_type in 'range_doppler_angle':
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
                            rd_outputs, ra_outputs = self.net(rd_data, ra_data, ad_data)
                        else:
                            rd_outputs, ra_outputs = self.net(rd_data, ra_data)
                        rd_outputs = rd_outputs.to(self.device)
                        ra_outputs = ra_outputs.to(self.device)
                        if self.mulmask:
                            x_mask_rd = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                            x_mask_rd[x_mask_rd > 1] = 1  # (b,4,h,w)
                            x_mask_ra = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                            x_mask_ra[x_mask_ra > 1] = 1  # (b,4,h,w)
                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                             for c in rd_criterion]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, -1, :, :], axis=1))
                                             for c in ra_criterion]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                loss = torch.mean(rd_loss + ra_loss)
                            else:
                                # Case with the CoL
                                # Select the wCE and wSDice
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:, :, -1, :, :], axis=1))
                                             for c in rd_criterion[:2]]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:, :, -1, :, :], axis=1))
                                             for c in ra_criterion[:2]]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                # Coherence loss
                                coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                                loss = torch.mean(rd_loss + ra_loss + coherence_loss)

                        else:
                            if nb_losses < 3:
                                # Case without the CoL
                                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                             for c in rd_criterion]
                                rd_loss = torch.mean(torch.stack(rd_losses))
                                ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                             for c in ra_criterion]
                                ra_loss = torch.mean(torch.stack(ra_losses))
                                loss = torch.mean(rd_loss + ra_loss)
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
                                coherence_loss = rd_criterion[2](rd_outputs, ra_outputs)
                                loss = torch.mean(rd_loss + ra_loss + coherence_loss)
                        loss.backward()
                        optimizer.step()
                        running_losses.append(loss.data.cpu().numpy()[()])
                        rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                        rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                        rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                        ra_running_losses.append(ra_loss.data.cpu().numpy()[()])
                        ra_running_global_losses[0].append(ra_losses[0].data.cpu().numpy()[()])
                        ra_running_global_losses[1].append(ra_losses[1].data.cpu().numpy()[()])
                        if nb_losses > 2:
                            coherence_running_losses.append(coherence_loss.data.cpu().numpy()[()])


                    if iteration % self.loss_step == 0:
                        train_loss = np.mean(running_losses)
                        if self.signal_type in 'range_doppler':
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
                        elif self.signal_type in 'range_angle':
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
                        elif self.signal_type in 'range_doppler_angle':
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


                            if nb_losses > 2:
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
                        # zlw@20220107 get_lr() --> get_last_lr()
                        self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                    if iteration % self.val_step == 0 and iteration > 0:
                        if iteration % self.viz_step == 0 and iteration > 0:
                            val_metrics = self.tester.predict(self.net, val_loader, iteration,
                                                              add_temp=add_temp)
                        else:
                            val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                        self.visualizer.update_multi_val_metrics(val_metrics, iteration)
                        if self.signal_type in 'range_doppler':
                            print('[Epoch {}/{}] Val losses: '
                                  'RD={}'.format(epoch + 1,
                                                        self.nb_epochs,
                                                        val_metrics['range_doppler']['loss']))

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
                        elif self.signal_type in 'range_angle':
                            print('[Epoch {}/{}] Val losses: '
                                  'RA={}'.format(epoch + 1,
                                                 self.nb_epochs,
                                                 val_metrics['range_angle']['loss'], ))

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
                        elif self.signal_type in 'range_doppler_angle':
                            print('[Epoch {}/{}] Val losses: '
                                'RD={}, RA{}'.format(epoch+1,
                                                    self.nb_epochs,
                                                    val_metrics['range_doppler']['loss'],
                                                    val_metrics['range_angle']['loss'],))
                            if nb_losses > 2:
                                print('[Epoch {}/{}] Val Coherence Loss: '
                                'CoL={}'.format(epoch+1,
                                                self.nb_epochs,
                                                val_metrics['coherence_loss']))

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

                        if test_metrics['global_dice'] > best_test_dice and iteration > 0:
                            best_test_dice = test_metrics['global_dice']
                            if self.signal_type in 'range_doppler':
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

                            elif self.signal_type in 'range_angle':
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

                            elif self.signal_type in 'range_doppler_angle':
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
