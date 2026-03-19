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
import torch.distributed as dist
import matplotlib.pyplot as plt  
import matplotlib  

matplotlib.use('Agg')  


class Model_global_dataset(nn.Module):
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

        self.start_epoch = 0
        self.start_iteration = 0
        self.best_test_dop = 0.0

    def train(self, add_temp=False, checkpoint_path=None):
        if self.rank == 0:
            self.writer.add_text('Comments', self.comments)
        train_loader, val_loader, test_loader = self.dataloaders
        transformations = get_transformations(self.transform_names, sizes=(self.w_size, self.h_size))
        self._set_seeds()

        if self.dataset in ['CWRdata', 'PDRdata', 'Carrada_RD', 'CarradaRAC_RD', 'XPDdata', 'RDSyn', 'NuaaSim', "LSFJamming"]:
            signal_type = 'range_doppler'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device)
        elif self.dataset in ['Carrada_RA', 'CarradaRAC_RA']:
            signal_type = 'range_angle'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device)
        else:
            raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
        nb_losses = len(criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]

        optimizer = optim.AdamW(self.net.parameters(), lr=self.lr)
        if self.schedular_type == 'exp':
            scheduler = ExponentialLR(optimizer, gamma=0.9)
        elif self.schedular_type == 'cos':
            scheduler = CosineAnnealingLR(optimizer, T_max=self.T_max-1, eta_min=1e-7)
        else:
            raise KeyError("we only implement schedular of exp and cos")

        if checkpoint_path is not None and os.path.exists(checkpoint_path):
            self.start_epoch, self.start_iteration, self.best_test_dop, opt_state, sch_state = self._load_checkpoint(checkpoint_path)
            model_vis_flag = True
            optimizer.load_state_dict(opt_state)
            scheduler.load_state_dict(sch_state)
        else:
            self.start_epoch = 0
            self.start_iteration = 0
            self.best_test_dop = 0.0
            model_vis_flag = True

        iteration = self.start_iteration 
        flag_save = False
        name_result = []
        best_val_dop = 0
        best_val_ang = 0
        best_val_dop_plus_ang = 0
        best_test_ang = 0
        best_test_dop_plus_ang = 0


        all_datasets = []
        for seq_name, seq in train_loader:
            if self.dataset == 'LSFJamming':
                path_to_frames = self.paths['LSFJamming'] / seq_name[0]
            elif self.dataset == 'NuaaSim':
                path_to_frames = self.paths['NuaaSim'] / seq_name[0]
            elif self.dataset in ['Carrada_RA', 'Carrada_RD', 'CarradaRAC_RA', 'CarradaRAC_RD']:
                path_to_frames = self.paths['Carrada'] / seq_name[0]
            dataset = PeCoRSVDataset(
                dataset=seq,
                dataset_type=self.dataset,
                annotation_type=self.annot_type,
                signal_type=signal_type,
                path_to_frames=path_to_frames,
                process_signal=self.process_signal,
                n_frames=self.n_frames,
                cmplx_flag=self.cmplx_flag,
                transformations=transformations,
                add_temp=add_temp
            )
            all_datasets.append(dataset)
        combined_dataset = ConcatDataset(all_datasets)

        for epoch in range(self.start_epoch, self.nb_epochs):
            if self.schedular_type == 'exp':
                if epoch % self.lr_step == 0 and epoch != 0:
                    scheduler.step()
            else:
                if epoch != 0 and epoch < self.T_max:
                    scheduler.step()

            if self.distributed:
                sampler = torch.utils.data.distributed.DistributedSampler(combined_dataset)
                sampler.set_epoch(epoch)  
            else:
                sampler = torch.utils.data.RandomSampler(combined_dataset)

            frame_loader = DataLoader(
                combined_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                num_workers=self.num_workers,
                persistent_workers=True,
                pin_memory=self.pin_mem,
                drop_last=True
            )

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

                rd_outputs = self.net(rd_data)
                rd_outputs = rd_outputs.to(self.device)
                rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1)) for c in criterion]
                rd_loss = torch.mean(torch.stack(rd_losses))
                loss = rd_loss

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

                        self.visualizer.update_multi_train_loss(train_loss, rd_train_loss, rd_train_losses, iteration)
                        self.visualizer.update_learning_rate(scheduler.get_last_lr()[0], iteration)

                        running_losses = list()
                        rd_running_losses = list()


                if iteration % self.val_step == 0 and iteration > 0:
                    if self.reck == "ReCK-1":
                        self.net.remove_hooks()

                    if iteration % self.viz_step == 0 and iteration > 0:
                        val_metrics = self.tester.predict(self.net, val_loader, iteration, add_temp=add_temp)
                    else:
                        val_metrics = self.tester.predict(self.net, val_loader, add_temp=add_temp)

                    if self.rank == 0:
                        self.visualizer.update_multi_val_metrics(signal_type, val_metrics, iteration)
                        if signal_type == 'range_doppler':
                            print('[Epoch {}/{}] Val losses: RD={}'.format(epoch+1, 
                                                                           self.nb_epochs, 
                                                                           val_metrics[signal_type]['loss']))
                            print('[Epoch {}/{}] Val Prec: RD={}'.format(epoch+1, 
                                                                         self.nb_epochs, 
                                                                         val_metrics[signal_type]['prec'], ))
                            print('[Epoch {}/{}] Val mIoU: RD={}'.format(epoch+1, 
                                                                         self.nb_epochs, 
                                                                         val_metrics[signal_type]['miou']))
                            print('[Epoch {}/{}] Val Dice: RD={}'.format(epoch+1, 
                                                                         self.nb_epochs, 
                                                                         val_metrics[signal_type]['dice']))
                            print('[Epoch {}/{}] Val mIoU by class: RD={}'.format(epoch+1, 
                                                                                  self.nb_epochs, 
                                                                                  val_metrics[signal_type]['miou_by_class']))

                    test_metrics = self.tester.predict(self.net, test_loader, add_temp=add_temp)

                    if self.rank == 0:
                        self.visualizer.update_multi_test_metrics(signal_type, test_metrics, iteration)
                        if signal_type == 'range_doppler':
                            print('[Epoch {}/{}] Test losses: RD={}'.format(epoch+1, 
                                                                            self.nb_epochs, 
                                                                            test_metrics[signal_type]['loss']))
                            print('[Epoch {}/{}] Test Prec: RD={}'.format(epoch+1, 
                                                                                self.nb_epochs, 
                                                                                test_metrics[signal_type]['prec']))
                            print('[Epoch {}/{}] Test mIoU: RD={}'.format(epoch+1, 
                                                                          self.nb_epochs, 
                                                                          test_metrics[signal_type]['miou']))
                            print('[Epoch {}/{}] Test Dice: RD={}'.format(epoch+1, 
                                                                          self.nb_epochs, 
                                                                          test_metrics[signal_type]['dice']))
                            print('[Epoch {}/{}] Test mIoU by class: RD={}'.format(epoch+1, 
                                                                                   self.nb_epochs, 
                                                                                   test_metrics[signal_type]['miou_by_class']))

                        self.results['epoch'] = epoch + 1
                        self.results['rd_train_loss'] = rd_train_loss.item()
                        self.results['train_loss'] = train_loss.item()
                        self.results['val_metrics'] = val_metrics
                        self.results['test_metrics'] = test_metrics

                        if test_metrics[signal_type]['dice'] > self.best_test_dop and iteration > 0:
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
                                best_dice=self.best_test_dop
                            )
                            flag_save = False
                            name_result = []

                    self.net.train()
                    if self.reck == "ReCK-1":
                        self.net.register_hooks()

                iteration += 1 

            if epoch % 10 == 0 and self.rank == 0:
                checkpoint_name = [f'{self.model_name}_epoch_{epoch:04d}']
                self._save_results(
                    checkpoint_name,
                    save_full_checkpoint=True,
                    current_epoch=epoch,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    current_iteration=iteration,
                    best_dice=self.best_test_dop
                )

        if self.rank == 0:
            self.writer.close()
            if self.reck == "ReCK-1":
                self.net.remove_hooks()
            print(f"Training completed! Best Test Dice: {self.best_test_dop:.4f}")

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

    def _set_seeds(self):
        torch.cuda.manual_seed_all(self.torch_seed)
        torch.manual_seed(self.torch_seed)
        np.random.seed(self.numpy_seed)
        random.seed(self.numpy_seed)

    def _save_results(self, names, save_full_checkpoint=False, current_epoch=None,
                    optimizer=None, scheduler=None, current_iteration=None, best_dice=None):
        """
        Save model results:
        - Best model: inference weights + results JSON (+ optional full checkpoint)
        - Periodic: only full checkpoint for resuming training
        """
        for filename in names:
            save_dir = self.paths['results']
            save_dir.mkdir(parents=True, exist_ok=True)

            net_state = self.net.module.state_dict() if (
                self.distributed or isinstance(self.net, nn.parallel.DataParallel)
            ) else self.net.state_dict()

            # Best model (no "epoch" in filename)
            if "epoch" not in filename:
                # Inference weights
                weights_path = save_dir / f"{filename}.pt"
                torch.save(net_state, weights_path)

                # Results JSON
                json_path = save_dir / f"{filename}.json"
                with open(json_path, "w") as fp:
                    json.dump(self.results, fp, indent=2)

                # Optional full checkpoint for resuming training
                if save_full_checkpoint:
                    checkpoint_path = save_dir / f"{filename}_checkpoint_epoch{current_epoch}.pt"
                    checkpoint = {
                        'epoch': current_epoch,
                        'iteration': current_iteration,
                        'net_state_dict': net_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_dice': best_dice,
                        'results': self.results,
                        'cfg': self.cfg
                    }
                    torch.save(checkpoint, checkpoint_path)

            # Periodic checkpoint (filename contains "epoch")
            else:
                if save_full_checkpoint:
                    checkpoint_path = save_dir / f"{filename}_checkpoint_epoch{current_epoch}.pt"
                    checkpoint = {
                        'epoch': current_epoch,
                        'iteration': current_iteration,
                        'net_state_dict': net_state,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'best_dice': best_dice,
                        'results': self.results,
                        'cfg': self.cfg
                    }
                    torch.save(checkpoint, checkpoint_path)

    def _load_checkpoint(self, checkpoint_path=None):
        """
        Resume training from a given checkpoint.

        Parameters
        ----------
        checkpoint_path: str or Path
            Path to the checkpoint file. Supports both best-model checkpoints
            and periodic checkpoints.
        """
        if not os.path.exists(checkpoint_path):
            if self.rank == 0:
                print(f"Checkpoint does not exist: {checkpoint_path}, starting from scratch.")
            return 0, 0, 0.0

        if self.rank == 0:
            print(f"Resuming training from checkpoint: {checkpoint_path}")

        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=False
            )
        except Exception as e:
            if self.rank == 0:
                print(f"Failed to load checkpoint: {e}, starting from scratch.")
            return 0, 0, 0.0

        required_keys = ['net_state_dict', 'epoch', 'iteration', 'best_dice']
        missing_keys = [key for key in required_keys if key not in checkpoint]
        if missing_keys:
            if self.rank == 0:
                print(f"Invalid checkpoint format, missing keys: {missing_keys}, starting from scratch.")
            return 0, 0, 0.0

        net_state_dict = checkpoint['net_state_dict']

        # Handle (D)DP / DataParallel prefix
        if self.distributed or isinstance(self.net, nn.parallel.DataParallel):
            if not list(net_state_dict.keys())[0].startswith('module.'):
                net_state_dict = {f'module.{k}': v for k, v in net_state_dict.items()}
        else:
            net_state_dict = {k.replace('module.', ''): v for k, v in net_state_dict.items()}

        try:
            self.net.load_state_dict(net_state_dict, strict=True)
        except Exception as e:
            if self.rank == 0:
                print(f"Failed to load model weights: {e}, starting from scratch.")
            return 0, 0, 0.0

        start_epoch = checkpoint['epoch'] + 1
        start_iteration = checkpoint['iteration']
        best_dice = checkpoint['best_dice']
        self.results = checkpoint.get('results', {})
        opt_state = checkpoint.get('optimizer_state_dict')
        sch_state = checkpoint.get('scheduler_state_dict')

        if self.rank == 0:
            print("Checkpoint loaded successfully:")
            print(f"   - Resumed starting epoch: {start_epoch}")
            print(f"   - Resumed global iteration: {start_iteration}")
            print(f"   - Restored best Test Dice: {best_dice:.4f}")
            
        return start_epoch, start_iteration, best_dice, opt_state, sch_state