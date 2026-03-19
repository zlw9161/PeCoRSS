"""Class to test a model"""
import json
import numpy as np
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from svrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from svrss.utils.paths import Paths
from svrss.utils.metrics import Evaluator
from svrss.loaders.dataloaders import PeCoRSVDataset
from svrss.utils.distributed_utils import reduce_value, get_rank, all_gather


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
        self.visualizer = visualizer
        self.model = self.cfg['model']
        self.dataset = self.cfg['dataset']
        self.nb_classes = self.cfg['nb_classes']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.distributed = self.cfg['distributed']
        self.num_workers = self.cfg['num_workers']
        self.cmplx_flag = self.cfg['cmplx_flag']
        self.paths = Paths().get()
        self.test_results = dict()

    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=False):
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
        if self.dataset in ['Carrada_RA', 'CarradaRAC_RA']:
            signal_type = 'range_angle'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device)
        elif self.dataset in ['CWRdata', 'PDRdata', 'Carrada_RD', 'CarradaRAC_RD', 'XPDdata', 'RDSyn', 'NuaaSim', 'LSFJamming']:
            signal_type = 'range_doppler'
            criterion = define_loss(self.dataset, signal_type, self.custom_loss, self.device)
        else:
            raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
        # rd_criterion = define_loss(self.dataset, 'range_doppler', self.custom_loss, self.device)
        # ra_criterion = define_loss('range_angle', self.custom_loss, self.device)
        nb_losses = len(criterion)
        running_losses = list()
        rd_running_losses = list()
        rd_running_global_losses = [list(), list()]
        # ra_running_losses = list()
        # ra_running_global_losses = [list(), list()]
        # coherence_running_losses = list()
        rd_metrics = Evaluator(num_class=self.nb_classes)
        # ra_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            # list_ori_idx1_rd = []
            # list_ori_idx2_rd = []
            # list_ori_idx1_ra = []
            # list_ori_idx2_ra = []
            # list_ori_idx1_ad = []
            # list_ori_idx2_ad = []
            # from tqdm import tqdm
            for i, sequence_data in enumerate(seq_loader):
                print('processing: {} / {}'.format(i, len(seq_loader)))
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
                    sampler_test = torch.utils.data.distributed.DistributedSampler(_dataset, shuffle=False)
                else:
                    sampler_test = torch.utils.data.SequentialSampler(_dataset)
                # lt@20240301 for pdr dataset, fix batch size to 2.
                if self.dataset in ['CWRdata', 'Carrada_RA', 'Carrada_RD', 'CarradaRAC_RA', 'CarradaRAC_RD', 'XPDdata', 'RDSyn', 'NuaaSim', 'LSFJamming']:
                    frame_dataloader = DataLoader(_dataset,
                                                  shuffle=False,
                                                  sampler=sampler_test,
                                                  batch_size=self.batch_size,
                                                  num_workers=self.num_workers)
                elif self.dataset == 'PDRdata':
                    frame_dataloader = DataLoader(_dataset,
                                                  shuffle=False,
                                                  sampler=sampler_test,
                                                  batch_size=1,
                                                  num_workers=0)
                else:
                    raise KeyError(f'Dataset {self.dataset} has not been supported yet.')
                
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_rd = self.n_frames-1
                    quali_iter_ra = self.n_frames-1
                
                for j, frame in enumerate(frame_dataloader):
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
                    rd_outputs = net(rd_data) # svrss:rd_data=[6,1,5,124,2048]
                    rd_outputs = rd_outputs.to(self.device)
                    
                    # ra_outputs = ra_outputs.to(self.device)
                    # ori_idx1_rd = all_gather(ori_idx1_rd)
                    # ori_idx2_rd = all_gather(ori_idx2_rd)
                    # ori_idx1_ra = all_gather(ori_idx1_ra)
                    # ori_idx2_ra = all_gather(ori_idx2_ra)
                    # ori_idx1_ad = all_gather(ori_idx1_ad)
                    # ori_idx2_ad = all_gather(ori_idx2_ad)

                    # list_ori_idx1_rd.append(ori_idx1_rd.cpu())
                    # list_ori_idx2_rd.append(ori_idx2_rd.cpu())
                    # list_ori_idx1_ra.append(ori_idx1_ra.cpu())
                    # list_ori_idx2_ra.append(ori_idx2_ra.cpu())
                    # list_ori_idx1_ad.append(ori_idx1_ad.cpu())
                    # list_ori_idx2_ad.append(ori_idx2_ad.cpu())

                    if get_quali:
                        quali_iter_rd = get_qualitatives(rd_outputs, rd_mask, self.paths,
                                                         seq_name, quali_iter_rd, signal_type,
                                                         self.dataset)
                        # quali_iter_ra = get_qualitatives(ra_outputs, ra_mask, self.paths,
                        #                                  seq_name, quali_iter_ra, 'range_angle')

                    rd_metrics.add_batch_dist(torch.argmax(rd_mask, axis=1),
                                         torch.argmax(rd_outputs, axis=1))
                    # ra_metrics.add_batch_dist(torch.argmax(ra_mask, axis=1),
                    #                      torch.argmax(ra_outputs, axis=1))
                    
                    # Case without the CoL
                    rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                    for c in criterion]
                    rd_loss = torch.mean(torch.stack(rd_losses))
                    # ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                    #                 for c in ra_criterion]
                    # ra_loss = torch.mean(torch.stack(ra_losses))
                    # loss = torch.mean(rd_loss + ra_loss)
                    loss = rd_loss
                    rd_losses = reduce_value(rd_losses)
                    rd_loss = reduce_value(rd_loss)
                    loss = reduce_value(loss)
                    
                    running_losses.append(loss.data.cpu().numpy()[()])
                    rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
                    rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
                    rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
                
                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                            # ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                            rd_gt_masks = torch.argmax(rd_mask, axis=1)[:5]
                            # ra_gt_masks = torch.argmax(ra_mask, axis=1)[:5]
                            rd_pred_grid = make_grid(transform_masks_viz(rd_pred_masks,
                                                                         self.nb_classes))
                            # ra_pred_grid = make_grid(transform_masks_viz(ra_pred_masks,
                            #                                              self.nb_classes))
                            rd_gt_grid = make_grid(transform_masks_viz(rd_gt_masks,
                                                                       self.nb_classes))
                            # ra_gt_grid = make_grid(transform_masks_viz(ra_gt_masks,
                            #                                            self.nb_classes))
                            if self.visualizer:
                                self.visualizer.update_multi_img_masks(rd_pred_grid, rd_gt_grid,
                                                                   iteration)
            self.test_results = dict()
            self.test_results[signal_type] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                             [np.mean(sub_loss) for sub_loss
                                                              in rd_running_global_losses])
            # self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
            #                                                [np.mean(sub_loss) for sub_loss
            #                                                 in ra_running_global_losses])
            # if nb_losses > 2:
            #     self.test_results['coherence_loss'] = np.mean(coherence_running_losses).item()
            self.test_results['global_acc'] = self.test_results[signal_type]['acc']
            self.test_results['global_prec'] = self.test_results[signal_type]['prec']
            self.test_results['global_dice'] = self.test_results[signal_type]['dice']
            # add global recall and mIoU zlw@20220704
            self.test_results['global_recall'] = self.test_results[signal_type]['recall']
            self.test_results['global_miou'] = self.test_results[signal_type]['miou']

            self.test_results['miou_by_class'] = self.test_results[signal_type]['miou_by_class']

            rd_metrics.reset()
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
