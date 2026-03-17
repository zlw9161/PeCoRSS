"""Class to test a model"""
# single view version
# merge model resume & cos lr strategy
# zlw@20230818
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
import torch.nn as nn
from mvrss.utils import MVRSS_HOME

from mvrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, \
    get_qualitatives, soft_mask_pdf, soft_mask_gaus
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

    def __init__(self, cfg, track_loss=False, visualizer=None):
        self.cfg = cfg
        self.model_name = self.cfg['model']
        self.nb_classes = self.cfg['nb_classes']
        self.dataset_type = self.cfg['dataset']
        self.annot_type = self.cfg['annot_type']
        self.process_signal = self.cfg['process_signal']
        self.w_size = self.cfg['w_size']
        self.h_size = self.cfg['h_size']
        self.n_frames = self.cfg['nb_input_channels']
        self.batch_size = self.cfg['batch_size']
        self.device = self.cfg['device']
        self.custom_loss = self.cfg['custom_loss']
        self.trackloss = track_loss
        if self.trackloss:
            self.soft_trackloss = self.cfg['soft_trackloss']
        self.flag_coherence_track_loss = self.cfg['ct_loss']
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()
        self.visualizer = visualizer

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


    def predict(self, net, seq_loader, iteration=None, get_quali=False, add_temp=True):
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
        rd_metrics = Evaluator(num_class=self.nb_classes)
        ra_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths['Carrada'] / seq_name[0]
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.dataset_type,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp,
                                                             self.trackloss),
                                              shuffle=False,
                                              batch_size=2,
                                              num_workers=4)
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_sv = self.n_frames - 1
                for j, frame in enumerate(frame_dataloader):
                    
                    rd_data = frame['rd_matrix'].to(self.device).float()
                    ra_data = frame['ra_matrix'].to(self.device).float()
                    ad_data = frame['ad_matrix'].to(self.device).float()
                    rd_mask = frame['rd_mask'].to(self.device).float()
                    ra_mask = frame['ra_mask'].to(self.device).float()
                    rd_data = normalize(rd_data, 'range_doppler', norm_type=self.norm_type)
                    ra_data = normalize(ra_data, 'range_angle', norm_type=self.norm_type)
                    # zlw@20220325
                    if self.model_name != 'mvnet':
                        ad_data = normalize(ad_data, 'angle_doppler', norm_type=self.norm_type)
                        if self.model_name == 'tfoc':
                            rd_outputs, ra_outputs, x_pre_rd, x_pre_ra = net(rd_data, ra_data, ad_data)
                        else:
                            rd_outputs, ra_outputs= net(rd_data, ra_data, ad_data)
                    else:
                        if self.model_name == 'tfoc':
                            rd_outputs, ra_outputs, x_pre_rd, x_pre_ra = net(rd_data, ra_data)
                        else:
                            rd_outputs, ra_outputs = net(rd_data, ra_data)
                    rd_outputs = rd_outputs.to(self.device)
                    ra_outputs = ra_outputs.to(self.device)
                    if get_quali:
                        if self.trackloss:
                            quali_iter_rd = get_qualitatives(rd_outputs, rd_mask[:,:,-1,:,:], self.paths,
                                                                 seq_name, quali_iter_rd, 'range_doppler')
                            quali_iter_ra = get_qualitatives(ra_outputs, ra_mask[:,:,-1,:,:], self.paths,
                                                                 seq_name, quali_iter_ra, 'range_angle')
                        else:
                            quali_iter_rd = get_qualitatives(rd_outputs, rd_mask, self.paths,
                                                                 seq_name, quali_iter_rd, 'range_doppler')
                            quali_iter_ra = get_qualitatives(ra_outputs, ra_mask, self.paths,
                                                                 seq_name, quali_iter_ra, 'range_angle')
                    if self.trackloss:
                        rd_metrics.add_batch(torch.argmax(rd_mask[:,:,-1,:,:], axis=1).cpu(),
                                                 torch.argmax(rd_outputs, axis=1).cpu())
                        ra_metrics.add_batch(torch.argmax(ra_mask[:,:,-1,:,:], axis=1).cpu(),
                                                 torch.argmax(ra_outputs, axis=1).cpu())
                        if "col" in self.cfg["custom_loss"]:
                            # Case with the CoL
                            # Select the wCE and wSDice
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:,:,-1,:,:], axis=1))
                                             for c in rd_criterion[:-2]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:,:,-1,:,:], axis=1))
                                         for c in ra_criterion[:-2]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            # Coherence loss
                            coherence_loss = rd_criterion[-2](rd_outputs, ra_outputs)
                            if self.flag_coherence_track_loss:
                                coherence_track_loss = rd_criterion[-2](x_pre_rd, x_pre_ra)
                                coherence_loss = torch.mean(coherence_track_loss + coherence_track_loss)
                            loss = torch.mean(rd_loss + ra_loss + coherence_loss)
                        else:
                            # Case without the CoL
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask[:,:,-1,:,:], axis=1))
                                         for c in rd_criterion[:-1]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask[:,:,-1,:,:], axis=1))
                                         for c in ra_criterion[:-1]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            loss = torch.mean(rd_loss + ra_loss)
                        
                        x_mask_rd = torch.sum(rd_mask, dim=2)  # (b,4,h,w)
                        x_mask_rd[x_mask_rd > 1] = 1  # (b,4,h,w)
                        #track_loss_rd = rd_criterion[0](x_pre_rd, torch.argmax(x_mask_rd, axis=1))
                        x_mask_ra = torch.sum(ra_mask, dim=2)  # (b,4,h,w)
                        x_mask_ra[x_mask_ra > 1] = 1  # (b,4,h,w)
                        #track_loss_ra = ra_criterion[0](x_pre_ra, torch.argmax(x_mask_ra, axis=1))
                        if self.soft_trackloss:
                            x_mask_rd = soft_mask_pdf(x_mask_rd, signal_type ='range_doppler') #[b,4,h,w]
                            x_mask_ra = soft_mask_pdf(x_mask_ra, signal_type ='range_angle')
                            track_loss_rd = rd_criterion[-1](x_pre_rd, x_mask_rd)
                            track_loss_ra = ra_criterion[2-1](x_pre_ra, x_mask_ra)
                        else:
                            track_loss_rd = rd_criterion[-1](x_pre_rd, torch.argmax(x_mask_rd, axis=1))
                            track_loss_ra = ra_criterion[-1](x_pre_ra, torch.argmax(x_mask_ra, axis=1))
                        track_loss = torch.mean(track_loss_ra + track_loss_rd)
                        loss += track_loss
                    else:
                        rd_metrics.add_batch(torch.argmax(rd_mask, axis=1).cpu(),
                                             torch.argmax(rd_outputs, axis=1).cpu())
                        ra_metrics.add_batch(torch.argmax(ra_mask, axis=1).cpu(),
                                             torch.argmax(ra_outputs, axis=1).cpu())
                        if "col" in self.cfg["custom_loss"]:
                            # Case with the CoL
                            # Select the wCE and wSDice
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                         for c in rd_criterion[:-1]]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                         for c in ra_criterion[:-1]]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            # Coherence loss
                            coherence_loss = rd_criterion[-1](rd_outputs, ra_outputs)
                            loss = torch.mean(rd_loss + ra_loss + coherence_loss)
                        else:
                            # Case without the CoL
                            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                                         for c in rd_criterion]
                            rd_loss = torch.mean(torch.stack(rd_losses))
                            ra_losses = [c(ra_outputs, torch.argmax(ra_mask, axis=1))
                                         for c in ra_criterion]
                            ra_loss = torch.mean(torch.stack(ra_losses))
                            loss = torch.mean(rd_loss + ra_loss)
                        
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


                if iteration and i == rand_seq:
                    if j == rand_frame:
                        
                        rd_pred_masks = torch.argmax(rd_outputs, axis=1)[:5]
                        ra_pred_masks = torch.argmax(ra_outputs, axis=1)[:5]
                        if self.trackloss:
                            rd_gt_masks = torch.argmax(rd_mask[:, :, -1, :, :], axis=1)[:5]
                            ra_gt_masks = torch.argmax(ra_mask[:, :, -1, :, :], axis=1)[:5]
                        else:
                            rd_gt_masks = torch.argmax(rd_mask, axis=1)[:5]
                            ra_gt_masks = torch.argmax(ra_mask, axis=1)[:5]
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
            self.test_results = dict()
            
            self.test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                                 [np.mean(sub_loss) for sub_loss
                                                                  in rd_running_global_losses])
            self.test_results['range_angle'] = get_metrics(ra_metrics, np.mean(ra_running_losses),
                                                               [np.mean(sub_loss) for sub_loss
                                                                in ra_running_global_losses])

            if "col" in self.cfg["custom_loss"]:
                self.test_results['coherence_loss'] = np.mean(coherence_running_losses).item()
            if self.trackloss:
                self.test_results['track_loss'] = np.mean(track_running_losses).item()
            self.test_results['global_acc'] = (1 / 2) * (self.test_results['range_doppler']['acc'] +
                                                         self.test_results['range_angle']['acc'])
            self.test_results['global_prec'] = (1 / 2) * (self.test_results['range_doppler']['prec'] +
                                                          self.test_results['range_angle']['prec'])
            self.test_results['global_dice'] = (1 / 2) * (self.test_results['range_doppler']['dice'] +
                                                          self.test_results['range_angle']['dice'])
            # add global recall and mIoU zlw@20220704
            self.test_results['global_recall'] = (1 / 2) * (self.test_results['range_doppler']['recall'] +
                                                            self.test_results['range_angle']['recall'])
            self.test_results['global_miou'] = (1 / 2) * (self.test_results['range_doppler']['miou'] +
                                                          self.test_results['range_angle']['miou'])
            rd_metrics.reset()
            ra_metrics.reset()

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
