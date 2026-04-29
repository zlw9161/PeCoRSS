"""Class to test a model"""
# single view version
# merge model resume & cos lr strategy
# zlw@20230818
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from svrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, \
    get_qualitatives
from svrss.utils.paths import Paths
from svrss.utils.metrics import Evaluator
from svrss.loaders.dataloaders import CarradaDataset


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
        self.transform_names = self.cfg['transformations'].split(',')
        self.norm_type = self.cfg['norm_type']
        self.paths = Paths().get()
        self.test_results = dict()
        self.visualizer = visualizer

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
        if self.signal_type in 'range_doppler':
            sv_criterion = define_loss('range_doppler', self.custom_loss, self.device)
        elif self.signal_type in 'range_angle':
            sv_criterion = define_loss('range_angle', self.custom_loss, self.device)
        nb_losses = len(sv_criterion)
        running_losses = list()
        running_ce_losses = list()
        running_dice_losses = list()
        sv_metrics = Evaluator(num_class=self.nb_classes)
        if iteration:
            rand_seq = np.random.randint(len(seq_loader))
        with torch.no_grad():
            for i, sequence_data in enumerate(seq_loader):
                seq_name, seq = sequence_data
                path_to_frames = self.paths['carrada'] / seq_name[0]
                frame_dataloader = DataLoader(CarradaDataset(seq,
                                                             self.annot_type,
                                                             path_to_frames,
                                                             self.process_signal,
                                                             self.n_frames,
                                                             transformations,
                                                             add_temp),
                                              shuffle=False,
                                              batch_size=2,
                                              num_workers=4)
                if iteration and i == rand_seq:
                    rand_frame = np.random.randint(len(frame_dataloader))
                if get_quali:
                    quali_iter_sv = self.n_frames - 1
                for j, frame in enumerate(frame_dataloader):
                    if self.signal_type in 'range_doppler':
                        sv_data = frame['rd_matrix'].to(self.device).float()
                        sv_mask = frame['rd_mask'].to(self.device).float()
                        sv_data = normalize(sv_data, 'range_doppler', norm_type=self.norm_type)
                    elif self.signal_type in 'range_angle':
                        sv_data = frame['ra_matrix'].to(self.device).float()
                        sv_mask = frame['ra_mask'].to(self.device).float()
                        sv_data = normalize(sv_data, 'range_angle', norm_type=self.norm_type)

                    #zyc@20230920
                    sv_outputs = net(sv_data)
                    sv_outputs = sv_outputs.to(self.device)
                    if get_quali:
                        quali_iter_sv = get_qualitatives(sv_outputs, sv_mask, self.paths,
                                                         seq_name, quali_iter_sv, self.signal_type)
                    sv_metrics.add_batch(torch.argmax(sv_mask, axis=1).cpu(),
                                         torch.argmax(sv_outputs, axis=1).cpu())

                    if nb_losses < 3:
                        # Case without the CoL
                        sv_losses = [c(sv_outputs, torch.argmax(sv_mask, axis=1))
                                     for c in sv_criterion]
                        loss = torch.mean(torch.stack(sv_losses))

                    running_losses.append(loss.data.cpu().numpy()[()])
                    running_ce_losses.append(sv_losses[0].data.cpu().numpy()[()])
                    running_dice_losses.append(sv_losses[1].data.cpu().numpy()[()])

                    if iteration and i == rand_seq:
                        if j == rand_frame:
                            sv_pred_masks = torch.argmax(sv_outputs, axis=1)[:5]
                            sv_gt_masks = torch.argmax(sv_mask, axis=1)[:5]
                            sv_pred_grid = make_grid(transform_masks_viz(sv_pred_masks,
                                                                         self.nb_classes))
                            sv_gt_grid = make_grid(transform_masks_viz(sv_gt_masks,
                                                                       self.nb_classes))
                            self.visualizer.update_img_masks(sv_pred_grid, sv_gt_grid, iteration)
            self.test_results = dict()
            if self.signal_type in 'range_doppler':
                self.test_results['range_doppler'] = get_metrics(sv_metrics, np.mean(running_losses),
                                                                 [np.mean(running_ce_losses),
                                                                  np.mean(running_dice_losses)])
            elif self.signal_type in 'range_angle':
                self.test_results['range_angle'] = get_metrics(sv_metrics, np.mean(running_losses),
                                                               [np.mean(running_ce_losses),
                                                                np.mean(running_dice_losses)])

            sv_metrics.reset()
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
