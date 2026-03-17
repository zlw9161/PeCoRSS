"""A lot of functions used in our pipelines"""
import json
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import scipy
import scipy.stats

from mvrss.utils import MVRSS_HOME
from mvrss.losses.soft_dice import SoftDiceLoss
from mvrss.losses.soft_dice_v2 import SoftDiceLossV2
from mvrss.losses.coherence import CoherenceLoss
from mvrss.losses.soft_coherence import SoftCoherenceLoss
from mvrss.losses.sparse_coherence import SparseCoherenceLoss
from mvrss.losses.smospa_coherence import SmoSpaCoherenceLoss
from mvrss.losses.focal_loss import FocalLoss
from mvrss.losses.distribution_coherence import DistributionCoherenceLoss
from mvrss.losses.denoise_coherence import DenoiseCoherenceLoss
from mvrss.loaders.dataloaders_F import Rescale, Flip, HFlip, VFlip
from torch.utils.data import DataLoader
from mvrss.loaders.dataloaders_F import CarradaDataset


def get_class_weights(signal_type):
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

# zlw@20220302 Balance the RD and RA Losses
def get_loss_weight(signal_type):
    """Load weight for rd and ra loss
    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'

    RETURNS
    -------
    weight: numpy float
    """
    if signal_type in ('range_angle'):
        weight = 2.0
    elif signal_type in ('range_doppler'):
        weight = 1.0
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))
    return weight

def transform_masks_viz(masks, nb_classes):
    """Used for visualization"""
    masks = masks.unsqueeze(1)
    masks = (masks.float()/nb_classes)
    return masks


def get_metrics(metrics, loss, losses=None):
    """Structure the metric results

    PARAMETERS
    ----------
    metrics: object
        Contains statistics recorded during inference
    loss: tensor
        Loss value
    losses: list
        List of loss values

    RETURNS
    -------
    metrics_values: dict
    """
    metrics_values = dict()
    metrics_values['loss'] = loss.item()
    if isinstance(losses, list):
        metrics_values['loss_ce'] = losses[0].item()
        metrics_values['loss_dice'] = losses[1].item()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = metrics.get_pixel_prec_class()
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['prec'] = prec
    metrics_values['prec_by_class'] = prec_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    return metrics_values


def normalize(data, signal_type, norm_type='local'):
    """
    Method to normalise the radar views

    PARAMETERS
    ----------
    data: numpy array
        Radar view (batch)
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' and 'angle_doppler'
    norm_type: str
        Type of normalisation to apply
        Supported: 'local', 'tvt'

    RETURNS
    -------
    norm_data: numpy array
        normalised radar view
    """
    if norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        norm_data = torch.div(torch.sub(data, min_value), torch.sub(max_value, min_value))
        return norm_data

    elif signal_type == 'range_doppler':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'config_files' / 'rd_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            rd_stats = json.load(fp)
        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])

    elif signal_type == 'range_angle':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'config_files' / 'ra_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ra_stats = json.load(fp)
        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])

    elif signal_type == 'angle_doppler':
        if norm_type == 'tvt':
            file_path = MVRSS_HOME / 'config_files' / 'ad_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ad_stats = json.load(fp)
        min_value = torch.tensor(ad_stats['min_val'])
        max_value = torch.tensor(ad_stats['max_val'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    norm_data = torch.div(torch.sub(data, min_value),
                          torch.sub(max_value, min_value))
    return norm_data


def define_loss(signal_type, custom_loss, device, view_weight):
    """
    Method to define the loss to use during training

    PARAMETERS
    ----------
    signal_type: str
        Type of radar view
        Supported: 'range_doppler', 'range_angle' or 'angle_doppler'
    custom loss: str
        Short name of the custom loss to use
        Supported: 'wce', 'sdice', 'wce_w10sdice' or 'wce_w10sdice_w5col'
        Default: Cross Entropy is used for any other str
    devide: str
        Supported: 'cuda' or 'cpu'
    """
    if custom_loss == 'ce':
        loss = [nn.CrossEntropyLoss(), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'wce':
        weights = get_class_weights(signal_type)
        loss = [nn.CrossEntropyLoss(weight=weights.to(device).float()), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'sdice':
        loss = [SoftDiceLoss(), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'focal':
        loss = [FocalLoss(weight=None), lambda x, y: torch.tensor(0., device=x.device)]
    elif custom_loss == 'wce_w10sdice':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.)]
    elif custom_loss == 'wce_w10sdice_w5col':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        #MSE_loss = nn.MSELoss()
        #loss = [ce_loss, SoftDiceLoss(global_weight=view_weight), MSE_loss, CoherenceLoss(global_weight=5.)]
        loss = [ce_loss, SoftDiceLoss(global_weight=view_weight), CoherenceLoss(global_weight=5.)]
        #focal
    elif custom_loss == 'w20focal_w10sdice_w5col':
        weights = get_class_weights(signal_type)
        focal_loss = FocalLoss(weight=weights.to(device).float(), global_weight=20.)
        loss = [focal_loss, SoftDiceLoss(global_weight=10.), CoherenceLoss(global_weight=5.)]
    elif custom_loss == 'wfocal_w10sdice_w5col':
        weights = get_class_weights( signal_type)
        focal_loss = FocalLoss(weight=weights.to(device).float(), global_weight=1.)
        loss = [focal_loss, SoftDiceLoss(global_weight=10.), CoherenceLoss(global_weight=5.)]
        # lt @20230412
    elif custom_loss == 'wfocal_w10sdicev2_w5col':
        weights = get_class_weights( signal_type)
        focal_loss = FocalLoss(weight=weights.to(device).float())
        loss = [focal_loss, SoftDiceLossV2(global_weight=10.), CoherenceLoss(global_weight=5.)]
        # lt @20230416
    elif custom_loss == 'w20focal_wesub2sdicev2_w5col':
        weights = get_class_weights( signal_type)
        focal_loss = FocalLoss(weight=weights.to(device).float(), global_weight=20.)
        loss = [focal_loss, SoftDiceLossV2(global_weight=1e-2), CoherenceLoss(global_weight=5.)]
    elif custom_loss == 'w20focal_wesub1sdicev2_w5col':
        weights = get_class_weights(signal_type)
        focal_loss = FocalLoss(weight=weights.to(device).float(), global_weight=20.)
        loss = [focal_loss, SoftDiceLossV2(global_weight=1e-1), CoherenceLoss(global_weight=5.)]
    # zlw@20220302
    elif custom_loss == 'wce_w10sdice_w5col_sig_blnc':
        weights = get_class_weights(signal_type)
        loss_weight = get_loss_weight(signal_type)
        weights = loss_weight*weights
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        ce_loss = ce_loss
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.*loss_weight),
                CoherenceLoss(global_weight=5.)]
    # zlw@20220304
    elif custom_loss == 'wce_w10sdice_w5sofcol':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        ce_loss = ce_loss
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.),
                SoftCoherenceLoss(global_weight=5., relax_factor=0.2, margin=0.01)]
    # zlw@20220321
    elif custom_loss == 'wce_w10sdice_w5spacol':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                SparseCoherenceLoss(global_weight=5.)]
    # zlw@20220322
    elif custom_loss == 'wce_w10sdice_w5smospacol':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                SmoSpaCoherenceLoss(global_weight=5.)]
    # zlw@20220322
    elif custom_loss == 'wce_w10sdice_w5discol':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                DistributionCoherenceLoss(global_weight=5.)]
    # zlw@20220324
    elif custom_loss == 'wce_w10sdice_w5dnscol':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                DenoiseCoherenceLoss(global_weight=5.)]
    else:
        loss = nn.CrossEntropyLoss()
    return loss


def get_transformations(transform_names, split='train', sizes=None):
    """Create a list of functions used for preprocessing

    PARAMETERS
    ----------
    transform_names: list
        List of str, one for each transformation
    split: str
        Split currently used
    sizes: int or tuple (optional)
        Used for rescaling
        Default: None
    """
    transformations = list()
    if 'rescale' in transform_names:
        transformations.append(Rescale(sizes))
    if 'flip' in transform_names and split == 'train':
        transformations.append(Flip(0.5))
    if 'vflip' in transform_names and split == 'train':
        transformations.append(VFlip())
    if 'hflip' in transform_names and split == 'train':
        transformations.append(HFlip())
    return transformations


def mask_to_img(mask):
    """Generate colors per class, only 3 classes are supported"""
    mask_img = np.zeros((mask.shape[0],
                         mask.shape[1], 3), dtype=np.uint8)
    mask_img[mask == 1] = [255, 255, 0]
    mask_img[mask == 2] = [0, 255, 255]
    mask_img[mask == 3] = [255, 0, 0]
    mask_img = Image.fromarray(mask_img)
    return mask_img


def get_qualitatives(outputs, masks, paths, seq_name, quali_iter, signal_type=None):
    """
    Method to get qualitative results

    PARAMETERS
    ----------
    outputs: torch tensor
        Predicted masks
    masks: torch tensor
        Ground truth masks
    paths: dict
    seq_name: str
    quali_iter: int
        Current iteration on the dataset
    signal_type: str

    RETURNS
    -------
    quali_iter: int
    """
    if signal_type:
        folder_path = paths['logs'] / signal_type / seq_name[0]
    else:
        folder_path = paths['logs'] / seq_name[0]
    folder_path.mkdir(parents=True, exist_ok=True)
    outputs = torch.argmax(outputs, axis=1).cpu().numpy()
    masks = torch.argmax(masks, axis=1).cpu().numpy()
    for i in range(outputs.shape[0]):
        mask_img = mask_to_img(masks[i])
        mask_path = folder_path / 'mask_{}.png'.format(quali_iter)
        mask_img.save(mask_path)
        output_img = mask_to_img(outputs[i])
        output_path = folder_path / 'output_{}.png'.format(quali_iter)
        output_img.save(output_path)
        quali_iter += 1
    return quali_iter


def count_params(model):
    """Count trainable parameters of a PyTorch Model"""
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    return nb_params

def soft_mask_pdf(mask, signal_type, rf=1e-3, back_rd = scipy.stats.burr12(34.9515157685159, 2.1722270663431407, -0.48580492034916495, 60.60559988024067),
    cls1_rd = scipy.stats.exponnorm(6.001245879409501, 57.497402292024674, 1.682861392005516),
    cls2_rd = scipy.stats.exponnorm(5.531019338187409, 57.889627734415235, 1.797513346237698),
    cls3_rd = scipy.stats.exponnorm(4.431594797044482, 58.92546842859643, 2.348180230263354),
    back_ra =scipy.stats.cauchy(53.52227677399285, 0.8921131373133487),
    cls1_ra = scipy.stats.beta(1.8057417741306563, 4.178700106957197, 50.466799020059085, 45.864195945966024),
    cls2_ra = scipy.stats.exponpow(1.2513010537180245, 50.99913276966751, 22.911814286018107),
    cls3_ra = scipy.stats.beta(3.1006844559828908, 5.150272604213587, 46.33035702170341, 59.98657103097602)):


    b = mask.size()[0]
    c = 1
    h = mask.size()[2]
    w = mask.size()[3]
    n_clas = mask.size()[1]
    mask_tgt_cls1 = mask[:, 1, :, :].reshape(-1)  # [b*1*h*w]
    mask_tgt_cls2 = mask[:, 2, :, :].reshape(-1)  # [b*1*h*w]
    mask_tgt_cls3 = mask[:, 3, :, :].reshape(-1)  # [b*1*h*w]
    mask_tgt_bac = mask[:, 0, :, :].reshape(-1)  # [b*1*h*w]
    mask_size = mask_tgt_bac.size()
    device = mask_tgt_cls1.device

    if signal_type == 'range_doppler':
        ### sampling background noise
        rd_bac_samp = back_rd.rvs((b, n_clas, h, w))
        rd_bac_samp = torch.Tensor(back_rd.pdf(rd_bac_samp)).to(device)
        rd_bac = rd_bac_samp[:, 0, :, :].unsqueeze(dim=1)
        ### find tgt index
        cls1_rd_index = torch.nonzero(mask_tgt_cls1)
        cls2_rd_index = torch.nonzero(mask_tgt_cls2)
        cls3_rd_index = torch.nonzero(mask_tgt_cls3)
        ### sample tgt value
        cls1_rd_samp = cls1_rd.rvs(cls1_rd_index.size())#scipy.stats.gumbel_r(63.54582566090489, 6.561165424321485, size=cls1_rd_index.size())
        cls1_rd_samp = cls1_rd.pdf(cls1_rd_samp)
        mask_tgt_cls1[cls1_rd_index] = torch.Tensor(cls1_rd_samp).to(device)
        rd_bac_samp1 = rd_bac_samp[:, 1, :, :].reshape(-1)
        padding_z = torch.zeros(cls1_rd_index.size()).to(device)
        rd_bac_samp1[cls1_rd_index] = padding_z
        mask_tgt_cls1 += rd_bac_samp1*rf
        rd_cls1 = mask_tgt_cls1.reshape(b, c, h, w)

        cls2_rd_samp = cls2_rd.rvs(cls2_rd_index.size())#scipy.stats.gumbel_r(63.85117299240772, 6.488343792662817, size=cls2_rd_index.size())
        cls2_rd_samp = cls2_rd.pdf(cls2_rd_samp)
        mask_tgt_cls2[cls2_rd_index] = torch.Tensor(cls2_rd_samp).to(device)
        rd_bac_samp2 = rd_bac_samp[:, 2, :, :].reshape(-1)
        padding_z = torch.zeros(cls2_rd_index.size()).to(device)
        rd_bac_samp2[cls2_rd_index] = padding_z
        mask_tgt_cls2 += rd_bac_samp2*rf
        rd_cls2 = mask_tgt_cls2.reshape(b, c, h, w)

        cls3_rd_samp =cls3_rd.rvs(cls3_rd_index.size()) #scipy.stats.gumbel_r(65.07091995589674, 7.024154128372851, size=cls3_rd_index.size())
        cls3_rd_samp = cls3_rd.pdf(cls3_rd_samp)
        mask_tgt_cls3[cls3_rd_index] = torch.Tensor(cls3_rd_samp).to(device)
        rd_bac_samp3 = rd_bac_samp[:, 3, :, :].reshape(-1)
        padding_z = torch.zeros(cls3_rd_index.size()).to(device)
        rd_bac_samp3[cls3_rd_index] = padding_z
        mask_tgt_cls3 += rd_bac_samp3*rf
        rd_cls3 = mask_tgt_cls3.reshape(b, c, h, w)
        rd_soft_mask = torch.cat([rd_bac, rd_cls1, rd_cls2, rd_cls3], dim=1)  # b,4,h,w
        return rd_soft_mask

    if signal_type == 'range_angle':
        ### sampling background noise
        ra_bac_samp = back_ra.rvs((b, n_clas, h, w))
        ra_bac_samp = torch.Tensor(back_ra.pdf(ra_bac_samp)).to(device)
        ra_bac = ra_bac_samp[:, 0, :, :].unsqueeze(dim=1)
        ### find tgt index
        cls1_ra_index = torch.nonzero(mask_tgt_cls1)
        cls2_ra_index = torch.nonzero(mask_tgt_cls2)
        cls3_ra_index = torch.nonzero(mask_tgt_cls3)
        ### sample tgt value
        cls1_ra_samp = cls1_ra.rvs(cls1_ra_index.size())  # scipy.stats.gumbel_r(63.54582566090489, 6.561165424321485, size=cls1_rd_index.size())
        cls1_ra_samp = cls1_ra.pdf(cls1_ra_samp)
        mask_tgt_cls1[cls1_ra_index] = torch.Tensor(cls1_ra_samp).to(device)
        ra_bac_samp1 = ra_bac_samp[:, 1, :, :].reshape(-1)
        padding_z = torch.zeros(cls1_ra_index.size()).to(device)
        ra_bac_samp1[cls1_ra_index] = padding_z
        mask_tgt_cls1 += ra_bac_samp1*rf
        ra_cls1 = mask_tgt_cls1.reshape(b, c, h, w)

        cls2_ra_samp = cls2_ra.rvs(cls2_ra_index.size())  # scipy.stats.gumbel_r(63.85117299240772, 6.488343792662817, size=cls2_rd_index.size())
        cls2_ra_samp = cls2_ra.pdf(cls2_ra_samp)
        mask_tgt_cls2[cls2_ra_index] = torch.Tensor(cls2_ra_samp).to(device)
        ra_bac_samp2 = ra_bac_samp[:, 2, :, :].reshape(-1)
        padding_z = torch.zeros(cls2_ra_index.size()).to(device)
        ra_bac_samp2[cls2_ra_index] = padding_z
        mask_tgt_cls2 += ra_bac_samp2*rf
        ra_cls2 = mask_tgt_cls2.reshape(b, c, h, w)

        cls3_ra_samp = cls3_ra.rvs(cls3_ra_index.size())  # scipy.stats.gumbel_r(65.07091995589674, 7.024154128372851, size=cls3_rd_index.size())
        cls3_ra_samp = cls3_ra.pdf(cls3_ra_samp)
        mask_tgt_cls3[cls3_ra_index] = torch.Tensor(cls3_ra_samp).to(device)
        ra_bac_samp3 = ra_bac_samp[:, 3, :, :].reshape(-1)
        padding_z = torch.zeros(cls3_ra_index.size()).to(device)
        ra_bac_samp3[cls3_ra_index] = padding_z
        mask_tgt_cls3 += ra_bac_samp3*rf
        ra_cls3 = mask_tgt_cls3.reshape(b, c, h, w)
        ra_soft_mask = torch.cat([ra_bac, ra_cls1, ra_cls2, ra_cls3], dim=1)  # b,4,h,w
        return ra_soft_mask



def soft_mask_gaus():
    """softing track_mask from [0,1] distribution to Gaussian distribution  """

    def __init__(self):
        self.rd_tgt_mean_db =  [247.36124473, 140.52827164, 141.41046716, 144.49500235]
        self.ra_tgt_mean_db = [246.48394007, 121.83962928, 122.5454695,  142.09800318]
        self.rd_tgt_var_db =  [473.04930608, 290.671848,   286.84696244, 300.7555305 ]
        self.ra_tgt_var_db = [471.82160782, 241.93556995, 244.05629684, 293.11994744]

    def forward(self, mask, signal_type): #b, c, t, h, w
        mask_tgt = mask[:,:,-1,:,:] #b,c,h,w
        mask_tgt_cls1 = mask_tgt[:,1,:,:].unsqueeze(dim=1) #[b,1,h,w]
        mask_tgt_cls2 = mask_tgt[:, 2, :, :].unsqueeze(dim=1)  # [b,1,h,w]
        mask_tgt_cls3 = mask_tgt[:, 3, :, :].unsqueeze(dim=1)  # [b,1,h,w]
        mask_tgt_bac = mask_tgt[:, 0, :, :].unsqueeze(dim=1)  # [b,1,h,w]
        mask_size = mask_tgt_cls1.size()

        if signal_type in ('range_angle'):
            ra_cls1 = torch.normal(mean=self.ra_tgt_mean_db[1],std=torch.sqrt(self.ra_tgt_var_db[1], size=mask_size))
            ra_cls1 = ra_cls1 * mask_tgt_cls1
            ra_cls2 = torch.normal(mean=self.ra_tgt_mean_db[2], std=torch.sqrt(self.ra_tgt_var_db[2], size=mask_size))
            ra_cls2 = ra_cls2 * mask_tgt_cls2
            ra_cls3 = torch.normal(mean=self.ra_tgt_mean_db[3], std=torch.sqrt(self.ra_tgt_var_db[3], size=mask_size))
            ra_cls3 = ra_cls3 * mask_tgt_cls3
            ra_bac = torch.normal(mean=self.ra_tgt_mean_db[0], std=torch.sqrt(self.ra_tgt_var_db[0], size=mask_size))
            ra_bac = ra_bac * mask_tgt_bac
            ra_soft_mask = torch.cat([ra_bac, ra_cls1, ra_cls2, ra_cls3], dim=1) #b,4,h,w
            return ra_soft_mask
        if signal_type in ('range_dopper'):
            rd_cls1 = torch.normal(mean=self.rd_tgt_mean_db[1], std=torch.sqrt(self.rd_tgt_var_db[1], size=mask_size))
            rd_cls1 = rd_cls1 * mask_tgt_cls1
            rd_cls2 = torch.normal(mean=self.rd_tgt_mean_db[2], std=torch.sqrt(self.rd_tgt_var_db[2], size=mask_size))
            rd_cls2 = rd_cls2 * mask_tgt_cls2
            rd_cls3 = torch.normal(mean=self.rd_tgt_mean_db[3], std=torch.sqrt(self.rd_tgt_var_db[3], size=mask_size))
            rd_cls3 = rd_cls3 * mask_tgt_cls3
            rd_bac = torch.normal(mean=self.rd_tgt_mean_db[0], std=torch.sqrt(self.rd_tgt_var_db[0], size=mask_size))
            rd_bac = rd_bac * mask_tgt_bac
            rd_soft_mask = torch.cat([rd_bac, rd_cls1, rd_cls2, rd_cls3], dim=1)  # b,4,h,w
            return rd_soft_mask


####zyc@20230605
class CFAR():
    def __init__(self, gb, rb, alpha=0.2):
        self.gb = gb
        self.rb = rb
        self.padding = (self.gb + rb,
                        self.gb + rb,
                        self.gb + rb,
                        self.gb + rb)
        self.noise_size = (self.gb * 2 + self.rb + 1) * 4

        # padding order: l-r-t-b
        self.zero_padding = nn.ZeroPad2d(self.padding)

        self.alpha = alpha


    def forward(self, x): #b, c, t, h, w

        h, w = x.shape[-2], x.shape[-1]
        N = self.noise_size
        dtype = x.data.type()
        b = x.size(0)

        x_ori = x.copy()

        x = self.zero_padding(x)

        # let p_r denote the reference point positions
        # p_r shape: Bx2NxHxW
        p_r = self._get_p_r(b, h, w, N, dtype)
        # sample x_r (b, c, t, h, w, N) using p_r
        x_noise_mean = self._sample_x(x, p_r, N)

        # comparing x_ori and threthold and then denoise
        thld = self.alpha * x_noise_mean
        x_diff = x_ori - thld
        x_cfar = torch.where(x_diff > 0, x_ori, 0)

        return x_cfar

    # getting p_c (the center point coords) from the padded grid of input x

    def _get_p_c(self, b, h, w, N, dtype):
        # generating pc_grid
        p_c_x, p_c_y = torch.meshgrid(
            torch.arange(self.padding[2], h * self.pc_strd + self.padding[2], self.pc_strd),
            torch.arange(self.padding[0], w * self.pc_strd + self.padding[0], self.pc_strd))
        p_c_x = torch.flatten(p_c_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_c_y = torch.flatten(p_c_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        # p_c: 1x2NxHxW
        p_c = torch.cat([p_c_x, p_c_y], 1).type(dtype)
        # (B, 2N, h, w)
        p_c = p_c.repeat(b, 1, 1, 1)
        return p_c

        # generating peak receptive field grid

    def _gen_prf_grid(self, rb, gb, N, dtype):
        h_t = -(rb + gb)
        h_d = rb + gb
        w_l = -(rb + gb)
        w_r = rb + gb
        w_prf = (rb + gb) * 2 + 1
        h_prf = (rb + gb) * 2 + 1
        prf_x_idx, prf_y_idx = torch.meshgrid(
            torch.arange(h_t, h_d + 1),
            torch.arange(w_l, w_r + 1))
        # taking positions clockwise
        prf_xt = prf_x_idx[0:rb]
        prf_xr = prf_x_idx[rb:(h_prf - 2 * rb + 1), (w_prf - rb):w_prf]
        prf_xd = prf_x_idx[(h_prf - rb):h_prf]
        prf_xl = prf_x_idx[rb:(h_prf - 2 * rb + 1), 0:rb]
        prf_x = torch.cat([torch.flatten(prf_xt),
                           torch.flatten(prf_xr),
                           torch.flatten(prf_xd),
                           torch.flatten(prf_xl)], 0)
        prf_yt = prf_y_idx[0:rb]
        prf_yr = prf_y_idx[rb:(h_prf - 2 * rb + 1), (w_prf - rb):w_prf]
        prf_yd = prf_y_idx[(h_prf - rb):h_prf]
        prf_yl = prf_y_idx[rb:(h_prf - 2 * rb + 1), 0:rb]
        prf_y = torch.cat([torch.flatten(prf_yt),
                           torch.flatten(prf_yr),
                           torch.flatten(prf_yd),
                           torch.flatten(prf_yl)], 0)

        prf = torch.cat([prf_x, prf_y], 0)
        prf = prf.view(1, 2 * N, 1, 1).type(dtype)
        return prf

        # getting p_r positions from each p_c

    def _get_p_r(self, b, h, w, N, dtype):
        rb = self.rb
        gb = self.gb
        # (1, 2N, 1, 1)
        prf = self._gen_prf_grid(rb, gb, N, dtype)
        # (B, 2N, h, w)
        p_c = self._get_p_c(b, h, w, N, dtype)
        # (B, 2N, h, w)
        p_r = p_c + prf
        return p_r

    # expanding hitoric map using grid_gs
    def _sample_x(self, x_pad, p, N):
        # p:(b,2N,h,w)  x_pad:(b,c,t-1,h_p,w_p)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b,h,w,2n)
        b, h, w, _ = p.size()
        h_pad = x_pad.size(3)
        w_pad = x_pad.size(4)
        c = x_pad.size(1)
        t = x_pad.size(2)
        # (b, c, t-1, h_pad*w_pad)
        # strech each spatial channel of x_padas 1-D vector
        x_pad = x_pad.contiguous().view(b, c, t, -1)  # cuda0 float32
        # (b, h_pad, w_pad, n)
        # transform spatial coord of p into the 1-D index
        index = p[..., :N] * w_pad + p[..., N:]  # cpu,int64
        # index_x = torch.clamp(p[..., :N], 0, h_pad-1)
        # index_y = torch.clamp(p[..., N:], 0, w_pad-1)
        # index = index_x * w_pad + index_y
        # (b, c, t-1, h*w*N)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, t, -1, -1, -1)
        index = index.contiguous().unsqueeze(dim=1).expand(-1, c, -1, -1, -1, -1)
        index = index.contiguous().view(b, c, t, -1)
        index = index.long().cuda()  # cuda0 int64 [b,128,t-1,h*w*N]
        x_r = x_pad.gather(dim=-1, index=index)
        x_r = x_r.contiguous().view(b, c, t, h, w, N)
        x_r = x_r.mean(dim=-1) #(b, c, t, h, w)
        return x_r  # cuda0 float32





