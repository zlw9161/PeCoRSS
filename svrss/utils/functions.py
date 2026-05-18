"""A lot of functions used in our pipelines"""
import json
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from svrss.utils import SVRSS_HOME
from svrss.losses.soft_dice import SoftDiceLoss
from svrss.losses.coherence import CoherenceLoss
from svrss.losses.soft_coherence import SoftCoherenceLoss
from svrss.losses.sparse_coherence import SparseCoherenceLoss
from svrss.losses.smospa_coherence import SmoSpaCoherenceLoss
from svrss.losses.distribution_coherence import DistributionCoherenceLoss
from svrss.losses.denoise_coherence import DenoiseCoherenceLoss
from svrss.loaders.dataloaders_sv import Rescale, Flip, HFlip, VFlip

# Class index order must match segmentation mask ids used in training / CE.
CARRADA_CLASS_WEIGHT_KEYS = ('background', 'pedestrian', 'cyclist', 'car')
# Official KuRALS keys (kurals/utils/functions.py, KuRALS_CW / KuRALS_PD).
KURALS_CW_CLASS_WEIGHT_KEYS = ('Background', 'UAV', 'Pedestrian', 'Vehicle')
KURALS_PD_CLASS_WEIGHT_KEYS = ('Background', 'UAV', 'Pedestrian', 'Car', 'Boat')


def build_track_heatmap(mask_seq: torch.Tensor, gamma: float = 0.9, use_history: bool = True) -> torch.Tensor:
    """Build a per-class temporal track heatmap from one-hot masks.

    Parameters
    ----------
    mask_seq: torch.Tensor
        Shape ``(B, C, T, H, W)``.
    gamma: float
        Temporal decay in ``(0, 1]``. Recent frames have larger weights.
    use_history: bool
        When True, use frames ``[0..T-2]`` to supervise the target frame ``T-1``.
    """
    if mask_seq.dim() != 5:
        raise ValueError(
            f"mask_seq must be 5D (B,C,T,H,W), got shape {tuple(mask_seq.shape)}"
        )

    _, c, t, _, _ = mask_seq.shape
    if t < 2 and use_history:
        raise ValueError("Need at least 2 frames to build history track heatmap.")

    k = int(t - 1 if use_history else t)
    masks = mask_seq[:, :, :k, :, :].float()
    device = masks.device

    exponents = torch.arange(k - 1, -1, -1, device=device, dtype=masks.dtype)
    gamma_t = torch.tensor(float(gamma), device=device, dtype=masks.dtype)
    weights = torch.pow(gamma_t, exponents)
    weights = weights / (weights.sum() + 1e-8)
    weights = weights.view(1, 1, k, 1, 1)

    heat = torch.sum(masks * weights, dim=2)
    heat = torch.clamp(heat, 0.0, 1.0)

    if c >= 2:
        fg = heat[:, 1:, :, :]
        bg = 1.0 - torch.max(fg, dim=1, keepdim=True).values
        heat = torch.cat([torch.clamp(bg, 0.0, 1.0), fg], dim=1)

    return heat


def class_weight_keys_for_cfg(cfg):
    """JSON key order for `rd_weights.json` / `ra_weights.json` tensors."""
    if cfg is None:
        return CARRADA_CLASS_WEIGHT_KEYS
    override = cfg.get('class_weight_keys')
    if override:
        return tuple(override)

    dl = (cfg.get('dataset') or '').strip().lower()
    if dl in ('cwr', 'cwrdata'):
        return KURALS_CW_CLASS_WEIGHT_KEYS
    if dl in ('pdr', 'pdrdata'):
        return KURALS_PD_CLASS_WEIGHT_KEYS
    return CARRADA_CLASS_WEIGHT_KEYS


def _weight_json_lookup(wdict, canon_key):
    """Resolve a weight entry; dataset files may mix ``background`` vs ``Background``."""
    if canon_key in wdict:
        return wdict[canon_key]
    lk = canon_key.lower()
    for jk, v in wdict.items():
        if str(jk).lower() == lk:
            return v
    return None


def _class_weight_vec_for_keys(wdict, keys):
    """Build weight vector if every key resolves; else ``(None, None)``."""
    vals = [_weight_json_lookup(wdict, k) for k in keys]
    if all(v is not None for v in vals):
        return np.array(vals, dtype=np.float64), keys
    return None, None


def _class_weight_vec_from_json(wdict, cfg):
    """Pick key order compatible with loaded JSON (KuRALS vs legacy Carrada names)."""
    keys = class_weight_keys_for_cfg(cfg)
    vec, used = _class_weight_vec_for_keys(wdict, keys)
    if vec is not None:
        return vec, used

    dl = '' if cfg is None else (cfg.get('dataset') or '').strip().lower()
    if dl in ('cwr', 'cwrdata', 'pdr', 'pdrdata') and all(
            _weight_json_lookup(wdict, k) is not None for k in CARRADA_CLASS_WEIGHT_KEYS):
        if dl in ('pdr', 'pdrdata'):
            raise KeyError(
                'KuRALS-PD expects five weight keys {}; found only Carrada-style keys {}'
                .format(list(class_weight_keys_for_cfg(cfg)),
                        list(CARRADA_CLASS_WEIGHT_KEYS))
            )
        warnings.warn(
            'Using Carrada-named keys in rd/ra_weights.json for KuRALS-CW '
            '(background, pedestrian, cyclist, car). Prefer official KuRALS keys '
            '(Background, UAV, Pedestrian, Vehicle).',
            UserWarning,
            stacklevel=2,
        )
        return (
            np.array([_weight_json_lookup(wdict, k) for k in CARRADA_CLASS_WEIGHT_KEYS],
                     dtype=np.float64),
            CARRADA_CLASS_WEIGHT_KEYS,
        )

    missing = [k for k in keys if _weight_json_lookup(wdict, k) is None]
    raise KeyError(
        'Class weight JSON missing keys {} (expected {}). Keys present: {}'.format(
            missing, list(keys), list(wdict.keys())
        )
    )


def resolve_class_weights_dir(cfg, paths):
    """Prefer dataset-root or explicit cfg weights (e.g. CWR KuCW/rd_weights.json).

    PARAMETERS
    ----------
    cfg: dict
        May set ``class_weights_dir`` / ``weights_json_dir`` (directory holding
        rd_weights.json or ra_weights.json).
    paths: dict
        Paths from Paths().get() (e.g. CWRdata, PDRdata).

    RETURNS
    -------
    pathlib.Path | None
        Directory to load weights from, or None to use repo ``config_files/``.
    """
    st = cfg.get('signal_type', '')
    if st == 'range_angle':
        file_name = 'ra_weights.json'
    elif st == 'range_doppler':
        file_name = 'rd_weights.json'
    else:
        return None

    cd = cfg.get('class_weights_dir') or cfg.get('weights_json_dir')
    if cd:
        root = Path(cd).expanduser()
        if (root / file_name).is_file():
            return root

    dl = (cfg.get('dataset') or '').strip().lower()
    if dl in ('cwr', 'cwrdata'):
        root = paths.get('CWRdata')
    elif dl in ('pdr', 'pdrdata'):
        root = paths.get('PDRdata')
    else:
        root = None
    if root is not None:
        root = Path(root)
        if (root / file_name).is_file():
            return root
    return None


def resolve_tvt_stats_json_path(signal_type, cfg, paths):
    """Resolve ``*_stats_all.json`` for ``normalize(..., tvt)``.

    Order: ``cfg['norm_stats_dir']`` / ``cfg['stats_json_dir']`` → same directory as
    class weights (if ``rd_weights.json`` exists) → CWR/PDR dataset root from ``paths``
    even when weight files are missing. Without a match, ``normalize`` falls back to
    ``svrss/config_files/*_stats_all.json`` (Carrada-scale), which breaks KuPD inputs in
    ``[0, 1]``.
    """
    if cfg is None or paths is None:
        return None
    if signal_type == 'range_angle':
        name = 'ra_stats_all.json'
    elif signal_type == 'range_doppler':
        name = 'rd_stats_all.json'
    elif signal_type == 'angle_doppler':
        name = 'ad_stats_all.json'
    else:
        return None

    nd = cfg.get('norm_stats_dir') or cfg.get('stats_json_dir')
    if nd:
        p = Path(nd).expanduser() / name
        if p.is_file():
            return p

    wd = resolve_class_weights_dir(cfg, paths)
    if wd is not None:
        p = Path(wd) / name
        if p.is_file():
            return p

    dl = (cfg.get('dataset') or '').strip().lower()
    root = None
    if dl in ('cwr', 'cwrdata'):
        root = paths.get('CWRdata')
    elif dl in ('pdr', 'pdrdata'):
        root = paths.get('PDRdata')
    if root is not None:
        p = Path(root) / name
        if p.is_file():
            return p
    return None


def resolved_class_weights_json_path(signal_type, cfg, paths):
    """Return the absolute JSON path loaded for weighted CE given cfg and Paths()."""
    wd = resolve_class_weights_dir(cfg, paths)
    return _class_weights_json_path(signal_type, weights_dir=wd).resolve()


def _class_weights_json_path(signal_type, weights_dir=None):
    if signal_type in ('range_angle',):
        file_name = 'ra_weights.json'
    elif signal_type in ('range_doppler',):
        file_name = 'rd_weights.json'
    else:
        raise ValueError('Signal type {} is not supported.'.format(signal_type))

    fallback = SVRSS_HOME / 'config_files' / file_name

    if weights_dir is None:
        return fallback

    p = Path(weights_dir).expanduser() / file_name
    return p if p.is_file() else fallback


def get_class_weights(signal_type, weights_dir=None, cfg=None):
    """Load class weights for custom loss

    PARAMETERS
    ----------
    signal_type: str
        Supported: 'range_doppler', 'range_angle'
    weights_dir: str | pathlib.Path | None
        If given and contains the matching ``*_weights.json``, use it; else
        ``svrss/config_files``.
    cfg: dict | None
        Used to pick JSON key order: Carrada vs KuRALS-CW vs KuRALS-PD (see
        ``class_weight_keys_for_cfg``). Optional ``class_weight_keys`` in cfg
        overrides the default list.

    RETURNS
    -------
    weights: torch.Tensor
    """
    file_path = _class_weights_json_path(signal_type, weights_dir=weights_dir)
    with open(file_path, 'r') as fp:
        wdict = json.load(fp)

    vec, used_keys = _class_weight_vec_from_json(wdict, cfg)

    if cfg is not None:
        nb = cfg.get('nb_classes')
        if nb is not None and int(nb) != len(used_keys):
            raise ValueError(
                'nb_classes={} does not match weight key count {} for dataset/class_weight_keys'
                .format(nb, len(used_keys))
            )

    weights = torch.from_numpy(vec)
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
    if isinstance(losses, list) and len(losses) > 1:
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


def _normalize_instance_minmax(data: torch.Tensor) -> torch.Tensor:
    """Per-batch-item and per-temporal-frame min-max (instance / frame-wise).

    Matches TFoc SV tensors shaped ``(B, C, T, H, W)`` after ``add_temp``: for each
    pair ``(b, t)`` normalise the slice over ``(C, H, W)``. Also supports common
    variants ``(B, T, H, W)``, ``(B, C, H, W)``, and ``(B, Cp, T, C, H, W)``.
    """
    x = data
    if not torch.is_tensor(x):
        x = torch.as_tensor(x)
    if x.dim() < 3:
        min_v = torch.min(x)
        max_v = torch.max(x)
        denom = torch.sub(max_v, min_v)
        denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
        return torch.div(torch.sub(x, min_v), denom)

    reduce_dims = None
    if x.dim() == 6:
        # e.g. (B, Cp, T, C, H, W)
        reduce_dims = (1, 3, 4, 5)
    elif x.dim() == 5:
        # (B, C, T, H, W) — primary layout for Conv3d / TFoc
        reduce_dims = (1, 3, 4)
    elif x.dim() == 4:
        d1, d2, d3 = x.shape[1], x.shape[2], x.shape[3]
        spatial_min = min(d2, d3)
        if d1 < spatial_min and spatial_min >= 4 * max(d1, 1):
            # (B, T, H, W): one channel per time slice
            reduce_dims = (2, 3)
        else:
            # (B, C, H, W) or ambiguous: one "frame" = all channels × spatial
            reduce_dims = (1, 2, 3)
    else:
        reduce_dims = tuple(range(1, x.dim()))

    min_v = x.amin(dim=reduce_dims, keepdim=True) 
    max_v = x.amax(dim=reduce_dims, keepdim=True)
    denom = torch.sub(max_v, min_v)
    denom = torch.where(denom.abs() < 1e-12, torch.ones_like(denom), denom)
    return torch.div(torch.sub(x, min_v), denom)


def normalize(data, signal_type, norm_type='local', stats_json_path=None):
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
        Supported: 'local', 'tvt', 'instance'
    stats_json_path: str | pathlib.Path | None
        For ``norm_type=='tvt'``, optional JSON with ``min_val`` / ``max_val``
        (e.g. PDR/CWR dataset ``rd_stats_all.json``). If ``None`` or missing,
        falls back to ``svrss/config_files/*_stats_all.json``.

    RETURNS
    -------
    norm_data: numpy array
        normalised radar view
    """
    if norm_type in ('local'):
        min_value = torch.min(data)
        max_value = torch.max(data)
        denom = torch.sub(max_value, min_value)
        # Avoid NaN when the tensor is constant (e.g. empty crop).
        denom = torch.where(
            denom.abs() < 1e-12,
            torch.ones_like(denom),
            denom,
        )
        norm_data = torch.div(torch.sub(data, min_value), denom)
        return norm_data

    if norm_type in ('instance',):
        return _normalize_instance_minmax(data)

    elif signal_type == 'range_doppler':
        if norm_type == 'tvt':
            file_path = Path(stats_json_path) if stats_json_path else None
            if file_path is None or not file_path.is_file():
                file_path = SVRSS_HOME / 'config_files' / 'rd_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            rd_stats = json.load(fp)
        min_value = torch.tensor(rd_stats['min_val'])
        max_value = torch.tensor(rd_stats['max_val'])

    elif signal_type == 'range_angle':
        if norm_type == 'tvt':
            file_path = Path(stats_json_path) if stats_json_path else None
            if file_path is None or not file_path.is_file():
                file_path = SVRSS_HOME / 'config_files' / 'ra_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ra_stats = json.load(fp)
        min_value = torch.tensor(ra_stats['min_val'])
        max_value = torch.tensor(ra_stats['max_val'])

    elif signal_type == 'angle_doppler':
        if norm_type == 'tvt':
            file_path = Path(stats_json_path) if stats_json_path else None
            if file_path is None or not file_path.is_file():
                file_path = SVRSS_HOME / 'config_files' / 'ad_stats_all.json'
        else:
            raise TypeError('Global type {} is not supported'.format(norm_type))
        with open(file_path, 'r') as fp:
            ad_stats = json.load(fp)
        min_value = torch.tensor(ad_stats['min_val'])
        max_value = torch.tensor(ad_stats['max_val'])

    else:
        raise TypeError('Signal {} is not supported.'.format(signal_type))

    denom = torch.sub(max_value, min_value)
    if not torch.is_tensor(denom):
        denom = torch.tensor(denom)
    denom = torch.where(
        denom.abs() < 1e-12,
        torch.ones_like(denom),
        denom,
    )
    norm_data = torch.div(torch.sub(data, min_value), denom)
    return norm_data


def define_loss(signal_type, custom_loss, device, weights_dir=None, cfg=None):
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
    weights_dir: str | pathlib.Path | None
        Optional directory with ``rd_weights.json`` / ``ra_weights.json`` (see
        ``resolve_class_weights_dir``).
    cfg: dict | None
        Training / inference config for class-weight key ordering (KuRALS vs Carrada).
    """
    kw = {'weights_dir': weights_dir, 'cfg': cfg}
    if custom_loss == 'ce':
        loss = nn.CrossEntropyLoss()
    elif custom_loss == 'wce':
        weights = get_class_weights(signal_type, **kw)
        loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
    elif custom_loss == 'sdice':
        loss = SoftDiceLoss()
    elif custom_loss == 'wce_w10sdice':
        weights = get_class_weights(signal_type, **kw)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, SoftDiceLoss(global_weight=10.)]
    elif custom_loss == 'wce_w10sdice_w5col':
        weights = get_class_weights(signal_type)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        MSE_loss = nn.MSELoss()
        sdice_w = float(view_weight) if view_weight is not None else 10.0
        loss = [ce_loss, SoftDiceLoss(global_weight=sdice_w), MSE_loss, CoherenceLoss(global_weight=5.)]
    # zlw@20220302
    elif custom_loss == 'wce_w10sdice_w5col_sig_blnc':
        weights = get_class_weights(signal_type, **kw)
        loss_weight = get_loss_weight(signal_type)
        weights = loss_weight*weights
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        ce_loss = ce_loss
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.*loss_weight),
                CoherenceLoss(global_weight=5.)]
    # zlw@20220304
    elif custom_loss == 'wce_w10sdice_w5sofcol':
        weights = get_class_weights(signal_type, **kw)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        ce_loss = ce_loss
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.),
                SoftCoherenceLoss(global_weight=5., relax_factor=0.2, margin=0.01)]
    # zlw@20220321
    elif custom_loss == 'wce_w10sdice_w5spacol':
        weights = get_class_weights(signal_type, **kw)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                SparseCoherenceLoss(global_weight=5.)]
    # zlw@20220322
    elif custom_loss == 'wce_w10sdice_w5smospacol':
        weights = get_class_weights(signal_type, **kw)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                SmoSpaCoherenceLoss(global_weight=5.)]
    # zlw@20220322
    elif custom_loss == 'wce_w10sdice_w5discol':
        weights = get_class_weights(signal_type, **kw)
        ce_loss = nn.CrossEntropyLoss(weight=weights.to(device).float())
        loss = [ce_loss, 
                SoftDiceLoss(global_weight=10.), 
                DistributionCoherenceLoss(global_weight=5.)]
    # zlw@20220324
    elif custom_loss == 'wce_w10sdice_w5dnscol':
        weights = get_class_weights(signal_type, **kw)
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





