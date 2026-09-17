"""
End-to-End Radar Echo Segmentation Visualization
"""

import os
import json
import math
import time
import numpy as np

# Import pandas early and touch an attribute to avoid a circular-import problem during Plotly serialisation
import pandas
pandas.Series  # Touch an attribute to make sure pandas is fully initialised

from scipy.fftpack import fft, fftshift
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from pathlib import Path
from scipy.io import loadmat
import scipy.io as scio
import torch.nn.init as init
import io
import base64
from PIL import Image
import plotly.graph_objs as go
import tempfile
import matplotlib.colors as mcolors
from collections import OrderedDict
import traceback


# ==================== Configuration ====================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== Logo Configuration ====================
# NOTE: the logo now lives in logos/, keep this in sync if the path changes
LOGO_PATH = os.path.join("logos", "logo.jpg")


def get_logo_base64():
    """Read the logo image and convert it to base64"""
    try:
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode()
    except Exception as e:
        print(f"[warn] Could not load logo from '{LOGO_PATH}': {e}")
        return ""


LOGO_BASE64 = get_logo_base64()

# ==================== Model Path Configuration ====================
# Model runs are discovered from disk at runtime, see asset_paths.discover_runs():
#   models/<dataset>/<run>/config.json   -> declares the model name
#   models/<dataset>/<run>/results/<model>.pt
# Nothing has to be maintained here when a dataset is added or a model is retrained.

# Default model and dataset metadata
DEFAULT_MODEL = 'echmo'

# Colour configuration - for the segmentation masks and the UI
COLORS = {
    'background': "#444486",
    'surface': '#14141f',
    'panel': '#1a1a2e',
    'primary': '#7c3aed',
    'secondary': '#f59e0b',
    'accent': '#06b6d4',
    'text': '#e2e8f0',
    'text_dim': '#94a3b8',
    'border': '#2d2d44',
}

# Segmentation colors (RGB) - make sure these colors are used consistently
SEG_COLORS_2 = [
    "#440154",     # 1. dark purple (fixed - background)
    "#FDE725",     # 2. bright yellow (fixed - main target)
]

SEG_COLORS_9 = [
    "#440154",     # 1. dark purple (fixed - background)
    "#FDE725",     # 2. bright yellow (fixed - main target)
    # --------------------------
    # First group: three **similar purple** entries
    "#B71C1C",     # 3. deep red (darkest, base)
    "#E53935",     # 4. red (middle, main red)
    "#EF9A9A",     # 5. light red (brightest, pink)
    # --------------------------
    # Second group: two **similar teal/green** entries
    "#2563EB",     # 3. dodger blue 
    "#7DD3FC",     # 5. light sky blue 
    # --------------------------
    # Finish the Viridis ramp (9 colours in total)
    "#45CA77",     # 8. grass green
    "#84E354"      # 9. yellow green
]

# Local module: single source of truth for dataset / model paths (standard library only)
import asset_paths as assets

# Import all models for comparison
from models.echmo import EchMo
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus_Res101, DeepLabV3
from models.fcn8s import FCN8s
from models.tmva_net_sv import TMVANet_SV
from models.vit import ViT

# ==================== Traditional Signal Processing ====================

def load_transmit_signal(dataset_path, dataset_name):
    """Load the transmit signal from a .mat file"""
    tx_path = os.path.join(dataset_path, 'coefs', 's_tx.mat')
    
    if not os.path.exists(tx_path):
        raise FileNotFoundError(f"Transmit signal not found: {tx_path}")
    
    mat_data = loadmat(tx_path)
    
    # Locate the transmit-signal variable
    s_tx = None
    var_name = None
    
    if 's_tx' in mat_data:
        s_tx = mat_data['s_tx']
        var_name = 's_tx'
    elif 'st' in mat_data:
        s_tx = mat_data['st']
        var_name = 'st'
    else:
        for key in mat_data.keys():
            if not key.startswith('__'):
                s_tx = mat_data[key]
                var_name = key
                break
    
    if s_tx is None:
        raise ValueError(f"No valid transmit signal found in {tx_path}")
    
    print(f"[DEBUG] {dataset_name}: Loaded '{var_name}' shape={s_tx.shape}, dtype={s_tx.dtype}")
    
    return s_tx


def load_transmit_signal_from_file(file_path):
    """
    Load the transmit signal from an uploaded file (.mat or .npy)
    """
    if file_path is None:
        raise ValueError("Please upload a transmit signal file")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext == '.mat':
        mat_data = loadmat(file_path)
        s_tx = None
        var_name = None
        
        for key in ['s_tx', 'st', 'transmit', 'tx_signal', 's']:
            if key in mat_data:
                s_tx = mat_data[key]
                var_name = key
                break
        
        if s_tx is None:
            for key in mat_data.keys():
                if not key.startswith('__'):
                    s_tx = mat_data[key]
                    var_name = key
                    break
        
        if s_tx is None:
            raise ValueError(f"No valid transmit signal found in {file_path}")
        
        print(f"[DEBUG] Loaded transmit signal '{var_name}' shape={s_tx.shape}, dtype={s_tx.dtype}")
        return s_tx
        
    elif file_ext == '.npy':
        s_tx = np.load(file_path)
        print(f"[DEBUG] Loaded transmit signal from .npy shape={s_tx.shape}, dtype={s_tx.dtype}")
        return s_tx
        
    else:
        raise ValueError(f"Unsupported file format: {file_ext}. Please use .mat or .npy")


def traditional_lmf_mti_mtd(raw_data, dataset_name=None, dataset_path=None, tx_signal=None):
    """
    Traditional signal processing using actual transmit signal
    """
    
    if raw_data.shape[0] == 2:
        raw_cmplx = raw_data[0] + 1j * raw_data[1]
    else:
        raw_cmplx = raw_data
    
    signal_len = raw_cmplx.shape[1]
    
    # Load the actual transmit signal
    if tx_signal is not None:
        s_tx = tx_signal
        print(f"[DEBUG] Using uploaded transmit signal shape={s_tx.shape}")
    elif dataset_path is not None and dataset_name is not None:
        s_tx = load_transmit_signal(dataset_path, dataset_name)
    else:
        raise ValueError("Either tx_signal or (dataset_path and dataset_name) must be provided")
    
    # Handle the transmit-signal format
    if s_tx.ndim == 2:
        if s_tx.shape[0] == 2:
            mf = s_tx[0] + 1j * s_tx[1]
        else:
            mf = s_tx[:, 0] + 1j * s_tx[:, 1] if s_tx.shape[1] == 2 else s_tx.flatten()
    else:
        mf = s_tx.flatten()
    
    mf = np.asarray(mf, dtype=np.complex64)
    N_mf = len(mf)
    
    print(f"[DEBUG] Transmit signal length: {N_mf}")
    
    # option 1: no pre-flip
    mf_padded = np.zeros(signal_len, dtype=np.complex64)
    mf_padded[:N_mf] = mf
    mf_fft = np.conj(np.fft.fft(mf_padded))
    
    # multiply in the FFT domain
    fft_raw = np.fft.fft(raw_cmplx, axis=1)
    lmf_fft = fft_raw * mf_fft
    lmf_data = np.fft.ifft(lmf_fft, axis=1)
    
    print(f"[DEBUG] LMF output max at: {np.unravel_index(np.argmax(np.abs(lmf_data)), lmf_data.shape)}")
    
    # MTI
    mti_data = np.zeros_like(lmf_data)
    mti_data[1:] = lmf_data[1:] - lmf_data[:-1]
    
    # MTD
    # mtd_data = np.fft.fftshift(np.fft.fft(lmf_data, axis=0), axes=0)
    window = np.hamming(raw_cmplx.shape[0]).reshape(-1, 1)
    mti_windowed = mti_data * window
    # MTD: whether to apply fftshift is declared per dataset in dataset_meta.json
    # (Syn-RED: false, HiL-RED: true), no longer hard-coded by dataset name
    meta = assets.load_dataset_meta(dataset_name) if dataset_name else assets.DEFAULT_META
    if meta.get("mtd_fftshift", True):
        # Standard pipeline: FFT + fftshift
        mtd_data = np.fft.fftshift(np.fft.fft(mti_windowed, axis=0), axes=0)
        print(f"[DEBUG] Dataset={dataset_name}: MTD = FFT + fftshift")
    else:
        # Doppler FFT only, no frequency shift
        mtd_data = np.fft.fft(mti_windowed, axis=0)
        print(f"[DEBUG] Dataset={dataset_name}: MTD = FFT only (no fftshift)")

     # CFAR   
    cfar_mask = ca_cfar_2d(mtd_data, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-4)
    print(f"[DEBUG] CFAR detected {np.sum(cfar_mask)} targets")

    return lmf_data, mti_data, mtd_data, cfar_mask

def ca_cfar_2d(mtd_data, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-4):
    """
    2D CA-CFAR (Cell Averaging CFAR) detection
    
    Parameters:
    -----------
    mtd_data : ndarray
       MTD output data (Doppler x Range)
    guard_cells : tuple
       Guard cell size (guard_doppler, guard_range)
    training_cells : tuple
       Training cell size (train_doppler, train_range)
    pfa : float
       Probability of false alarm
    
    Returns:
    --------
    cfar_mask : ndarray
        Binary detection mask
    """
    # Convert to amplitude
    data = np.abs(mtd_data)
    doppler_bins, range_bins = data.shape
    
    # Compute the CFAR threshold factor
    # N = number of training cells
    train_doppler, train_range = training_cells
    guard_doppler, guard_range = guard_cells
    
    N = (2 * train_doppler + 2 * guard_doppler + 1) * (2 * train_range + 2 * guard_range + 1) - \
        (2 * guard_doppler + 1) * (2 * guard_range + 1)
    
    # Threshold factor (OS-CFAR, or the CA-CFAR approximation)
    alpha = N * (pfa ** (-1/N) - 1)
    
    # Initialise the detection mask
    cfar_mask = np.zeros_like(data, dtype=bool)
    
    # Sliding-window loop
    for d in range(doppler_bins):
        for r in range(range_bins):
            # Compute the training-window bounds
            d_start = max(0, d - train_doppler - guard_doppler)
            d_end = min(doppler_bins, d + train_doppler + guard_doppler + 1)
            r_start = max(0, r - train_range - guard_range)
            r_end = min(range_bins, r + train_range + guard_range + 1)
            
            # Extract the window
            window = data[d_start:d_end, r_start:r_end].copy()
            
            # Remove the guard cells (centre region)
            gd_start = max(0, d - guard_doppler) - d_start
            gd_end = min(doppler_bins, d + guard_doppler + 1) - d_start
            gr_start = max(0, r - guard_range) - r_start
            gr_end = min(range_bins, r + guard_range + 1) - r_start
            
            if gd_end > gd_start and gr_end > gr_start:
                window[gd_start:gd_end, gr_start:gr_end] = 0
            
            # Mean of the training cells
            train_sum = np.sum(window)
            train_count = np.count_nonzero(window)
            
            if train_count > 0:
                threshold = alpha * train_sum / train_count
                
                # Detection
                if data[d, r] > threshold:
                    cfar_mask[d, r] = True
    
    return cfar_mask


def os_cfar_2d(mtd_data, guard_cells=(2, 2), training_cells=(4, 4), rank=0.75, pfa=1e-4):
    """
    2D OS-CFAR (Ordered Statistic CFAR) detection - more robust in multi-target environments
    
    Parameters:
    -----------
    mtd_data : ndarray
        MTD output data
    guard_cells : tuple
        Guard cell size
    training_cells : tuple
        Training cell size
    rank : float
        Rank of the order statistic (0-1), defaults to the 75th percentile
    pfa : float
       Probability of false alarm
    
    Returns:
    --------
    cfar_mask : ndarray
        Binary detection mask
    """
    data = np.abs(mtd_data)
    doppler_bins, range_bins = data.shape
    
    train_doppler, train_range = training_cells
    guard_doppler, guard_range = guard_cells
    
    cfar_mask = np.zeros_like(data, dtype=bool)
    
    for d in range(doppler_bins):
        for r in range(range_bins):
            # Extract the training cells
            d_start = max(0, d - train_doppler - guard_doppler)
            d_end = min(doppler_bins, d + train_doppler + guard_doppler + 1)
            r_start = max(0, r - train_range - guard_range)
            r_end = min(range_bins, r + train_range + guard_range + 1)
            
            window = data[d_start:d_end, r_start:r_end].copy()
            
            # Remove the guard cells and the CUT (cell under test)
            gd_start = max(0, d - guard_doppler) - d_start
            gd_end = min(doppler_bins, d + guard_doppler + 1) - d_start
            gr_start = max(0, r - guard_range) - r_start
            gr_end = min(range_bins, r + guard_range + 1) - r_start
            
            if gd_end > gd_start and gr_end > gr_start:
                window[gd_start:gd_end, gr_start:gr_end] = np.nan
            
            # Collect the training-cell values (excluding nan)
            train_values = window[~np.isnan(window)]
            
            if len(train_values) > 0:
                # Sort and pick the value at the requested rank
                sorted_values = np.sort(train_values)
                k = int(rank * len(sorted_values))
                threshold = sorted_values[min(k, len(sorted_values)-1)]
                
                # OS-CFAR threshold adjustment
                N = len(train_values)
                alpha = N * (pfa ** (-1/N) - 1)
                threshold = alpha * threshold
                
                if data[d, r] > threshold:
                    cfar_mask[d, r] = True
    
    return cfar_mask

# ==================== Visualization ====================
def create_figure_no_colorbar(data, title, cmap='viridis', mask_mode=False, n_classes=2):
    """Create a figure without a colorbar, using the predefined SEG_COLORS"""
    fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS['background'])
    ax.set_facecolor(COLORS['background'])
    
    vmin, vmax = np.percentile(data, (1, 99))

    if mask_mode and n_classes <= 9:
        # Use the predefined SEG_COLORS so colours stay consistent
        if n_classes == 2:
            hex_colors = SEG_COLORS_2
        else:
            hex_colors = SEG_COLORS_9[:n_classes]
        
        # Convert hex to RGB in [0,1]
        colors = [mcolors.to_rgb(c) for c in hex_colors]
        
        cmap_obj = ListedColormap(colors)
        im = ax.imshow(data, cmap=cmap_obj, aspect='auto', vmin=0, vmax=n_classes-1, interpolation='none')
    else:
        # Non-mask data uses viridis
        im = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)
    
    ax.set_title(title, fontsize=10, color=COLORS['secondary'], fontweight='600', pad=8)
    ax.tick_params(axis='both', labelsize=7, colors=COLORS['text_dim'])
    
    for spine in ax.spines.values():
        spine.set_color(COLORS['border'])
        spine.set_linewidth(0.5)
    
    plt.tight_layout()
    return fig


def create_overlay_figure(echmo_mask, cfar_mask, alpha=0.5, n_classes=2, scan_x=None,
                          state_label=None, label_color=None, label_opacity=1.0):
    """Build the EChMo vs CFAR overlay figure - interactive, via Plotly
    
    Args:
        scan_x: scan-line position (0-range_cells), None means do not draw it
        state_label: text shown in the corner ('CFAR detection' / 'EchMo prediction');
                     derived from alpha when not given
        label_color: colour of that label; derived from state_label when not given
        label_opacity: 1.0 right after a switch, dimmed while a result is held
    """
    # Local plotly import (pandas is pre-loaded at module level)
    import plotly.graph_objs as go
    
    # Make sure the mask is a 2D numpy array
    if echmo_mask.ndim == 3:
        echmo_mask = echmo_mask[0] if echmo_mask.shape[0] == 1 else echmo_mask.argmax(axis=0)
    if cfar_mask.ndim == 3:
        cfar_mask = cfar_mask[0]
    
    # Cast to a numpy array
    echmo_mask = np.asarray(echmo_mask, dtype=np.float32)
    cfar_mask = np.asarray(cfar_mask, dtype=bool)
    doppler_cells, range_cells = echmo_mask.shape
    
    fig = go.Figure()
    
    # Colour configuration
    if n_classes == 2:
        hex_colors = SEG_COLORS_2
    else:
        hex_colors = SEG_COLORS_9[:n_classes]
    
    # 1. Solid backdrop plus the CFAR detections as round markers.
    #    The backdrop keeps this panel looking like the other mask panels, and the
    #    detections are circles in screen space so a single-cell detection stays
    #    visible without being inflated into a square blob.
    fig.add_shape(
        type='rect', xref='paper', yref='paper', x0=0, y0=0, x1=1, y1=1,
        fillcolor=SEG_COLORS_2[0], line=dict(width=0), layer='below'
    )
    
    cfar_y, cfar_x = np.where(cfar_mask)
    fig.add_trace(go.Scatter(
        x=cfar_x.tolist(),
        y=cfar_y.tolist(),
        mode='markers',
        marker=dict(
            symbol='circle',
            size=3.5,
            color='#45CA77'   # same colour the legend uses for CFAR targets
        ),
        marker_opacity=float(min(1.0, max(0.0, 1.0 - alpha))),  # visible while CFAR is shown
        hoverinfo='skip',
        name='CFAR Detection'
    ))
    
    # 2. EchMo layer - colour-mapped (on top, with opacity)
    #    zmin/zmax are pinned to the full class range so every class keeps the
    #    colour the legend shows for it. Without them Plotly stretches the
    #    colorscale over the classes present in *this* frame: a HiL-RED frame
    #    holding only classes 1 and 6 painted Target with the background colour.
    z_echmo = echmo_mask.astype(np.float32).tolist()
    colorscale_echmo = [[float(i)/(n_classes-1), hex_colors[i]] for i in range(n_classes)]
    
    fig.add_trace(go.Heatmap(
        z=z_echmo,
        colorscale=colorscale_echmo,
        showscale=False,
        opacity=alpha,
        zmin=0,
        zmax=n_classes - 1,
        hoverinfo='skip',
        name='EchMo Prediction'
    ))
    
    # 3. Scan line effect (shown during the transition)
    if scan_x is not None:
        fig.add_vline(
            x=scan_x,
            line=dict(color='#00ccff', width=3, dash='solid'),
            opacity=0.9
        )
        # Add a glow (wider translucent line)
        fig.add_vline(
            x=scan_x,
            line=dict(color='#00ccff', width=12),
            opacity=0.3
        )
    
    # 4. State label - says which of the two results is currently displayed
    if state_label is None:
        if alpha >= 0.5:
            state_label, label_color = 'EchMo prediction', '#FDE725'
        else:
            state_label, label_color = 'CFAR detection', '#45CA77'
    if label_color is None:
        label_color = '#FDE725' if state_label.startswith('EchMo') else '#45CA77'
    if label_opacity > 0.02:
        # Plain text, no frame - it is meant as a subtle cue, not a badge
        fig.add_annotation(
            text=state_label,
            xref='paper', yref='paper',
            x=0.015, y=0.035, xanchor='left', yanchor='bottom',
            showarrow=False,
            font=dict(size=13, color=label_color, family='Inter, sans-serif'),
            opacity=label_opacity
        )
    
    # Set up the axes
    range_step = max(1, range_cells // 8)
    doppler_step = max(1, doppler_cells // 6)
    range_ticks = list(range(0, range_cells, range_step))
    doppler_ticks = list(range(0, doppler_cells, doppler_step))
    
    fig.update_layout(
        # title removed - using HTML header instead
        xaxis=dict(
            title=dict(text='Range cell', font=dict(size=11, color='#ffffff')),
            tickmode='array',
            tickvals=range_ticks,
            ticktext=[str(i) for i in range_ticks],
            tickfont=dict(size=9, color='#ffffff'),
            # No grid, exactly like every other mask panel: on the flat CFAR
            # backdrop the faint grid lines were the only thing standing out.
            showgrid=False,
            zeroline=False,
            # Pin the range to the data extent. Plotly otherwise pads the range to
            # fit the CFAR markers, which pushed the image ~29 px in from the left
            # edge (an empty band below range cell 0 with no tick labels) whenever
            # detections sat near the range limits -> e.g. HiL-RED deceptive sets.
            range=[-0.5, range_cells - 0.5]
        ),
        yaxis=dict(
            title=dict(text='Doppler cell', font=dict(size=11, color='#ffffff')),
            tickmode='array',
            tickvals=doppler_ticks,
            ticktext=[str(i) for i in doppler_ticks],
            tickfont=dict(size=9, color='#ffffff'),
            showgrid=False,
            zeroline=False,
            # Reversed (row 0 on top), same as the other mask panels
            range=[doppler_cells - 0.5, -0.5]
        ),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        # Same box and margins as every other panel (see arr2plot), so the image
        # area of this one is pixel-identical to the Ground Truth panel next to
        # it - otherwise the same mask is drawn at a different scale.
        margin=dict(l=0, r=0, t=0, b=10),
        height=280,
        # Enable interaction
        dragmode='zoom',
        selectdirection='any'
    )
    
    return fig


# ==================== Overlay animation (EchMo vs CFAR) ====================
# The overlay panel alternates between the two results on a 3 s cycle and the
# timer ticks every 0.05 s, so a phase advances in 60 steps of OVERLAY_STEP.
OVERLAY_STEP = 0.01667          # phase advance per tick
OVERLAY_HOLD_SECONDS = 1.0      # how long a clicked result stays pinned on screen
OVERLAY_SWEEP_START = 0.433     # phase where the sweep towards EchMo starts
OVERLAY_ECHMO_START = 0.500     # phase where the EchMo hold starts
OVERLAY_CFAR_START = 0.933      # phase where the sweep back to CFAR starts
OVERLAY_SWEEP_WIDTH = OVERLAY_ECHMO_START - OVERLAY_SWEEP_START


def overlay_phase_value(alpha, default=0.0):
    """Normalise an animation phase into [0, 1)."""
    try:
        return float(alpha) % 1.0
    except (TypeError, ValueError):
        return default


def overlay_manual_pair(manual):
    """Normalise the pinned-result state into (layer, deadline).

    Layer is -1 while nothing is pinned, 0 for CFAR and 1 for EchMo.
    """
    if isinstance(manual, (tuple, list)) and len(manual) == 2:
        try:
            return int(manual[0]), float(manual[1])
        except (TypeError, ValueError):
            pass
    return -1, 0.0


def overlay_range_cells(echmo_mask):
    """Number of range cells of a (possibly batched) segmentation mask."""
    mask = echmo_mask
    if mask.ndim == 3:
        mask = mask[0] if mask.shape[0] == 1 else mask.argmax(axis=0)
    return int(mask.shape[1])


def overlay_phase_for_layer(layer):
    """Phase that starts the hold of the given layer (0 = CFAR, 1 = EchMo)."""
    return OVERLAY_ECHMO_START if layer else 0.0


def overlay_scan_state(alpha, range_cells):
    """Describe one frame of the animation.

    Phase map of the 3 s cycle (60 steps of 0.05 s):
        0.000-0.433 (0.0-1.3 s): CFAR hold
        0.433-0.500 (1.3-1.5 s): sweep towards EchMo, left to right
        0.500-0.933 (1.5-2.8 s): EchMo hold
        0.933-1.000 (2.8-3.0 s): sweep back to CFAR, right to left

    Returns the EchMo layer opacity, the scan-line position, how far the current
    hold has progressed, which layer the viewer sees and which layer the state
    label announces.
    """
    if alpha < OVERLAY_SWEEP_START:
        return {'layer_alpha': 0.0, 'scan_x': None, 'since_switch': alpha,
                'layer': 0, 'label_layer': 0}
    if alpha < OVERLAY_ECHMO_START:
        t = (alpha - OVERLAY_SWEEP_START) / OVERLAY_SWEEP_WIDTH
        return {'layer_alpha': t, 'scan_x': t * range_cells, 'since_switch': 0.0,
                'layer': 1 if t >= 0.5 else 0, 'label_layer': 1}
    if alpha < OVERLAY_CFAR_START:
        return {'layer_alpha': 1.0, 'scan_x': None,
                'since_switch': alpha - OVERLAY_ECHMO_START,
                'layer': 1, 'label_layer': 1}
    t = (alpha - OVERLAY_CFAR_START) / OVERLAY_SWEEP_WIDTH
    return {'layer_alpha': 1.0 - t, 'scan_x': (1.0 - t) * range_cells,
            'since_switch': 0.0, 'layer': 1 if t < 0.5 else 0, 'label_layer': 0}


def overlay_label_opacity(since_switch):
    """Label brightness: full at the switch, gone 0.26 into a hold phase."""
    if since_switch <= 0.05:
        return 1.0
    if since_switch >= 0.26:
        return 0.0
    return 1.0 - (since_switch - 0.05) / 0.21


def overlay_layer_label(layer):
    """Text and colour of the state label (0 = CFAR, 1 = EchMo)."""
    if layer:
        return 'EchMo prediction', '#FDE725'
    return 'CFAR detection', '#45CA77'


def overlay_frame(alpha, echmo_mask, cfar_mask, n_classes, manual=None, now=None):
    """Build one frame of the animated overlay (timer callback).

    `manual` is the (layer, deadline) pair written by `overlay_flip`. While that
    deadline is in the future the clicked result stays on screen and its phase is
    re-asserted on every tick, so a tick that was still in flight when the user
    clicked can never undo the click.

    Returns (figure, next_phase).
    """
    if echmo_mask is None or cfar_mask is None:
        return go.Figure(), overlay_phase_value(alpha)
    if now is None:
        now = time.monotonic()
    range_cells = overlay_range_cells(echmo_mask)
    layer, deadline = overlay_manual_pair(manual)

    if layer >= 0 and now < deadline:
        phase = overlay_phase_for_layer(layer)
        frame = overlay_scan_state(phase, range_cells)
        label, color = overlay_layer_label(layer)
        fig = create_overlay_figure(
            echmo_mask, cfar_mask, alpha=frame['layer_alpha'], n_classes=n_classes,
            state_label=label, label_color=color, label_opacity=1.0
        )
        return fig, phase

    phase = overlay_phase_value(alpha)
    frame = overlay_scan_state(phase, range_cells)
    label, color = overlay_layer_label(frame['label_layer'])
    fig = create_overlay_figure(
        echmo_mask, cfar_mask, alpha=frame['layer_alpha'], n_classes=n_classes,
        scan_x=frame['scan_x'], state_label=label, label_color=color,
        label_opacity=overlay_label_opacity(frame['since_switch'])
    )
    return fig, (phase + OVERLAY_STEP) % 1.0


def overlay_flip(alpha, now=None):
    """Show the result that is not on the screen right now (panel click).

    Returns (figure, phase, manual). `phase` is the hold phase of the clicked
    result, so the animation simply continues from there, and `manual` pins that
    result for OVERLAY_HOLD_SECONDS. The figure is None while no run has produced
    an overlay yet.
    """
    phase = overlay_phase_value(alpha)
    echmo_mask = _global_results.get('seg_data')
    cfar_mask = _global_results.get('cfar_mask')
    if echmo_mask is None or cfar_mask is None:
        return None, phase, (-1, 0.0)

    seen = overlay_scan_state(phase, overlay_range_cells(echmo_mask))['layer']
    layer = 1 - seen                       # the result that is currently hidden
    if now is None:
        now = time.monotonic()
    label, color = overlay_layer_label(layer)
    fig = create_overlay_figure(
        echmo_mask, cfar_mask, alpha=float(layer),
        n_classes=_global_results.get('seg_n_classes', 2),
        state_label=label, label_color=color, label_opacity=1.0
    )
    return fig, overlay_phase_for_layer(layer), (layer, now + OVERLAY_HOLD_SECONDS)


def fig_to_pil(fig):
    """Convert a matplotlib figure to a PIL Image"""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=fig.get_facecolor(), 
                edgecolor='none', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img = Image.open(buf)
    plt.close(fig)
    return img


def tensor_to_numpy(tensor):
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


# ==================== Model Loading ====================
def load_model(dataset_name, model_type=DEFAULT_MODEL):
    """
    Load model based on dataset name and model type.
    
    Args:
        dataset_name: Dataset folder name, discovered from datasets/ (e.g. 'Syn-RED')
        model_type: Type of model to load (default: 'echmo')
    
    Returns:
        model: Loaded PyTorch model
        cfg: Model configuration dictionary
    """
    runs = assets.discover_runs(dataset_name)
    if not runs:
        raise ValueError(
            f"No model runs found for dataset '{dataset_name}'. "
            f"Available datasets: {assets.available_datasets()}"
        )
    
    if model_type not in runs:
        raise ValueError(
            f"Model type '{model_type}' not found for dataset '{dataset_name}'. "
            f"Available: {sorted(runs)}"
        )
    
    # Model folder/filename discovered from models/<dataset>/*/config.json
    model_folder, model_filename = runs[model_type]
    
    # Construct paths
    model_path = assets.run_weights_path(dataset_name, model_folder, model_filename)
    config_path = assets.run_config_path(dataset_name, model_folder)
    dataset_path = assets.dataset_dir(dataset_name)
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # utf-8-sig tolerates a BOM in the config file
    with open(config_path, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)
    
    class EchMo_CFG:
        def __init__(self):
            self.task_name = 'Segmentation'
    
    model_cfg = EchMo_CFG()
    model_cfg.task_name = cfg.get('task_name', 'Segmentation')
    model_cfg.coef_path = 'coefs'
    model_cfg.dataset_path = dataset_path
    model_cfg.n_pulse = cfg['w_size']
    model_cfg.n_range = cfg['h_size']
    model_cfg.n_classes = cfg['nb_classes']
    model_cfg.n_frames = cfg['nb_input_channels']
    model_cfg.scale = cfg['norm_type']
    model_cfg.lmf_chains = cfg['lmf_chains']
    model_cfg.lmf_channels = cfg['lmf_channels']
    model_cfg.lmf_layers = cfg['lmf_layers']
    model_cfg.lmf_coefs = cfg['lmf_coefs']
    model_cfg.ldc_channels = cfg['ldc_channels']
    model_cfg.ldc_orders = cfg['ldc_orders']
    model_cfg.df_channels = cfg['nb_doppler']
    model_cfg.activation = cfg['activation']
    model_cfg.bias = cfg['bias']
    model_cfg.df_groups = cfg['ldc_orders']

    model = EchMo(model_cfg)
    
    if os.path.exists(model_path):
        state_dict = torch.load(model_path, map_location=DEVICE)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
    
    model.to(DEVICE)
    model.eval()
    return model, cfg


def load_comparison_models(dataset_name):
    """
    Load all 6 models for comparison.
    Returns a dict of {model_name: (model, config)}
    """
    models_dict = {}
    
    model_configs = {
        'echmo': {'class': EchMo, 'extra_args': True},
        'unet': {'class': UNet, 'extra_args': False},
        'deeplabv3plus': {'class': DeepLabV3Plus_Res101, 'extra_args': False},
        'fcn8s': {'class': FCN8s, 'extra_args': False},
        'tmva_net_sv': {'class': TMVANet_SV, 'extra_args': True},
        'vit': {'class': ViT, 'extra_args': False},
    }
    
    for model_name, (folder, filename) in assets.discover_runs(dataset_name).items():
        config_path = assets.run_config_path(dataset_name, folder)
        model_path = assets.run_weights_path(dataset_name, folder, filename)
        
        if not os.path.exists(config_path) or not os.path.exists(model_path):
            print(f"Warning: {model_name} model not found, skipping...")
            continue
            
        with open(config_path, 'r', encoding='utf-8-sig') as f:
            cfg = json.load(f)
        
        # Create model instance
        model_cfg = model_configs[model_name]
        if model_name == 'echmo':
            class EchMo_CFG:
                pass
            m_cfg = EchMo_CFG()
            m_cfg.task_name = cfg.get('task_name', 'Segmentation')
            m_cfg.coef_path = 'coefs'
            m_cfg.dataset_path = assets.dataset_dir(dataset_name)
            m_cfg.n_pulse = cfg['w_size']
            m_cfg.n_range = cfg['h_size']
            m_cfg.n_classes = cfg['nb_classes']
            m_cfg.n_frames = cfg['nb_input_channels']
            m_cfg.scale = cfg['norm_type']
            m_cfg.lmf_chains = cfg['lmf_chains']
            m_cfg.lmf_channels = cfg['lmf_channels']
            m_cfg.lmf_layers = cfg['lmf_layers']
            m_cfg.lmf_coefs = cfg['lmf_coefs']
            m_cfg.ldc_channels = cfg['ldc_channels']
            m_cfg.ldc_orders = cfg['ldc_orders']
            m_cfg.df_channels = cfg['nb_doppler']
            m_cfg.activation = cfg['activation']
            m_cfg.bias = cfg['bias']
            m_cfg.df_groups = cfg['ldc_orders']
            model = model_cfg['class'](m_cfg)
        elif model_name == 'tmva_net_sv':
            model = model_cfg['class'](
                n_classes=cfg['nb_classes'],
                n_frames=1,  # Use single frame
                in_ch=1,
                w=cfg['w_size'],
                h=cfg['h_size']
            )
        elif model_name == 'vit':
            model = model_cfg['class'](
                n_classes=cfg['nb_classes'],
                in_chans=1  # Use single channel
            )
        else:
            # UNet, DeepLabV3Plus, FCN8s - use n_frames=1
            model = model_cfg['class'](
                n_classes=cfg['nb_classes'],
                n_frames=1  # Use single frame
            )
        
        # Load weights
        state_dict = torch.load(model_path, map_location=DEVICE)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict, strict=False)
        model.to(DEVICE)
        model.eval()
        
        models_dict[model_name] = (model, cfg)
    
    return models_dict


def arr2plot(arr, colorscale='Viridis', is_mask=False, title=None, n_classes=2):
    """Create Plotly figure with axis labels - Fixed: Category to Color binding"""
    if arr is None:
        return go.Figure()
    if hasattr(arr, 'size') and arr.size == 0:
        return go.Figure()
    if is_mask and arr.ndim == 3:
        arr = arr[0] if arr.shape[0] == 1 else arr.argmax(axis=0)
    
    # Get array dimensions (Doppler cells, Range cells)
    if arr.ndim >= 2:
        doppler_cells, range_cells = arr.shape
    else:
        doppler_cells, range_cells = arr.shape[0], 1
    
    if is_mask:
        # Fixed: Category number = Color list index
        if n_classes == 2:
            color_list = SEG_COLORS_2
        else:
            color_list = SEG_COLORS_9

        # Build Plotly discrete color scale
        colorscale = []
        for class_idx in range(n_classes):
            hex_color = color_list[min(class_idx, len(color_list)-1)]
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            rgb_str = f"rgb({r},{g},{b})"
            
            colorscale.append([class_idx / n_classes, rgb_str])
            colorscale.append([(class_idx + 1) / n_classes, rgb_str])

        fig = go.Figure(go.Heatmap(
            z=arr, 
            colorscale=colorscale, 
            showscale=False,
            zmin=0,
            zmax=n_classes - 1,
            coloraxis=None
        ))
    else:
        fig = go.Figure(go.Heatmap(z=arr, colorscale='Viridis', showscale=False))
    
    # Axis styling
    range_step = max(1, range_cells // 8)
    doppler_step = max(1, doppler_cells // 6)
    range_ticks = list(range(0, range_cells, range_step))
    doppler_ticks = list(range(0, doppler_cells, doppler_step))
    
    fig.update_layout(
        margin=dict(l=0, r=0, t=20 if title else 0, b=10),
        xaxis_showgrid=False,
        yaxis_showgrid=False,
        xaxis=dict(
            title=dict(text='Range cell', font=dict(size=11, color='#ffffff')),
            showticklabels=True,
            tickmode='array',
            tickvals=range_ticks,
            ticktext=[str(i) for i in range_ticks],
            tickfont=dict(size=9, color='#ffffff'),
            side='bottom'
        ),
        yaxis=dict(
            title=dict(text='Doppler cell', font=dict(size=11, color='#ffffff')),
            showticklabels=True,
            tickmode='array',
            tickvals=doppler_ticks,
            ticktext=[str(i) for i in doppler_ticks],
            tickfont=dict(size=9, color='#ffffff'),
            autorange='reversed'
        ),
        height=280,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    if title:
        fig.update_layout(
            title=dict(
                text=title,
                font=dict(size=12, color='#FFC40E'),
                x=0.5,
                xanchor='center'
            )
        )
    
    return fig


# ==================== 3D View Support ====================
class ChartViewCache:
    """Track the 2D/3D view state of every chart"""
    def __init__(self):
        self._view_states = {}  # {chart_key: '2d'|'3d'}
        self._data_cache = {}   # {chart_key: numpy_array}
    
    def get_view_mode(self, chart_key):
        """Get the view mode of a chart, defaulting to 2d"""
        return self._view_states.get(chart_key, '2d')
    
    def set_view_mode(self, chart_key, mode):
        """Set the view mode of a chart"""
        self._view_states[chart_key] = mode
    
    def cache_data(self, chart_key, data):
        """Cache the raw chart data"""
        if data is not None:
            self._data_cache[chart_key] = data
    
    def get_data(self, chart_key):
        """Get the cached raw data"""
        return self._data_cache.get(chart_key)
    
    def toggle_view(self, chart_key):
        """Toggle the view mode and return the new mode"""
        current = self.get_view_mode(chart_key)
        new_mode = '3d' if current == '2d' else '2d'
        self.set_view_mode(chart_key, new_mode)
        return new_mode
    
    def clear(self):
        """Clear all caches"""
        self._view_states.clear()
        self._data_cache.clear()


# Global view-cache instance
_view_cache = ChartViewCache()


def arr2plot_3d(arr, title=None, colorscale='Viridis'):
    """Create 3D Surface plot for radar data - Fixed for Plotly API"""
    try:
        if arr is None:
            return go.Figure()
        if hasattr(arr, 'size') and arr.size == 0:
            return go.Figure()
        
        # Ensure 2D array
        if arr.ndim > 2:
            arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        
        # Ensure array is at least 2D
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        
        # Get dimensions
        doppler_cells, range_cells = arr.shape
        
        # Create coordinate grids for proper axis labeling
        x_coords = np.arange(range_cells)
        y_coords = np.arange(doppler_cells)
        
        # Create surface plot with explicit coordinates
        fig = go.Figure(data=[go.Surface(
            x=x_coords,
            y=y_coords,
            z=arr,
            colorscale=colorscale,
            showscale=False,
            hovertemplate='Range: %{x}<br>Doppler: %{y}<br>Amplitude: %{z:.4f}<extra></extra>',
            contours={
                'x': {'show': False},
                'y': {'show': False},
                'z': {'show': False}
            }
        )])
        
        # Calculate tick values
        range_step = max(1, range_cells // 6)
        doppler_step = max(1, doppler_cells // 6)
        
        # Fixed: Use correct Plotly API - title=dict(text=..., font=dict(...))
        fig.update_layout(
            margin=dict(l=0, r=0, t=40 if title else 10, b=0),
            scene=dict(
                xaxis=dict(
                    title=dict(text='Range cell', font=dict(size=10, color='#ffffff')),
                    tickfont=dict(size=9, color='#ffffff'),
                    tickmode='linear',
                    dtick=range_step,
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=False
                ),
                yaxis=dict(
                    title=dict(text='Doppler cell', font=dict(size=10, color='#ffffff')),
                    tickfont=dict(size=9, color='#ffffff'),
                    tickmode='linear',
                    dtick=doppler_step,
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=False,
                    autorange='reversed'
                ),
                zaxis=dict(
                    title=dict(text='Amplitude', font=dict(size=10, color='#ffffff')),
                    tickfont=dict(size=9, color='#ffffff'),
                    gridcolor='rgba(255,255,255,0.1)',
                    showbackground=False
                ),
                bgcolor='rgba(26,10,46,0.3)',
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=280,
            showlegend=False
        )
        
        if title:
            fig.update_layout(
                title=dict(
                    text=title,
                    font=dict(size=12, color='#FFC40E'),
                    x=0.5,
                    xanchor='center'
                )
            )
        
        return fig
    except Exception as e:
        print(f"Error in arr2plot_3d: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()


def arr2plot_3d_optimized(arr, title=None, colorscale='Viridis', zmin=None, zmax=None):
    """Create 3D Surface plot with optimized visuals - v2"""
    try:
        if arr is None:
            return go.Figure()
        if hasattr(arr, 'size') and arr.size == 0:
            return go.Figure()
        
        # Ensure 2D array
        if arr.ndim > 2:
            arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
        if arr.ndim < 2:
            arr = arr.reshape(1, -1)
        
        doppler_cells, range_cells = arr.shape
        x_coords = np.arange(range_cells)
        y_coords = np.arange(doppler_cells)
        
        # Z range for consistent scaling
        if zmin is None:
            zmin = np.percentile(arr, 0)
        if zmax is None:
            zmax = np.percentile(arr, 100)
        
        # Surface with lighting
        fig = go.Figure(data=[go.Surface(
            x=x_coords,
            y=y_coords,
            z=arr,
            colorscale=colorscale,
            showscale=False,
            cmin=zmin,
            cmax=zmax,
            hovertemplate='Range:%{x}<br>Doppler:%{y}<br>Amp:%{z:.3f}<extra></extra>',
            lighting=dict(ambient=0.6, diffuse=0.7, specular=0.8, roughness=0.4),
            lightposition=dict(x=100, y=100, z=200)
        )])
        
        range_step = max(1, range_cells // 5)
        doppler_step = max(1, doppler_cells // 5)
        
        # MATLAB default view: azimuth=-37.5°, elevation=30°
        camera_30deg = dict(eye=dict(x=2.0, y=-2.8, z=2.0), up=dict(x=0, y=0, z=1))
        
        fig.update_layout(
            margin=dict(l=0, r=0, t=25 if title else 0, b=0),
            scene=dict(
                xaxis=dict(
                    title=dict(text='Range', font=dict(size=9, color='rgba(255,255,255,0.6)')),
                    tickfont=dict(size=8, color='rgba(255,255,255,0.5)'),
                    tickmode='linear', dtick=range_step,
                    gridcolor='rgba(255,255,255,0.06)', gridwidth=1,
                    showbackground=False, zeroline=False,
                    showline=True, linecolor='rgba(255,255,255,0.12)', linewidth=1
                ),
                yaxis=dict(
                    title=dict(text='Doppler', font=dict(size=9, color='rgba(255,255,255,0.6)')),
                    tickfont=dict(size=8, color='rgba(255,255,255,0.5)'),
                    tickmode='linear', dtick=doppler_step,
                    gridcolor='rgba(255,255,255,0.06)', gridwidth=1,
                    showbackground=False, zeroline=False, autorange='reversed',
                    showline=True, linecolor='rgba(255,255,255,0.12)', linewidth=1
                ),
                zaxis=dict(
                    title=dict(text='Amp', font=dict(size=9, color='rgba(255,255,255,0.6)')),
                    tickfont=dict(size=8, color='rgba(255,255,255,0.5)'),
                    gridcolor='rgba(255,255,255,0.06)', gridwidth=1,
                    showbackground=False, zeroline=False,
                    showline=True, linecolor='rgba(255,255,255,0.12)', linewidth=1,
                    range=[zmin, zmax], nticks=5
                ),
                bgcolor='rgba(0,0,0,0)',
                camera=camera_30deg,
                # Inner aspect ratio matches the canvas, 2.5:1
                aspectmode='manual',
                aspectratio=dict(x=2.5, y=1.0, z=0.5)
            ),
            # paper_bgcolor='#1a0a2e',
            # plot_bgcolor='#1a0a2e',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            autosize=False,
            width=500,
            height=500
        )
        
        if title:
            fig.update_layout(title=dict(text=title, font=dict(size=11, color='#FFC40E'), x=0.5))
        
        return fig
    except Exception as e:
        print(f"Error in arr2plot_3d: {e}")
        return go.Figure()


def get_plot_for_chart(chart_key, data, title=None, is_mask=False, n_classes=2):
    """Build the chart for the current view mode"""
    try:
        if data is None:
            return go.Figure()
        
        # Cache the raw data
        _view_cache.cache_data(chart_key, data)
        
        # Get the current view mode
        view_mode = _view_cache.get_view_mode(chart_key)
        
        if view_mode == '3d' and not is_mask:
            # 3D mode is only for non-mask data - use the optimised variant
            return arr2plot_3d_optimized(data, title=title)
        else:
            # 2D mode (default)
            return arr2plot(data, is_mask=is_mask, title=title, n_classes=n_classes)
    except Exception as e:
        print(f"Error in get_plot_for_chart for {chart_key}: {e}")
        import traceback
        traceback.print_exc()
        return go.Figure()


def toggle_chart_view(chart_key, current_data, title=None, is_mask=False, n_classes=2):
    """Toggle the chart view mode and return the new figure"""
    new_mode = _view_cache.toggle_view(chart_key)
    
    # Use the passed-in data, otherwise fall back to the cache
    data = current_data if current_data is not None else _view_cache.get_data(chart_key)
    
    if data is None:
        return go.Figure(), new_mode
    
    if new_mode == '3d' and not is_mask:
        fig = arr2plot_3d(data, title=title)
    else:
        fig = arr2plot(data, is_mask=is_mask, title=title, n_classes=n_classes)
    
    return fig, new_mode


def run_comparison_inference(dataset_name, seq_name, frame_name):
    """
    Run inference with all models for comparison.
    Returns dict with results from all models.
    """
    if not dataset_name or not seq_name or not frame_name:
        return None, "Please select data source, sequence, and frame."
    
    # Load echo data
    echo_data = load_echo_data(dataset_name, seq_name, frame_name)
    if echo_data is None:
        return None, "Echo data not found"
    
    # Load annotation
    annotation = load_annotation(dataset_name, seq_name, frame_name)
    
    # Prepare input for EchMo (complex data)
    if echo_data.shape[0] == 2:
        real_part = echo_data[0]
        imag_part = echo_data[1]
        echo_cmplx = real_part + 1j * imag_part
    else:
        echo_cmplx = echo_data
    
    x_complex = torch.from_numpy(echo_cmplx).unsqueeze(0).unsqueeze(0).to(DEVICE)
    x_complex = torch.complex(x_complex.real.float(), x_complex.imag.float())
    
    # Prepare MTD amplitude input for other models.
    # The source folder is dataset-specific and declared in dataset_meta.json
    # (Syn-RED -> range_doppler_mtdData_amp, HiL-RED -> range_doppler_MTD)
    mtd_path = assets.mtd_amp_path(dataset_name, seq_name, frame_name)
    if os.path.exists(mtd_path):
        mtd_amp = np.abs(np.load(mtd_path))
    else:
        # Fallback: compute from echo data
        mtd_amp = np.abs(echo_cmplx)
    
    # Ensure mtd_amp is 2D
    if mtd_amp.ndim == 3:
        mtd_amp = mtd_amp[0]
    
    # Load all models
    models_dict = load_comparison_models(dataset_name)
    
    results = {}
    n_classes = 2
    
    with torch.no_grad():
        for model_name, (model, cfg) in models_dict.items():
            try:
                n_classes = cfg.get('nb_classes', 2)
                
                if model_name == 'echmo':
                    # EchMo uses complex input
                    output, _, _, _ = model(x_complex)
                elif model_name == 'tmva_net_sv':
                    # TMVA-Net expects 5D input: (batch, n_frames, in_ch, h, w)
                    # First expand_dims adds temporal dimension, second adds in_ch dimension
                    # Then unsqueeze(0) adds batch dimension
                    x_tmva = np.expand_dims(mtd_amp, axis=0)  # (1, h, w) - add temporal dim (n_frames=1)
                    x_tmva = np.expand_dims(x_tmva, axis=0)  # (1, 1, h, w) - add in_ch dim
                    x_tmva = torch.from_numpy(x_tmva).unsqueeze(0).to(DEVICE).float()  # (1, 1, 1, h, w) - add batch dim
                    output = model(x_tmva)
                else:
                    # Other models (U-Net, DeepLabv3+, FCN-8s, ViT) use single frame
                    # Input shape: (batch, 1, h, w)
                    x_other = torch.from_numpy(mtd_amp).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
                    output = model(x_other)
                
                pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
                results[model_name] = pred_mask
            except Exception as e:
                print(f"Error running {model_name}: {e}")
                import traceback
                traceback.print_exc()
                results[model_name] = None
    
    # Raw echo amplitude for display
    echo_amp = np.abs(echo_cmplx)
    if echo_amp.ndim == 3:
        echo_amp = echo_amp[0]
    
    # Ground truth (accepts one-hot and label maps, see annotation_to_label_map)
    gt_mask = annotation_to_label_map(annotation)
    
    return {
        'raw_echo': echo_amp,
        'mtd': mtd_amp,
        'gt_mask': gt_mask,
        'predictions': results,
        'n_classes': n_classes
    }, f"✓ Compared {len([r for r in results.values() if r is not None])} models on {dataset_name}/{seq_name}/{frame_name}"


# ==================== Data Loading ====================
# All paths are resolved by asset_paths (the folder layout is defined in one place)
def load_dataset_json(dataset_name):
    return assets.load_dataset_index(dataset_name)

def load_echo_data(dataset_name, seq_name, frame_name):
    echo_path = assets.echo_path(dataset_name, seq_name, frame_name)
    if os.path.exists(echo_path):
        return np.load(echo_path)
    return None

def load_annotation(dataset_name, seq_name, frame_name):
    annot_path = assets.annotation_path(dataset_name, seq_name, frame_name)
    if os.path.exists(annot_path):
        return np.load(annot_path)
    return None

def annotation_to_label_map(annotation):
    """
    Normalise an annotation into a 2D label map (Doppler, Range): one class id per pixel.

    Both on-disk layouts are supported, so old and new data just work:
      - one-hot   (C, Doppler, Range)  : all channels sum to 1 per pixel -> argmax
                  (1, Doppler, Range)  : single-channel mask -> drop the leading dim
                  (1, C, Doppler, Range): one extra leading dim -> drop it and recurse
      - label map (Doppler, Range)     : already a label map -> returned as-is

    Syn-RED stores one-hot (uint8, C=2); HiL-RED ships
    int8 label maps (converted from the original 9-channel float64 one-hot).
    """
    if annotation is None:
        return None
    arr = np.asarray(annotation)
    if arr.ndim < 2:
        return arr.reshape(1, -1)
    if arr.ndim == 2:
        return arr                      # already a label map
    if arr.ndim == 3 and arr.shape[0] == 1:
        return arr[0]                   # single-channel mask
    if arr.ndim == 3:
        return arr.argmax(axis=0)       # one-hot (C, D, R)
    return annotation_to_label_map(arr[0])   # drop the extra leading dim

def load_echo_data_from_file(file_path):
    """
    Load the echo signal from an uploaded file (.npy)
    """
    if file_path is None:
        raise ValueError("Please upload an echo signal file")
    
    file_ext = os.path.splitext(file_path)[1].lower()
    
    if file_ext != '.npy':
        raise ValueError(f"Unsupported echo file format: {file_ext}. Please use .npy format")
    
    echo_data = np.load(file_path)
    print(f"[DEBUG] Loaded echo signal from .npy shape={echo_data.shape}, dtype={echo_data.dtype}")
    
    if echo_data.ndim not in [2, 3]:
        raise ValueError(f"Invalid echo data shape: {echo_data.shape}. Expected 2D or 3D array")
    
    return echo_data


# ==================== Global Variables ====================
model_cache = {}
_global_results = {}


# ==================== Processing Functions ====================
def process_with_model(model, echo_data):
    if echo_data.shape[0] == 2:
        real_part = echo_data[0]
        imag_part = echo_data[1]
        echo_cmplx = real_part + 1j * imag_part
    else:
        echo_cmplx = echo_data
    
    x = torch.from_numpy(echo_cmplx).unsqueeze(0).unsqueeze(0).to(DEVICE)
    x = torch.complex(x.real.float(), x.imag.float())
    
    with torch.no_grad():
        output, lmf_out, mldc_out, gdf_out = model(x)
    
    return output, lmf_out, mldc_out, gdf_out


def run_inference(dataset_name, seq_name, frame_name):
    global _global_results
    if not dataset_name or not seq_name or not frame_name:
        msg = "Please select data source, sequence, and frame."
        print("[run_inference]", msg)
        return [None] * 10 + [msg, True]  # the last value flags whether ground truth exists
    
    if not assets.discover_runs(dataset_name):
        msg = f"Configuration not found for {dataset_name}"
        print("[run_inference]", msg)
        return [None] * 10 + [msg, True]
    
    dataset_path = assets.dataset_dir(dataset_name)
    if not os.path.exists(dataset_path):
        msg = f"Dataset path not found: {dataset_path}"
        print("[run_inference]", msg)
        return [None] * 10 + [msg, True]
    
    model_key = dataset_name
    if model_key not in model_cache:
        try:
            model, config = load_model(dataset_name)
            model_cache[model_key] = (model, config)
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]
    
    model, config = model_cache[model_key]
    n_classes = config.get('nb_classes', 2)
    
    echo_data = load_echo_data(dataset_name, seq_name, frame_name)
    if echo_data is None:
        msg = "Echo data not found"
        print("[run_inference]", msg)
        return [None] * 10 + [msg, True]
    
    annotation = load_annotation(dataset_name, seq_name, frame_name)
    
    try:
        model_output, lmf_out, mldc_out, gdf_out = process_with_model(model, echo_data)
    except Exception as e:
        msg = f"Error during model inference: {str(e)}"
        print("[run_inference]", msg)
        return [None] * 10 + [msg, True]
    
    # Traditional signal processing
    try:
        trad_lmf, trad_mti, trad_mtd, trad_cfar = traditional_lmf_mti_mtd(
            echo_data, 
            dataset_name=dataset_name,
            dataset_path=dataset_path
        )
    except Exception as e:
        msg = f"Error during traditional processing: {str(e)}"
        print("[run_inference]", msg)
        traceback.print_exc()
        return [None] * 10 + [msg, True]
    
    # Raw Echo Input
    if echo_data.shape[0] == 2:
        echo_amp = np.abs(echo_data[0] + 1j * echo_data[1])
    else:
        echo_amp = np.abs(echo_data)
    if echo_amp.ndim == 3:
        echo_amp = echo_amp[0]

    # EchMo LMF
    lmf_batch0 = lmf_out[0]
    lmf_amp = tensor_to_numpy(torch.abs(lmf_batch0).mean(dim=0))

    # Traditional LMF
    trad_lmf_amp = np.abs(trad_lmf)
    if trad_lmf_amp.ndim == 3:
        trad_lmf_amp = trad_lmf_amp[0]

    # EchMo MLDC
    mldc_batch0 = mldc_out[0]
    mldc_amp = tensor_to_numpy(torch.abs(mldc_batch0).mean(dim=0))

    # Traditional MTI
    trad_mti_amp = np.abs(trad_mti)
    if trad_mti_amp.ndim == 3:
        trad_mti_amp = trad_mti_amp[0]

    # EchMo GDF
    gdf_batch0 = gdf_out[0]
    gdf_amp = tensor_to_numpy(torch.abs(gdf_batch0).mean(dim=0))

    # Traditional MTD
    trad_mtd_amp = np.abs(trad_mtd)
    if trad_mtd_amp.ndim == 3:
        trad_mtd_amp = trad_mtd_amp[0]

    # Segmentation Prediction
    pred_mask = tensor_to_numpy(torch.argmax(model_output, dim=1)[0])

    # Store for curtain comparison
    _global_results['seg_data'] = pred_mask
    _global_results['seg_n_classes'] = n_classes
    _global_results['trad_rd_data'] = trad_mtd_amp
    _global_results['cfar_mask'] = trad_cfar

    # Ground Truth (accepts one-hot and label maps, see annotation_to_label_map)
    gt_mask = annotation_to_label_map(annotation)
    has_gt = annotation is not None

    status_msg = f"✓ Processed: {dataset_name} / {seq_name} / {frame_name}"
    if not has_gt:
        status_msg += " | Ground truth not found"

    return [
        echo_amp,         # 0: raw_input
        gt_mask,          # 1: ground_truth (after the swap: was pred_mask)
        pred_mask,        # 2: pred_mask (after the swap: was gt_mask)
        lmf_amp,          # 3: echmo_lmf
        trad_lmf_amp,     # 4: trad_lmf
        mldc_amp,         # 5: echmo_mldc
        trad_mti_amp,     # 6: trad_mti
        gdf_amp,          # 7: echmo_gdf
        trad_mtd_amp,     # 8: trad_mtd
        trad_cfar,        # 9: cfar_mask (added)
        status_msg,       # 10: status
        has_gt            # 11: has_ground_truth flag
    ]


# NOTE: the "Custom Data" entry has been removed from the UI, so this function
# is currently unreferenced.  It is kept so the feature can be re-enabled by
# adding assets.CUSTOM_DATASET back to `all_data_sources` in create_interface().
def run_inference_custom(tx_signal_file, echo_file, model_name):
    """
    Handle the files uploaded by the user (custom mode)
    """
    global _global_results
    
    if tx_signal_file is None or echo_file is None:
        msg = "Please upload both transmit signal and echo signal files."
        print("[run_inference_custom]", msg)
        return [None] * 10 + [msg, False]  # custom data has no ground truth
    
    if not model_name:
        msg = "Please select a model."
        print("[run_inference_custom]", msg)
        return [None] * 10 + [msg, False]
    
    # Load the transmit signal
    try:
        tx_signal = load_transmit_signal_from_file(tx_signal_file)
    except Exception as e:
        msg = f"Error loading transmit signal: {str(e)}"
        print("[run_inference_custom]", msg)
        return [None] * 10 + [msg, False]
    
    # Load the echo signal
    try:
        echo_data = load_echo_data_from_file(echo_file)
    except Exception as e:
        msg = f"Error loading echo signal: {str(e)}"
        print("[run_inference_custom]", msg)
        return [None] * 10 + [msg, False]
    
    # Check model config (note: in custom mode model_name actually means "which dataset's pretrained model")
    if not assets.discover_runs(model_name):
        msg = f"Configuration not found for {model_name}"
        print("[run_inference_custom]", msg)
        return [None] * 10 + [msg, False]
    
    # Load the model
    model_key = model_name
    if model_key not in model_cache:
        try:
            model, config = load_model(model_name)
            model_cache[model_key] = (model, config)
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            print("[run_inference_custom]", msg)
            return [None] * 10 + [msg, False]
    
    model, config = model_cache[model_key]
    n_classes = config.get('nb_classes', 2)
    
    # Model inference
    try:
        model_output, lmf_out, mldc_out, gdf_out = process_with_model(model, echo_data)
    except Exception as e:
        msg = f"Error during model inference: {str(e)}"
        print("[run_inference_custom]", msg)
        traceback.print_exc()
        return [None] * 10 + [msg, False]
    
   # Traditional signal processing - fix: capture trad_cfar as the 4th return value
    try:
        trad_lmf, trad_mti, trad_mtd, trad_cfar = traditional_lmf_mti_mtd(
            echo_data, 
            tx_signal=tx_signal
        )
    except Exception as e:
        msg = f"Error during traditional processing: {str(e)}"
        print("[run_inference_custom]", msg)
        traceback.print_exc()
        return [None] * 10 + [msg, False]
    
    # Raw Echo Input
    if echo_data.shape[0] == 2:
        echo_amp = np.abs(echo_data[0] + 1j * echo_data[1])
    else:
        echo_amp = np.abs(echo_data)
    if echo_amp.ndim == 3:
        echo_amp = echo_amp[0]

    # EchMo LMF
    lmf_batch0 = lmf_out[0]
    lmf_amp = tensor_to_numpy(torch.abs(lmf_batch0).mean(dim=0))

    # Traditional LMF
    trad_lmf_amp = np.abs(trad_lmf)
    if trad_lmf_amp.ndim == 3:
        trad_lmf_amp = trad_lmf_amp[0]

    # EchMo MLDC
    mldc_batch0 = mldc_out[0]
    mldc_amp = tensor_to_numpy(torch.abs(mldc_batch0).mean(dim=0))

    # Traditional MTI
    trad_mti_amp = np.abs(trad_mti)
    if trad_mti_amp.ndim == 3:
        trad_mti_amp = trad_mti_amp[0]

    # EchMo GDF
    gdf_batch0 = gdf_out[0]
    gdf_amp = tensor_to_numpy(torch.abs(gdf_batch0).mean(dim=0))

    # Traditional MTD
    trad_mtd_amp = np.abs(trad_mtd)
    if trad_mtd_amp.ndim == 3:
        trad_mtd_amp = trad_mtd_amp[0]

    # Segmentation Prediction
    pred_mask = tensor_to_numpy(torch.argmax(model_output, dim=1)[0])

    # Store for curtain comparison
    _global_results['seg_data'] = pred_mask
    _global_results['seg_n_classes'] = n_classes
    _global_results['trad_rd_data'] = trad_mtd_amp
    _global_results['cfar_mask'] = trad_cfar

    status_msg = f"✓ Processed custom data using model trained on {model_name} | No ground truth available"


    # NOTE: in Custom mode ground_truth is None; keep the same return order as run_inference
    return [
        echo_amp,         # 0: raw_input
        None,             # 1: ground_truth (None for custom data, position after the swap)
        pred_mask,        # 2: pred_mask (position after the swap)
        lmf_amp,          # 3: echmo_lmf
        trad_lmf_amp,     # 4: trad_lmf
        mldc_amp,         # 5: echmo_mldc
        trad_mti_amp,     # 6: trad_mti
        gdf_amp,          # 7: echmo_gdf
        trad_mtd_amp,     # 8: trad_mtd
        trad_cfar,        # 9: cfar_mask (added)
        status_msg,       # 10: status
        False             # 11: has_ground_truth = False
    ]


def get_available_datasets():
    """Datasets are discovered from disk (datasets/*/light_dataset_frame_oriented.json)"""
    return assets.available_datasets()

def get_sequences(dataset_name):
    return assets.get_sequences(dataset_name)

def get_frames(dataset_name, seq_name):
    return assets.get_frames(dataset_name, seq_name)


# ==================== Custom CSS ====================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Exo+2:wght@400;500;600;700&display=swap');

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #1a0a2e 0%, #2a004f 50%, #0d0618 100%) !important;
    color: #e2e8f0 !important;
    min-height: 100vh !important;
}

/* ===== Startup loading animation ===== */
#loading-screen {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: #1a0a2e;
    display: flex !important;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    z-index: 2147483647 !important;
    transition: opacity 0.5s ease-out;
}

#loading-screen.hidden {
    opacity: 0;
    pointer-events: none;
}

/* Main title - on top */
.loading-title {
    color: #FFC40E;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 16px;
    text-align: center;
    letter-spacing: 2px;
    font-family: 'Exo 2', sans-serif;
    text-shadow: 0 0 20px rgba(255, 196, 14, 0.5), 0 0 40px rgba(255, 196, 14, 0.3);
    position: relative;
    z-index: 2147483647;
}

/* Subtitle - on top */
.loading-subtitle {
    color: #c4b5fd;
    font-size: 1rem;
    font-weight: 400;
    margin-bottom: 80px;
    text-align: center;
    font-family: 'Exo 2', sans-serif;
    letter-spacing: 4px;
    position: relative;
    z-index: 2147483647;
}

/* Progress bar area - on top */
.loading-progress-area {
    width: 400px;
    position: relative;
    z-index: 2147483647;
}

/* Progress bar container - white border */
.progress-container {
    width: 100%;
    height: 8px;
    background: rgba(0, 0, 0, 0.4);
    border: 2px solid rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    padding: 2px;
    box-shadow: 
        0 0 20px rgba(124, 58, 237, 0.3),
        inset 0 2px 4px rgba(0, 0, 0, 0.5);
    position: relative;
    z-index: 2147483647;
}

/* Progress bar fill - gold */
.progress-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, 
        #f59e0b 0%, 
        #FFC40E 30%, 
        #fff5a8 50%, 
        #FFC40E 70%, 
        #f59e0b 100%
    );
    background-size: 200% 100%;
    border-radius: 6px;
    box-shadow: 
        0 0 10px rgba(255, 196, 14, 0.6),
        0 0 20px rgba(255, 196, 14, 0.3);
    position: relative;
    z-index: 2147483647;
    overflow: hidden;
    transition: width 0.1s linear;
}

/* Progress bar glow pulse */
@keyframes barGlow {
    0%, 100% { 
        box-shadow: 0 0 10px rgba(255, 196, 14, 0.6), 0 0 20px rgba(255, 196, 14, 0.3);
    }
    50% { 
        box-shadow: 0 0 20px rgba(255, 196, 14, 0.9), 0 0 40px rgba(255, 196, 14, 0.5);
    }
}

/* Inner shimmer of the progress bar */
.progress-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: -50%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, 
        transparent 0%, 
        rgba(255, 255, 255, 0.6) 50%, 
        transparent 100%
    );
    animation: shineMove 1.5s ease-in-out infinite;
}

@keyframes shineMove {
    0% { left: -50%; }
    100% { left: 150%; }
}

@keyframes barShimmer {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
}

/* Status text */
.loading-status-row {
    display: flex;
    justify-content: center;
    align-items: center;
    margin-top: 20px;
    gap: 12px;
    position: relative;
    z-index: 2147483647;
}

.loading-status {
    color: #c4b5fd;
    font-size: 0.9rem;
    font-family: 'Exo 2', monospace;
    letter-spacing: 1px;
    text-transform: uppercase;
}

.loading-percent {
    color: #FFC40E;
    font-size: 0.9rem;
    font-family: 'Exo 2', monospace;
    font-weight: 700;
    min-width: 40px;
    text-align: right;
}

/* ===== Header with Logo ===== */
.header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 30px 15px;
    position: relative;
}

.header-logo {
    width: 150px;
    height: 150px;
    object-fit: contain;
    filter: drop-shadow(0 0 20px rgba(255, 196, 14, 0.5));
    animation: logoFadeIn 1.2s ease-out, logoGlow 3s ease-in-out 1.2s infinite;
}

@keyframes logoFadeIn {
    0% {
        opacity: 0;
        transform: scale(0.8) rotate(-10deg);
    }
    100% {
        opacity: 1;
        transform: scale(1) rotate(0deg);
    }
}

@keyframes logoGlow {
    0%, 100% { 
        filter: drop-shadow(0 0 20px rgba(255, 196, 14, 0.5));
    }
    50% { 
        filter: drop-shadow(0 0 35px rgba(255, 196, 14, 0.8));
    }
}

.header-title-wrapper {
    text-align: center;
    flex: 1;
}

.header-title {
    color: #FFC40E;
    font-size: 1.8rem;
    font-weight: 700;
    letter-spacing: 2px;
    line-height: 1.2;
    text-align: center;
    font-family: 'Exo 2', sans-serif !important;
    /* Gradient text effect */
    background: linear-gradient(
        90deg,
        #FFC40E 0%,
        #fff5a8 25%,
        #FFC40E 50%,
        #fff5a8 75%,
        #FFC40E 100%
    );
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite, fadeInDown 1s ease-out;
}

/* Text shimmer animation */
@keyframes shimmer {
    0% { background-position: 200% center; }
    100% { background-position: -200% center; }
}

/* Fade-in falling animation */
@keyframes fadeInDown {
    0% {
        opacity: 0;
        transform: translateY(-20px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Subtitle animation */
.header-subtitle {
    animation: fadeInUp 1s ease-out 0.3s both;
}

@keyframes fadeInUp {
    0% {
        opacity: 0;
        transform: translateY(10px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

.header-subtitle {
    color: #c4b5fd;
    font-size: 0.95rem;
    font-weight: 400;
    margin-top: 6px;
    text-align: center;
    letter-spacing: 1px;
    animation: fadeInUp 1s ease-out 0.5s both, pulse 3s ease-in-out 1.5s infinite;
}

/* Fade-in rising animation */
@keyframes fadeInUp {
    0% {
        opacity: 0;
        transform: translateY(15px);
    }
    100% {
        opacity: 1;
        transform: translateY(0);
    }
}

/* Subtle breathing effect */
@keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}

@media (max-width: 768px) {
    .header-container {
        flex-direction: column;
        padding: 15px 20px;
    }
    .header-logo {
        width: 100px;
        height: 100px;
        margin-bottom: 10px;
    }
    .header-title {
        font-size: 1.4rem;
    }
}

/* ===== Tab button style - tech look ===== */
.tab-nav {
    background: transparent !important;
    border-bottom: 1px solid rgba(124, 58, 237, 0.3) !important;
    padding: 0 20px !important;
}

.tab-nav button,
[role="tab"],
.tabs .tab,
.gr-tab,
.gr-tabs button,
button[role="tab"] {
    color: #c4b5fd !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: none !important;
    padding: 14px 24px !important;
    transition: all 0.3s ease !important;
    border-radius: 8px 8px 0 0 !important;
    font-size: 0.95rem !important;
}

.tab-nav button:hover,
[role="tab"]:hover,
.tabs .tab:hover,
.gr-tab:hover {
    color: #f8fafc !important;
    background: rgba(124, 58, 237, 0.15) !important;
}

.tab-nav button.selected,
[role="tab"][aria-selected="true"],
.tabs .tab.selected,
.gr-tab[aria-selected="true"],
.gr-tabs button[data-bp="selected"],
button[role="tab"][aria-selected="true"] {
    color: #FFC40E !important;
    font-weight: 600 !important;
    background: linear-gradient(180deg, rgba(255, 196, 14, 0.1) 0%, transparent 100%) !important;
    border-bottom: 2px solid #FFC40E !important;
    box-shadow: 0 -2px 12px rgba(255, 196, 14, 0.15) !important;
}

/* Deeper nested selectors */
.gradio-container .tab-nav *,
.gradio-container [class*="tab"] button {
    color: #c4b5fd !important;
}

.gradio-container .tab-nav .selected,
.gradio-container [class*="tab"] [class*="selected"],
.gradio-container [class*="tab"] [aria-selected="true"] {
    color: #FFC40E !important;
}

/* Header */
.header-title {
    text-align: center;
    padding: 24px 0 20px;
    color: #FFC40E;
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}

/* ===== Control row - card style ===== */
.control-row {
    display: flex !important;
    align-items: flex-end !important;
    background: rgba(45, 27, 78, 0.6) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    padding: 16px 20px !important;
    margin: 0 0 16px 0 !important;
    min-height: unset !important;
    gap: 16px !important;
    overflow: visible !important;
    /* Remove backdrop-filter to avoid positioning issues */
}

.control-item {
    flex: 1;
    min-width: 0 !important;
    position: relative !important;
    overflow: visible !important;
}

.control-item label {
    display: block;
    font-size: 0.7rem;
    color: #a78bfa;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 500;
}

/* ===== Dropdown - unify inner and outer layers ===== */

/* Outer container - purple border, the overall frame */
.dark-dropdown {
    width: 380px !important;
    height: 36px !important;
    background: #1a0a2e !important;
    border: 1px solid #7c3aed !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    padding: 0 !important;
    margin-left: 10px !important;
}

/* Inner container - fills the outer one */
.dark-dropdown > div,
.dark-dropdown > div > div {
    width: 100% !important;
    height: 100% !important;
    min-height: 100% !important;
    min-width: 100% !important;
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
    box-sizing: border-box !important;
}

/* Input area - fill the box and drop all borders */
.dark-dropdown input,
.dark-dropdown [contenteditable],
.dark-dropdown [role="combobox"] {
    width: 100% !important;
    height: 100% !important;
    color: #ffffff !important;
    background: transparent !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    padding: 0 8px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    # min-width: 100% !important;
    # max-width: 100% !important;
    min-width: 450px !important;     /* Min width */
    max-width: 500px !important;     /* Max width */
}

.dark-dropdown:hover {
    border-color: #FFC40E !important;
    box-shadow: 0 0 8px rgba(255, 196, 14, 0.4) !important;
}

/* Inner listbox (input area) - transparent border */
.dark-dropdown [role="listbox"] {
    width: 100% !important;
    height: 100% !important;
    background: #1a0a2e !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Dropdown list (in the portal) - natural size, no border */
[role="listbox"] {
    background: rgba(26, 10, 46, 0.05) !important;  /* Translucent dark */
    backdrop-filter: blur(10px); 
    border: none !important;
    border-radius: 6px !important;
    width: auto !important;
    height: auto !important;
}

/* Hide all processing/loading progress bars */
[class*="loading"],
[class*="progress"],
[class*="spinner"],
[class*="processing"],
[class*="generating"],
[data-testid*="loader"],
[data-testid*="progress"],
svg[class*="animate"],
[role="status"] {
    display: none !important;
    visibility: hidden !important;
}

/* List item */
[role="listbox"] [role="option"] {
    color: #ffffff !important;
}

/* List item hover */
[role="listbox"] [role="option"]:hover {
    background: #2d1b4e !important;
    color: #FFC40E !important;
}

[role="listbox"] [aria-selected="true"] {
    background: rgba(26, 10, 46, 0.55) !important;  /* Translucent background */
    color: #FFC40E !important;  /* gold text */
}

/* ========== Run button - gold ========== */
.run-btn {
    background: linear-gradient(135deg, #FFC40E 0%, #f59e0b 100%) !important;
    border: none !important;
    color: #1a0a2e !important;
    font-weight: 700 !important;
    padding: 10px 32px !important;
    border-radius: 10px !important;
    height: 44px !important;
    min-width: 110px !important;
    box-shadow: 0 4px 15px rgba(255, 196, 14, 0.4), 0 0 20px rgba(255, 196, 14, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    align-self: flex-end !important;
    margin-bottom: 0 !important;
    transition: all 0.3s ease !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

.run-btn:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 25px rgba(255, 196, 14, 0.6), 0 0 30px rgba(255, 196, 14, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    filter: brightness(1.15);
}

.run-btn:active {
    transform: translateY(0);
}

/* ========== Other styles ========== */
.col-header {
    font-size: 1rem !important;
    color: #FFC40E !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    text-align: center;
    padding: 12px 16px !important;
    background: linear-gradient(135deg, rgba(45, 27, 78, 0.6) 0%, rgba(26, 10, 46, 0.6) 100%) !important;
    border-radius: 10px !important;
    margin-bottom: 12px !important;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    line-height: 1.2 !important;
    backdrop-filter: blur(10px) !important;
}

.plot-container {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}

.status-bar {
    margin: 20px !important;
    padding: 14px 20px !important;
    background: linear-gradient(135deg, rgba(45, 27, 78, 0.6) 0%, rgba(26, 10, 46, 0.6) 100%) !important;
    color: #f8fafc !important;
    border: 1px solid rgba(124, 58, 237, 0.2) !important;
    border-radius: 10px !important;
    font-size: 0.9rem !important;
    font-family: monospace !important;
    font-weight: 500 !important;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(10px) !important;
}

.status-bar > div,
.status-bar > div > div,
.status-bar textarea,
.status-bar input {
    background: transparent !important;
    color: #ffffff !important;
    border: none !important;
    box-shadow: none !important;
}

.model-selector {
    background: #181826 !important;
    border: 1px solid #232345 !important;
    border-radius: 4px !important;
}

.dynamic-label {
    font-size: 0.7rem;
    color: #ffffff;
    margin-bottom: 6px;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    display: block;
    transition: all 0.3s ease;
}

@media (max-width: 1100px) {
    .control-row {
        flex-wrap: wrap !important;
    }
    
    .control-item {
        min-width: 200px !important;
    }
}

/* ===== 3D View System - Floating Toggle Buttons ===== */

/* Chart container with relative positioning for absolute toggle */
.chart-with-toggle {
    position: relative !important;
    margin-bottom: 8px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    overflow: visible !important;
    max-width: 100% !important;
    height: 280px !important;  /* Fixed height keeps the rows aligned */
    min-height: 280px !important;
}

/* Centre the Gradio Plot component - layered selectors to make sure they match */
.chart-with-toggle > div,
.chart-with-toggle > div > div,
.chart-with-toggle .plot-container,
.chart-with-toggle [class*="plot"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    margin: 0 auto !important;
    width: 100% !important;
}

/* Centre the Plotly chart - scale the 3D view to fit */
.plot-container .js-plotly-plot,
.plot-container .plotly,
.plot-container > div {
    margin: 0 auto !important;
    max-width: 100% !important;
}

/* Scale the 3D chart to the container - keep the aspect ratio */
.plot-container svg,
.plot-container .main-svg {
    max-width: 100% !important;
    height: auto !important;
}

/* Floating toggle container - absolute positioned at top-right */
.view-toggle-float {
    position: absolute !important;
    top: 4px !important;
    right: auto !important;
    left: 50px !important;
    z-index: 100 !important;
    height: 28px !important;  /* Fixed height */
    margin: 0 !important;
    padding: 0 !important;
    display: flex !important;
    gap: 0 !important;
    background: rgba(26, 10, 46, 0.85) !important;
    border: 1px solid rgba(124, 58, 237, 0.3) !important;
    border-radius: 4px !important;
    padding: 2px !important;
    width: auto !important;
    min-width: auto !important;
    max-width: fit-content !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: flex-end !important;
}

/* Compact toggle buttons - side by side */
/* Force row to not expand */
.view-toggle-float > div,
.view-toggle-float .flex,
.view-toggle-float [class*="flex"] {
    width: auto !important;
    min-width: auto !important;
    flex: 0 0 auto !important;
    display: flex !important;
    flex-direction: row !important;
}

/* Gradio Button override - target internal button element */
.view-toggle-btn-compact,
.view-toggle-btn-compact > button,
.view-toggle-btn-compact button,
.view-toggle-btn-compact [data-testid="button"] {
    width: 40px !important;
    height: 24px !important;
    min-width: 40px !important;
    min-height: 24px !important;
    max-width: 40px !important;
    max-height: 24px !important;
    flex: 0 0 40px !important;
    padding: 0 !important;
    margin: 0 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    border: none !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    background: transparent !important;
    color: #94a3b8 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    letter-spacing: -0.3px !important;
    box-sizing: border-box !important;
    flex-shrink: 0 !important;
}

/* Left button - rounded left corners */
.view-toggle-btn-compact.view-toggle-left,
.view-toggle-btn-compact.view-toggle-left > button {
    border-radius: 2px 0 0 2px !important;
    border-right: 1px solid rgba(124, 58, 237, 0.2) !important;
}

/* Right button - rounded right corners */
.view-toggle-btn-compact.view-toggle-right,
.view-toggle-btn-compact.view-toggle-right > button {
    border-radius: 0 2px 2px 0 !important;
}

/* Hover state - higher specificity */
.view-toggle-btn-compact:hover,
.view-toggle-btn-compact:hover > button,
.view-toggle-btn-compact > button:hover {
    color: #ffffff !important;
    background: rgba(124, 58, 237, 0.2) !important;
}

/* Non-active state - transparent background */
.view-toggle-btn-compact:not(.view-active),
.view-toggle-btn-compact:not(.view-active) > button,
.view-toggle-btn-compact:not(.view-active) > button:active,
.view-toggle-float .view-toggle-btn-compact:not(.view-active) {
    background: transparent !important;
    color: rgba(255, 255, 255, 0.6) !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
}

/* Active state - golden - highest specificity */
.view-toggle-btn-compact.view-active,
.view-toggle-btn-compact.view-active > button,
.view-toggle-btn-compact.view-active > button:active,
.view-toggle-float .view-toggle-btn-compact.view-active {
    background: #FFC40E !important;
    color: #1a0a2e !important;
    border-color: #FFC40E !important;
}

/* ===== Global 2D/3D switch in the control row ===== */
/* .control-item forces min-width:0, which would collapse the switch to 0px */
.control-item.all-view-item {
    flex: 0 0 auto !important;
    min-width: 200px !important;
    width: 200px !important;
}
.all-view-toggle {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 6px;
}
.all-view-btn {
    flex: 0 1 auto !important;
    min-width: 0 !important;
    height: 30px !important;
    padding: 0 10px !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 0.4px !important;
    border-radius: 6px !important;
    border: 1px solid rgba(124, 58, 237, 0.45) !important;
    background: rgba(255, 255, 255, 0.04) !important;
    color: rgba(255, 255, 255, 0.72) !important;
    cursor: pointer !important;
    white-space: nowrap !important;
    transition: all 0.2s ease !important;
}
.all-view-btn:hover {
    background: rgba(124, 58, 237, 0.28) !important;
    border-color: rgba(124, 58, 237, 0.8) !important;
    color: #ffffff !important;
}
.all-view-btn.view-active {
    background: #FFC40E !important;
    border-color: #FFC40E !important;
    color: #1a0a2e !important;
}

/* ===== Overlay panel: browser-side sweep (both layers stay visible) ===== */
/* Popup shown when a class the current dataset does not have is clicked */
#class-avail-toast {
    position: fixed;
    top: 26px;
    left: 50%;
    transform: translate(-50%, -12px);
    z-index: 2147483000;
    max-width: 70vw;
    padding: 12px 20px;
    border-radius: 10px;
    background: linear-gradient(135deg, rgba(45, 27, 78, 0.97) 0%, rgba(26, 10, 46, 0.97) 100%);
    border: 1px solid rgba(255, 196, 14, 0.55);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.55);
    color: #ffffff;
    font-size: 14px;
    font-weight: 500;
    text-align: center;
    opacity: 0;
    pointer-events: none;
    transition: opacity 0.25s ease, transform 0.25s ease;
}
#class-avail-toast.show {
    opacity: 1;
    transform: translate(-50%, 0);
}

/* Gradio removes visible=False components from the DOM, so keep helper triggers
   off-screen instead - JavaScript can still click them. */
.offscreen-trigger {
    position: fixed !important;
    left: -9999px !important;
    top: 0 !important;
    width: 1px !important;
    height: 1px !important;
    opacity: 0 !important;
    pointer-events: none !important;
}

/* Sub header for individual chart titles - fixed height for alignment */
.col-header-sub {
    font-size: 0.8rem !important;
    color: #FFC40E !important;
    font-weight: 500 !important;
    padding: 4px 8px !important;
    margin: 4px 0 !important;
    background: transparent !important;
    border: none !important;
    text-align: center !important;
    height: 24px !important;  /* Fixed height */
    line-height: 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}

/* Plot containers */
.plot-container {
    position: relative !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}

/* 3D active state - golden glow */
.chart-3d-active {
    box-shadow: 0 0 0 1px rgba(255, 196, 14, 0.4), 0 0 15px rgba(255, 196, 14, 0.15) !important;
    border-radius: 6px !important;
}

/* Main content row - ensure three columns align */
.equal-height > div,
[equal_height="true"] > div {
    display: flex !important;
    flex-direction: column !important;
}

/* Ensure each column has consistent spacing */
.equal-height .col-header-sub:first-of-type {
    margin-top: 8px !important;
}

/* Plot containers - no borders */
.plot-container {
    position: relative !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
"""

# ==================== JavaScript for Loading Screen ====================
LOADING_SCREEN_JS = """
<script>
(function() {
    // Global loading-state manager
    window.AppLoader = {
        screen: null,
        bar: null,
        status: null,
        percent: null,
        progress: 0,
        targetProgress: 0,
        isComplete: false,
        loadStartTime: Date.now(),
        minDisplayTime: 2500,  // Show for at least 2.5 s
        
        // Create the loading screen
        create: function() {
            if (document.getElementById('loading-screen')) return;
            
            // Main container
            this.screen = document.createElement('div');
            this.screen.id = 'loading-screen';
            this.screen.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:#1a0a2e;display:flex;flex-direction:column;justify-content:center;align-items:center;z-index:2147483647;transition:opacity 0.5s ease-out;';
            
            // Title
            var title = document.createElement('div');
            title.textContent = 'End-to-End Radar Echo Segmentation';
            title.style.cssText = 'color:#FFC40E;font-size:2.2rem;font-weight:800;margin-bottom:16px;text-align:center;letter-spacing:2px;font-family:"Exo 2",sans-serif;text-shadow:0 0 20px rgba(255,196,14,0.5);';
            
            // Subtitle
            var subtitle = document.createElement('div');
            subtitle.textContent = 'Perception & Cognition Laboratory';
            subtitle.style.cssText = 'color:#c4b5fd;font-size:1rem;font-weight:400;margin-bottom:80px;text-align:center;font-family:"Exo 2",sans-serif;letter-spacing:4px;';
            
            // Progress bar container
            var progressWrap = document.createElement('div');
            progressWrap.style.cssText = 'width:400px;';
            
            // Progress bar frame
            var container = document.createElement('div');
            container.style.cssText = 'width:100%;height:8px;background:rgba(0,0,0,0.4);border:2px solid rgba(255,255,255,0.5);border-radius:10px;padding:2px;box-shadow:0 0 20px rgba(124,58,237,0.3);';
            
            // Progress bar fill
            this.bar = document.createElement('div');
            this.bar.style.cssText = 'height:100%;width:0%;background:linear-gradient(90deg,#f59e0b 0%,#FFC40E 30%,#fff5a8 50%,#FFC40E 70%,#f59e0b 100%);border-radius:6px;box-shadow:0 0 10px rgba(255,196,14,0.6);transition:width 0.2s ease-out;';
            
            container.appendChild(this.bar);
            progressWrap.appendChild(container);
            
            // Status row
            var statusRow = document.createElement('div');
            statusRow.style.cssText = 'display:flex;justify-content:center;align-items:center;margin-top:20px;gap:12px;';
            
            this.status = document.createElement('span');
            this.status.textContent = 'Initializing...';
            this.status.style.cssText = 'color:#c4b5fd;font-size:0.9rem;font-family:monospace;letter-spacing:1px;text-transform:uppercase;';
            
            this.percent = document.createElement('span');
            this.percent.textContent = '0%';
            this.percent.style.cssText = 'color:#FFC40E;font-size:0.9rem;font-family:monospace;font-weight:700;min-width:40px;text-align:right;';
            
            statusRow.appendChild(this.status);
            statusRow.appendChild(this.percent);
            progressWrap.appendChild(statusRow);
            
            this.screen.appendChild(title);
            this.screen.appendChild(subtitle);
            this.screen.appendChild(progressWrap);
            
            document.body.appendChild(this.screen);
            
            // Start the animation loop
            this.animate();
        },
        
        // Animation update
        animate: function() {
            var self = this;
            
            function updateFrame() {
                // Smoothly approach the target progress
                var diff = self.targetProgress - self.progress;
                if (Math.abs(diff) > 0.1) {
                    self.progress += diff * 0.1;
                } else {
                    self.progress = self.targetProgress;
                }
                
                // Update the display
                if (self.bar) {
                    self.bar.style.width = self.progress + '%';
                }
                if (self.percent) {
                    self.percent.textContent = Math.floor(self.progress) + '%';
                }
                
                // Update the status text
                if (self.status) {
                    if (self.progress < 30) self.status.textContent = 'Initializing...';
                    else if (self.progress < 60) self.status.textContent = 'Loading models...';
                    else if (self.progress < 85) self.status.textContent = 'Processing data...';
                    else if (self.progress < 100) self.status.textContent = 'Rendering charts...';
                    else self.status.textContent = 'Ready!';
                }
                
                // Check whether we can close
                if (self.isComplete && self.progress >= 99 && !self.isHiding) {
                    var elapsed = Date.now() - self.loadStartTime;
                    if (elapsed >= self.minDisplayTime) {
                        self.hide();
                        return;
                    }
                }
                
                if (!self.isHiding) {
                    requestAnimationFrame(updateFrame);
                }
            }
            
            requestAnimationFrame(updateFrame);
        },
        
        // Set the target progress
        setProgress: function(p) {
            this.targetProgress = Math.min(Math.max(p, 0), 100);
        },
        
        // Model loading phase (0-40%)
        setModelLoading: function() {
            this.setProgress(30);
        },
        
        // Data processing phase (40-80%)
        setProcessing: function() {
            this.setProgress(60);
        },
        
        // Chart rendering phase (80-99%)
        setRendering: function() {
            this.setProgress(85);
        },
        
        // Loading complete
        complete: function() {
            this.isComplete = true;
            this.setProgress(100);
        },
        
        // Hide the loading screen
        hide: function() {
            var self = this;
            if (this.isHiding || !this.screen) return;
            this.isHiding = true;
            
            this.screen.style.opacity = '0';
            setTimeout(function() {
                if (self.screen) {
                    self.screen.remove();
                    self.screen = null;
                }
                // Fire the event that signals loading finished
                window.dispatchEvent(new CustomEvent('appLoaderComplete'));
            }, 500);
        }
    };
    
    // Observe the Gradio loading state
    function observeGradioLoading() {
        var checkCount = 0;
        var maxChecks = 150;  // Check at most 150 times (15s)
        var lastPlotCount = 0;
        var stableCount = 0;  // How many times the chart count stayed stable
        
        var checkInterval = setInterval(function() {
            checkCount++;
            
            // Check whether any Plotly chart exists
            var plots = document.querySelectorAll('.js-plotly-plot');
            var hasPlots = plots.length >= 8;  // Need at least 8 charts
            
            if (hasPlots) {
                // Check that the chart count is stable (rendering finished)
                if (plots.length === lastPlotCount && plots.length > 0) {
                    stableCount++;
                } else {
                    stableCount = 0;
                }
                lastPlotCount = plots.length;
                
                // Check that every chart has data
                var allHaveData = true;
                for (var i = 0; i < plots.length; i++) {
                    var plot = plots[i];
                    // After initialisation a Plotly chart exposes _fullLayout or data
                    if (!plot.data || plot.data.length === 0) {
                        if (!plot._fullLayout) {
                            allHaveData = false;
                            break;
                        }
                    }
                }
                
                // Progress to 90% and wait until the charts confirm they rendered
                window.AppLoader.setProgress(90 + Math.min(stableCount, 8));
                
                // Condition: chart count is stable, every chart has data, and we checked at least 5 times
                if (stableCount >= 3 && allHaveData && plots.length >= 8) {
                    clearInterval(checkInterval);
                    window.AppLoader.complete();
                    return;
                }
            } else {
                // Waiting for the charts to be created
                window.AppLoader.setProgress(75 + Math.min(checkCount * 0.5, 10));
            }
            
            // Timeout safeguard
            if (checkCount >= maxChecks) {
                clearInterval(checkInterval);
                window.AppLoader.complete();
            }
        }, 100);  // Check every 100 ms
    }
    
    // Detect when all charts finished loading
    function areAllPlotsReady() {
        var plots = document.querySelectorAll('.js-plotly-plot');
        
        // need at least 8 charts (single tab) or more (both tabs)
        if (plots.length < 8) return false;
        
        for (var i = 0; i < plots.length; i++) {
            var plot = plots[i];
            // Check that the Plotly chart is fully initialised
            var hasData = plot.data && plot.data.length > 0;
            var hasLayout = plot._fullLayout || (plot.layout && Object.keys(plot.layout).length > 0);
            
            if (!hasData && !hasLayout) {
                return false;
            }
        }
        
        return true;
    }
    
    // Initialise
    function init() {
        window.AppLoader.create();
        window.AppLoader.setProgress(10);  // Startup phase
        
        // Give Gradio a moment to initialise
        setTimeout(function() {
            window.AppLoader.setProgress(25);
            
            // Start watching for chart loading
            setTimeout(observeGradioLoading, 500);
        }, 500);
    }
    
    // Make sure we initialise after the DOM is ready
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
</script>
"""

# ==================== JavaScript for 2D/3D View Toggle ====================
VIEW_TOGGLE_JS = """
<script>
(function() {
    // Global view-state manager
    window.ChartViewManager = {
        viewStates: {},
        
        // Read the view state, defaulting to 2D
        getView: function(chartKey) {
            return this.viewStates[chartKey] || '2d';
        },
        
        // Set the view state
        setView: function(chartKey, mode) {
            this.viewStates[chartKey] = mode;
            this.updateButtonState(chartKey, mode);
        },
        
        // Toggle the view
        toggleView: function(chartKey) {
            var current = this.getView(chartKey);
            var newMode = current === '2d' ? '3d' : '2d';
            this.setView(chartKey, newMode);
            return newMode;
        },
        
        // Update the button styles
        updateButtonState: function(chartKey, mode) {
            var btn2d = document.getElementById('btn-' + chartKey + '-2d');
            var btn3d = document.getElementById('btn-' + chartKey + '-3d');
            
            if (btn2d && btn3d) {
                if (mode === '2d') {
                    btn2d.classList.add('view-2d-active');
                    btn2d.classList.remove('view-3d-active');
                    btn3d.classList.remove('view-2d-active', 'view-3d-active');
                } else {
                    btn3d.classList.add('view-3d-active');
                    btn3d.classList.remove('view-2d-active');
                    btn2d.classList.remove('view-2d-active', 'view-3d-active');
                }
            }
        }
    };
    
    // Global toggle function (called from HTML buttons)
    window.toggleView = function(chartKey, mode) {
        // Update the UI state
        window.ChartViewManager.setView(chartKey, mode);
        
        // Trigger the Gradio event (if a callback is bound)
        var eventName = 'view_toggle_' + chartKey;
        if (window.gradioApp && window.gradioApp.submit) {
            window.gradioApp.submit(eventName, mode);
        }
    };
    
   // Initialise the view state (restore from localStorage)
    function initViewStates() {
        try {
            var saved = localStorage.getItem('chartViewStates');
            if (saved) {
                window.ChartViewManager.viewStates = JSON.parse(saved);
            }
        } catch(e) {
            console.log('No saved view states');
        }
    }
    
    // Persist the view state to localStorage
    function saveViewStates() {
        try {
            localStorage.setItem('chartViewStates', JSON.stringify(window.ChartViewManager.viewStates));
        } catch(e) {
            console.log('Failed to save view states');
        }
    }
    
    // Restore the state once the page has loaded
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initViewStates);
    } else {
        initViewStates();
    }
    
    // Save the state before the page unloads
    window.addEventListener('beforeunload', saveViewStates);
    
    // Double-click a 3D chart to reset the camera
    function setup3DResetHandler() {
        document.addEventListener('dblclick', function(e) {
            // Check whether the click landed on a 3D chart
            var plotlyDiv = e.target.closest('.js-plotly-plot');
            if (plotlyDiv && plotlyDiv.data && plotlyDiv.data[0] && plotlyDiv.data[0].type === 'surface') {
                // Reset the camera to a 30 degree view
                Plotly.relayout(plotlyDiv, {
                    'scene.camera': {
                        eye: {x: 0, y: -2.2, z: 1.3},
                        up: {x: 0, y: 0, z: 1},
                        center: {x: 0, y: 0, z: 0}
                    }
                });
            }
        });
    }
    
    // Set up the 3D reset behaviour
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup3DResetHandler);
    } else {
        setup3DResetHandler();
    }
    
    // Centre 3D charts
    function center3DPlots() {
        var plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach(function(plot) {
            // Check whether this is a 3D chart (surface type)
            if (plot.data && plot.data[0] && plot.data[0].type === 'surface') {
                // Find the outer container
                var container = plot.closest('.chart-with-toggle') || plot.parentElement;
                if (container) {
                    // Make the container a centred flex box
                    container.style.display = 'flex';
                    container.style.justifyContent = 'center';
                    container.style.alignItems = 'center';
                    // Centre the plot itself
                    plot.style.margin = '0 auto';
                }
            }
        });
    }
    
    // Watch DOM changes and centre 3D charts when they appear
    var plotObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    if (node.classList && node.classList.contains('js-plotly-plot')) {
                        setTimeout(center3DPlots, 100); // Small delay so rendering can finish
                    }
                    // Check the children
                    var plots = node.querySelectorAll ? node.querySelectorAll('.js-plotly-plot') : [];
                    if (plots.length > 0) {
                        setTimeout(center3DPlots, 100);
                    }
                }
            });
        });
    });
    
    // Start observing
    if (document.body) {
        plotObserver.observe(document.body, { childList: true, subtree: true });
        // Initial centring
        setTimeout(center3DPlots, 500);
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            plotObserver.observe(document.body, { childList: true, subtree: true });
            setTimeout(center3DPlots, 500);
        });
    }
    
    // Re-centre when the window is resized
    window.addEventListener('resize', center3DPlots);
})();
</script>
"""

# ==================== Global 2D/3D switch ====================
ALL_VIEW_TOGGLE_HTML = '''
<div class="all-view-toggle">
    <button class="all-view-btn all-view-btn-2d view-active" type="button"
            onclick="window.setAllChartViews('2d')">ALL 2D</button>
    <button class="all-view-btn all-view-btn-3d" type="button"
            onclick="window.setAllChartViews('3d')">ALL 3D</button>
</div>
'''

# ==================== Class availability per dataset ====================
# Which class-filter buttons make sense for each dataset, taken from
# valid_classes in datasets/<name>/dataset_meta.json. Classes the dataset does
# not contain are greyed out, and clicking them answers with a popup instead of
# running the filter. The server-side guard in apply_class_filter stays as a
# second line of defence.
CLASS_AVAILABILITY_JS = """
<script>
(function () {
    'use strict';

    var AVAILABILITY = __CLASS_AVAILABILITY__;   // {"<dataset>": [class indices]}
    var CLASS_NAMES = ['BACKGROUND', 'Target', 'DFTJ', 'ISRJ', 'SIJ', 'WJ', 'NJ'];
    var toastTimer = null;

    function legendButtons() {
        return Array.prototype.slice.call(
            document.querySelectorAll('.legend-btn-inner .legend-class-btn'));
    }

    // Only a string that is an actual dataset name is accepted, so stray text
    // from the dropdown markup can never be mistaken for the selection.
    function currentDataset() {
        var host = document.getElementById('cmp-data-source');
        if (!host) return null;
        var candidates = [];
        var inp = host.querySelector('input');
        if (inp && inp.value) candidates.push(inp.value.trim());
        Array.prototype.forEach.call(
            host.querySelectorAll('[role="listbox"], [role="combobox"]'),
            function (e) { candidates.push((e.textContent || '').trim()); });
        candidates.push((host.textContent || '').trim());
        for (var i = 0; i < candidates.length; i++) {
            if (Object.prototype.hasOwnProperty.call(AVAILABILITY, candidates[i])) {
                return candidates[i];
            }
        }
        return null;
    }

    function showToast(message) {
        var el = document.getElementById('class-avail-toast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'class-avail-toast';
            document.body.appendChild(el);
        }
        el.textContent = message;
        el.classList.add('show');
        if (toastTimer) clearTimeout(toastTimer);
        toastTimer = setTimeout(function () { el.classList.remove('show'); }, 2800);
    }

    function applyAvailability() {
        var dataset = currentDataset();
        var allowed = (dataset && AVAILABILITY[dataset]) ? AVAILABILITY[dataset] : null;
        legendButtons().forEach(function (btn, idx) {
            var ok = !allowed || allowed.indexOf(idx) !== -1;
            btn.classList.toggle('disabled', !ok);
            if (ok) {
                btn.removeAttribute('title');
            } else {
                btn.setAttribute('title', 'Not available in ' + dataset);
            }
        });
    }

    // Intercept in the capture phase so the filter never runs for such a class.
    // The availability is refreshed first, because Gradio re-renders the tab
    // content (and therefore the buttons) on its own schedule.
    document.addEventListener('click', function (ev) {
        var target = ev.target;
        var btn = (target && target.closest) ? target.closest('.legend-class-btn') : null;
        if (!btn) return;
        applyAvailability();
        if (!btn.classList.contains('disabled')) return;
        ev.preventDefault();
        ev.stopPropagation();
        var idx = legendButtons().indexOf(btn);
        var name = CLASS_NAMES[idx] || (btn.textContent || '').trim();
        showToast((currentDataset() || 'This dataset') +
                  ' does not contain the "' + name + '" class.');
    }, true);

    // Keep the greying in sync: the dropdown changes the available classes and
    // the tab is mounted lazily, so watch the DOM as well as the dropdown.
    function refreshIfChanged() {
        var key = currentDataset() + '|' + legendButtons().length +
                  '|' + legendButtons().map(function (b) {
                      return b.classList.contains('disabled') ? '1' : '0';
                  }).join('');
        if (key === lastKey) return;
        lastKey = key;
        applyAvailability();
    }

    var lastKey = null;
    var pending = null;
    function scheduleRefresh() {
        if (pending) return;
        pending = setTimeout(function () { pending = null; refreshIfChanged(); }, 150);
    }

    function start() {
        refreshIfChanged();
        new MutationObserver(scheduleRefresh).observe(document.body,
            { childList: true, subtree: true });
        document.addEventListener('change', scheduleRefresh, true);
        document.addEventListener('input', scheduleRefresh, true);
        window.addEventListener('resize', scheduleRefresh);
        setInterval(refreshIfChanged, 1000);   // cheap safety net
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start);
    } else {
        start();
    }
})();
</script>
"""


def build_class_availability_js():
    """Fill the availability script with each dataset's valid_classes."""
    availability = {
        name: assets.load_dataset_meta(name).get("valid_classes", [0, 1])
        for name in assets.available_datasets()
    }
    return CLASS_AVAILABILITY_JS.replace("__CLASS_AVAILABILITY__", json.dumps(availability))


# ==================== UI helpers: global 2D/3D switch ====================
UI_HELPERS_JS = """
<script>
(function () {
    'use strict';

    // ---- Global 2D/3D switch (drives the existing per-chart toggle buttons) ----
    // Only charts that really own a 2D/3D toggle react, and charts living in a
    // hidden tab are skipped, so 2D-only panels are left alone.
    window.setAllChartViews = function (mode) {
        var want3d = (mode === '3d');
        var sel = want3d ? '.view-toggle-btn-compact.view-toggle-right'
                         : '.view-toggle-btn-compact.view-toggle-left';
        var nodes = document.querySelectorAll(sel);
        var clicked = 0;
        for (var i = 0; i < nodes.length; i++) {
            var b = nodes[i];
            if (b.classList.contains('all-view-btn')) continue;
            if (b.classList.contains('view-active')) continue;
            if (b.getClientRects().length === 0) continue;   // hidden tab / collapsed
            b.click();
            clicked++;
        }
        var all = document.querySelectorAll('.all-view-btn');
        for (var j = 0; j < all.length; j++) {
            var is3d = all[j].classList.contains('all-view-btn-3d');
            all[j].classList.toggle('view-active', is3d === want3d);
        }
        return clicked;
    };

    // ---- Click the overlay panel to jump to the other result ----------------
    // Delegated from the document so it survives Gradio re-renders. A click only
    // counts as a tap (a zoom drag must not flip the view).
    var downX = null, downY = null;
    document.addEventListener('pointerdown', function (ev) {
        downX = ev.clientX; downY = ev.clientY;
    }, true);
    document.addEventListener('click', function (ev) {
        var host = document.getElementById('overlay-plot');
        if (!host || !host.contains(ev.target)) return;
        if (ev.target.closest && ev.target.closest('.modebar')) return;   // toolbar
        if (downX !== null && (Math.abs(ev.clientX - downX) > 6 || Math.abs(ev.clientY - downY) > 6)) return;
        var btn = document.getElementById('overlay-flip-trigger');
        if (btn) btn.click();
    }, false);
})();
</script>
"""

# ==================== JavaScript for Dropdown Position Fix ====================
DROPDOWN_FIX_JS = """
<script>
// Dropdown positioning is handled by CSS, no JS needed
</script>
"""

# ==================== JavaScript for Class Filter ====================
CLASS_FILTER_JS = """
<script>
(function() {
    'use strict';
    
    // Currently selected class index (-1 = show all)
    window.selectedClassIndex = -1;
    
    // Class name mapping
    const classNames = {
        0: 'BACKGROUND',
        1: 'Target', 
        2: 'DFTJ',
        3: 'ISRJ',
        4: 'SIJ',
        5: 'WJ',
        6: 'BJ',
        7: 'CFAR-Target'
    };
    
    // Class colour map (matches the static legend)
    const classColors = {
        0: '#440154',  // BACKGROUND - dark purple
        1: '#FDE725',  // Target - yellow
        2: '#B71C1C',  // DFTJ - deep red
        3: '#E53935',  // ISRJ - red
        4: '#EF9A9A',  // SIJ - light red
        5: '#2563EB',  // WJ - blue
        6: '#7DD3FC'   // BJ - light blue
    };
    
    // Add colour swatches to the class buttons - matched via the title attribute
    function initLegendButtonColors() {
        const colorMap = [
            { match: 'Target', color: '#FDE725' },
            { match: 'DFTJ', color: '#B71C1C' },
            { match: 'ISRJ', color: '#E53935' },
            { match: 'SIJ', color: '#EF9A9A' },
            { match: 'WJ', color: '#2563EB' },
            { match: 'BJ', color: '#7DD3FC' },
            { match: 'BACKGROUND', color: '#440154' }
        ];
        
        // Try several selectors
        let buttons = document.querySelectorAll('.legend-class-btn');
        
        // If that fails, try a more generic selector
        if (buttons.length === 0) {
            buttons = document.querySelectorAll('[class*="legend-c"]');
        }
        
        buttons.forEach(btn => {
            const text = btn.textContent.trim();
            
            // Skip buttons that were already processed
            if (btn.dataset.colorInit === 'true') return;
            
            for (const item of colorMap) {
                if (text === item.match || text.includes(item.match)) {
                    // Create the colour swatch
                    const colorBox = document.createElement('span');
                    colorBox.className = 'legend-color-box';
                    colorBox.style.cssText = `display:inline-block;width:10px;height:10px;background:${item.color};border-radius:2px;margin-right:5px;vertical-align:middle;border:1px solid rgba(255,255,255,0.3);flex-shrink:0;`;
                    
                    // Find the text container inside the button, or insert directly
                    const innerDiv = btn.querySelector('div');
                    if (innerDiv && innerDiv.firstChild) {
                        // Insert as the first child of the div inside the Gradio button
                        innerDiv.insertBefore(colorBox, innerDiv.firstChild);
                    } else {
                        // Insert directly at the front of the button
                        btn.insertBefore(colorBox, btn.firstChild);
                    }
                    btn.dataset.colorInit = 'true';
                    break;
                }
            }
        });
    }
    
    // Retry initialisation several times
    function initColorWithDelay() {
        [100, 300, 500, 1000, 2000, 3000].forEach(delay => {
            setTimeout(initLegendButtonColors, delay);
        });
    }
    
    // Re-initialise when the tab changes
    function setupTabObserver() {
        const tabBtns = document.querySelectorAll('[role="tab"]');
        tabBtns.forEach(tab => {
            tab.addEventListener('click', () => {
                setTimeout(initLegendButtonColors, 300);
                setTimeout(initLegendButtonColors, 600);
            });
        });
    }
    
   // Initialise on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initColorWithDelay();
            setTimeout(setupTabObserver, 1000);
        });
    } else {
        initColorWithDelay();
        setTimeout(setupTabObserver, 1000);
    }
    
    // Watch for DOM changes
    const colorObserver = new MutationObserver(function() {
        initLegendButtonColors();
    });
    
    setTimeout(function() {
        if (document.body) {
            colorObserver.observe(document.body, { childList: true, subtree: true });
        }
    }, 500);
    
    // Update the button visuals
    function updateButtonStates(activeIndex) {
        const buttons = document.querySelectorAll('.legend-btn');
        const resetBtn = document.getElementById('legend-reset');
        
        buttons.forEach(btn => {
            const idx = parseInt(btn.dataset.class);
            if (idx === activeIndex) {
                btn.classList.add('active');
            } else {
                btn.classList.remove('active');
            }
        });
        
        // Pulse hint on the reset button
        if (resetBtn) {
            if (activeIndex >= 0) {
                resetBtn.classList.add('has-active');
            } else {
                resetBtn.classList.remove('has-active');
            }
        }
    }
    
    // Call the Gradio filter function
    function triggerGradioFilter(classIdx) {
        const filterInput = document.getElementById('class-filter-input');
        if (filterInput) {
            filterInput.value = classIdx.toString();
            filterInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            const triggerBtn = document.getElementById('class-filter-trigger');
            if (triggerBtn) {
                triggerBtn.click();
            }
        }
    }
    
    // A class button was clicked
    window.filterByClass = function(classIdx, btnElement) {
        // BACKGROUND (0) is disabled
        if (classIdx === 0) return;
        
        // Clicking the selected button again clears the selection (show all)
        if (window.selectedClassIndex === classIdx) {
            window.resetClassFilter();
            return;
        }
        
        window.selectedClassIndex = classIdx;
        updateButtonStates(classIdx);
        triggerGradioFilter(classIdx);
    };
    
    // Reset the filter
    window.resetClassFilter = function() {
        window.selectedClassIndex = -1;
        updateButtonStates(-1);
        triggerGradioFilter(-1);
    };
})();
</script>
"""

# ==================== Build Interface ====================
def create_interface():
    available_datasets = get_available_datasets()
    if not available_datasets:
        print("[warn] No dataset found under datasets/ -- the data source dropdown will be empty. "
              "Download the models and datasets from Hugging Face first (see README Step 1).")
    
    # Only expose datasets that actually exist on disk
    all_data_sources = available_datasets
    
    with gr.Blocks(title="EchMo-Vis") as demo:
        # Inject the dropdown-positioning JavaScript
        gr.HTML(DROPDOWN_FIX_JS)
        # Inject the class-filter JavaScript
        gr.HTML(CLASS_FILTER_JS)
        
        # Header with logo - logo top-right, title centred
        logo_html = f'<img src="data:image/jpeg;base64,{LOGO_BASE64}" alt="PeCoLab Logo" class="header-logo">' if LOGO_BASE64 else ""
        gr.HTML(f'''
        <div class="header-container">
            <div class="header-title-wrapper">
                <div class="header-title">End-to-End Radar Echo Segmentation</div>
                <div class="header-subtitle">Perception & Cognition Laboratory</div>
            </div>
            {logo_html}
        </div>
        ''')
        
        # Global style container
        gr.HTML('''
        <style>
        .tabs-wrapper {
            position: relative;
        }
        /* Static legend style - used by the first tab */
        /* The legend itself is absolutely positioned, so its Gradio wrapper must
           not take part in the flow: otherwise it becomes an extra flex item in
           the tab column and pushes the control row down by one gap (16px).
           NOTE: only the wrapper is collapsed here - .legend-static keeps its size. */
        .legend-static-host {
            height: 0 !important;
            min-height: 0 !important;
            max-height: 0 !important;
            padding: 0 !important;
            margin: 0 0 calc(-1 * var(--spacing-xxl, 16px)) 0 !important;
            overflow: visible !important;
            border: none !important;
            background: transparent !important;
        }
        .legend-static {
            position: absolute;
            top: -70px;
            right: 10px;
            display: flex;
            justify-content: flex-start;
            align-items: center;
            gap: 16px;
            padding: 10px 18px;
            width: fit-content;
            max-width: 90%;
            background: linear-gradient(135deg, rgba(45, 27, 78, 0.85) 0%, rgba(26, 10, 46, 0.85) 100%) !important;
            border-radius: 8px;
            z-index: 100;
            border: 1px solid rgba(124, 58, 237, 0.3);
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.4);
            backdrop-filter: blur(10px);
        }
        .legend-static-item {
            display: flex;
            align-items: center;
            gap: 6px;
            font-size: 14px;
            color: #ffffff !important;
            font-weight: 500;
        }
        .legend-static-color {
            width: 14px;
            height: 14px;
            border-radius: 4px;
            border: 1px solid rgba(255, 255, 255, 0.3);
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
        }
        /* Button legend container - absolutely positioned */
        .legend-btn-container {
            position: absolute !important;
            top: -230px !important;  /* Shift up to line up with the tab titles */
            right: 0px !important;
            z-index: 1000 !important;
            /* Shrink to the real content width (see the flex rule on .legend-class-btn)
               so the bar never stretches left far enough to cover the tab buttons. */
            width: max-content !important;
            min-width: 0 !important;
            max-width: calc(100% - 8px) !important;
        }
        .legend-btn-inner {
            display: flex !important;
            flex-wrap: nowrap !important;
            justify-content: flex-end !important;
            align-items: center !important;
            gap: 4px !important;
            padding: 6px 12px !important;
            background: linear-gradient(135deg, rgba(45, 27, 78, 0.9) 0%, rgba(26, 10, 46, 0.9) 100%) !important;
            border-radius: 8px !important;
            border: 1px solid rgba(124, 58, 237, 0.4) !important;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.5) !important;
            backdrop-filter: blur(10px);
        }
        /* Legend button style - compact version */
        .legend-class-btn {
            font-size: 14px !important;
            color: #ffffff !important;
            font-weight: 500 !important;
            padding: 4px 10px !important;
            border-radius: 4px !important;
            background: rgba(255, 255, 255, 0.05) !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            height: 32px !important;
            min-height: 32px !important;
            line-height: 24px !important;
            white-space: nowrap !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            box-sizing: border-box !important;
            /* Do not let the flex row stretch every chip to the same width -
               keep each one at its own min_width so the bar stays compact. */
            flex: 0 1 auto !important;
        }
        .legend-class-btn:hover {
            background: rgba(255, 255, 255, 0.15) !important;
            border-color: rgba(255, 255, 255, 0.3) !important;
            transform: translateY(-1px) !important;
        }
        .legend-class-btn.active {
            box-shadow: 0 0 0 2px #FFC40E, 0 0 15px rgba(255, 196, 14, 0.5) !important;
            transform: scale(1.05) !important;
            z-index: 10 !important;
        }
        .legend-class-btn.disabled {
            opacity: 0.4;
            cursor: not-allowed !important;
        }
        /* Reset button style */
        .legend-reset-btn {
            width: 28px !important;
            height: 28px !important;
            min-width: 28px !important;
            max-width: 28px !important;
            flex: 0 0 28px !important;
            border-radius: 50% !important;
            background: rgba(255, 255, 255, 0.08) !important;
            border: 1px solid rgba(255, 196, 14, 0.4) !important;
            color: #ffffff !important;
            font-size: 14px !important;
            cursor: pointer !important;
            transition: all 0.2s ease !important;
            margin-left: 4px !important;
            padding: 0 !important;
            display: inline-flex !important;
            align-items: center !important;
            justify-content: center !important;
            flex-shrink: 0 !important;
            overflow: hidden !important;
        }
        .legend-reset-btn:hover {
            background: rgba(255, 196, 14, 0.2) !important;
            border-color: rgba(255, 196, 14, 0.8) !important;
            transform: scale(1.05) !important;
        }
        .legend-reset-btn.has-active {
            animation: resetPulse 1.5s ease-in-out infinite;
        }
        @keyframes resetPulse {
            0%, 100% { box-shadow: 0 0 5px rgba(255, 196, 14, 0.3); }
            50% { box-shadow: 0 0 20px rgba(255, 196, 14, 0.6); }
        }
        /* Hide the default Gradio button icon, use a square glyph instead */
        .legend-class-btn .lucide-icon {
            display: none !important;
        }
        /* Class button colour swatch style */
        .legend-class-btn .color-box {
            display: inline-block !important;
            width: 15px !important;
            height: 15px !important;
            border-radius: 2px !important;
            margin-right: 5px !important;
            vertical-align: middle !important;
            border: 1px solid rgba(255,255,255,0.3) !important;
            flex-shrink: 0;
        }
        /* CSS fallback: use a ::before pseudo-element */
        .legend-c1::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #FDE725; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        .legend-c2::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #B71C1C; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        .legend-c3::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #E53935; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        .legend-c4::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #EF9A9A; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        .legend-c5::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #2563EB; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        .legend-c6::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #7DD3FC; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        .legend-c0::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #440154; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
        /* Vertically centre the button text and pseudo-element */
        .legend-class-btn { 
            display: inline-flex !important; 
            align-items: center !important; 
            justify-content: center !important;
            flex-direction: row !important;
            white-space: nowrap !important;
        }
        /* Make sure the internal Gradio button structure is correct */
        .legend-class-btn > div,
        .legend-class-btn > div > div { 
            display: inline-flex !important; 
            align-items: center !important; 
            flex-direction: row !important;
            gap: 0 !important;
        }
        /* Align the colour swatch pseudo-element */
        .legend-class-btn::before {
            flex-shrink: 0 !important;
            align-self: center !important;
        }
        .legend-color-box {
            flex-shrink: 0 !important;
            align-self: center !important;
        }
        /* Hide the Plotly branding link in the modebar (zoom/pan controls stay).
           The long descendant selector is required - the short form
           ".modebar a[href*=plotly.com]" loses against Plotly's own modebar rules. */
        .js-plotly-plot .plotly .modebar .modebar-group a.modebar-btn--logo,
        .js-plotly-plot .plotly .modebar .modebar-group a.plotlyjsicon,
        .js-plotly-plot .plotly .plotly-logomark {
            display: none !important;
        }
        </style>
        <div class="tabs-wrapper">
        ''')
        
        with gr.Tabs():
            # Original Tab - Single Model
            with gr.Tab("⚔️ EchMo vs. Traditional RSP", id="single"):
                # Static legend - only used by the first tab
                gr.HTML('''
                <div class="legend-static">
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#440154;"></span>BACKGROUND</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#FDE725;"></span>Target</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#B71C1C;"></span>DFTJ</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#E53935;"></span>ISRJ</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#EF9A9A;"></span>SIJ</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#2563EB;"></span>WJ</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#7DD3FC;"></span>NJ</div>
                    <div class="legend-static-item"><span class="legend-static-color" style="background:#45CA77;box-shadow:0 0 4px rgba(69,202,119,0.5);"></span>CFAR-Target</div>
                </div>
                ''', elem_classes="legend-static-host")
                
                # State used to track ground-truth visibility
                has_gt_mask = gr.State(True)
                
                # Control bar - switches dynamically
                with gr.Row(variant="panel", elem_classes="control-row"):
                    # Data Source
                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Data Source</span>')
                        data_source_dropdown = gr.Dropdown(
                            choices=all_data_sources,
                            value=all_data_sources[0] if all_data_sources else None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )
                    
                    # Sequence
                    with gr.Column(scale=2, elem_classes="control-item"):
                        seq_label = gr.HTML('<span class="dynamic-label" id="seq-label-text">Sequence</span>')
                        sequence_dropdown = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"],
                            visible=True
                        )
                    
                    # Frame
                    with gr.Column(scale=2, elem_classes="control-item"):
                        frame_label = gr.HTML('<span class="dynamic-label" id="frame-label-text">Frame</span>')
                        frame_dropdown = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"],
                            visible=True
                        )
                    
                    # Global 2D/3D switch - applies to every chart that supports it
                    with gr.Column(scale=0, min_width=200, elem_classes="control-item all-view-item"):
                        gr.HTML('<span class="dynamic-label">View (all charts)</span>')
                        gr.HTML(ALL_VIEW_TOGGLE_HTML)
                    
                    # Run Button
                    with gr.Column(scale=0, min_width=120):
                        run_button = gr.Button("Run", variant="primary", elem_classes="run-btn")
                
                # Dynamic opacity state
                overlay_alpha = gr.State(0.0)
                # Result pinned by a click on the overlay panel: (layer, deadline).
                # Only the click handler writes this, the timer just reads it, so a
                # stale tick cannot undo the click.
                overlay_manual = gr.State((-1, 0.0))
                echmo_mask_cache = gr.State(None)
                cfar_mask_cache = gr.State(None)
                n_classes_cache = gr.State(2)
                
                # Main content - the shared display area
                with gr.Row(equal_height=True):
                    # First column: segmentation results (order after the swap: Raw -> GT -> Pred)
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">Segmentation Results: Before & After</div>')
                        # Raw Echo with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub" style="margin-top: 8px;">Raw Echo Input</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                raw_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                raw_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            raw_input_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                        # After the swap: Ground Truth is in position 2
                        gr.HTML('<div class="col-header-sub">Ground Truth</div>')
                        gt_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"], visible=True)
                        # Prediction Mask (EchMo + CFAR Overlay)
                        gr.HTML('<div class="col-header-sub">EchMo + CFAR Overlay</div>')
                        # elem_id is the hook the click-to-switch JavaScript uses
                        pred_mask_plot = gr.Plot(label="", show_label=False, container=True,
                                                 elem_classes=["plot-container"], elem_id="overlay-plot")
                        # Timer driving the dynamic-opacity animation
                        timer = gr.Timer(value=0.05, active=True)
                        # Kept in the DOM (off-screen) so JavaScript can click it
                        overlay_flip_trigger = gr.Button("flip", elem_id="overlay-flip-trigger",
                                                         elem_classes="offscreen-trigger")
                    
                    # Second column: EchMo
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">EchMo Intermediate Features</div>')
                        # EchMo LMF with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub" style="margin-top: 8px;">EchMo LMF</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                echmo_lmf_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                echmo_lmf_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            echmo_lmf_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                        # EchMo MLDC with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub">EchMo MLDC</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                echmo_mldc_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                echmo_mldc_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            echmo_mldc_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                        # EchMo GDF with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub">EchMo GDF</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                echmo_gdf_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                echmo_gdf_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            echmo_gdf_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    # Third column: traditional methods
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">Traditional RSP Intermediate Results</div>')
                        # Trad LMF with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub" style="margin-top: 8px;">Traditional LMF</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                trad_lmf_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                trad_lmf_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            trad_lmf_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                        # Trad MTI with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub">Traditional MTI</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                trad_mti_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                trad_mti_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            trad_mti_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                        # Trad MTD with 2D/3D toggle inside chart
                        gr.HTML('<div class="col-header-sub">Traditional MTD</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                trad_mtd_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                trad_mtd_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            trad_mtd_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                
                # Status
                status_text = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    elem_classes="status-bar",
                    show_label=False,
                    container=False,
                    lines=1
                )
        
                # ==================== Event Handlers ====================
                
                def on_data_source_change(data_source):
                    """Refresh the Sequence dropdown when the Data Source changes"""
                    sequences = get_sequences(data_source)
                    return (
                        gr.update(choices=sequences, value=sequences[0] if sequences else None, visible=True),
                        gr.update(value='<span class="dynamic-label">Sequence</span>'),
                        gr.update(value='<span class="dynamic-label">Frame</span>')
                    )
                
                def on_sequence_change(data_source, seq_name):
                    """When Sequence changes, update Frame list"""
                    if not seq_name:
                        return gr.update(choices=[], value=None)
                    frames = get_frames(data_source, seq_name)
                    return gr.update(choices=frames, value=frames[0] if frames else None)
                
                def run_processing(data_source, sequence, frame):
                    """Unified run processing function with 2D/3D view support"""
                    try:
                        results = run_inference(data_source, sequence, frame)
                        
                        has_gt = results[11] if len(results) > 11 else False
                        n_classes = _global_results.get('seg_n_classes', 2)
                        
                        # After swap: results[1] is ground_truth, results[2] is pred_mask, results[9] is cfar_mask
                        gt_fig = arr2plot(results[1], is_mask=True, title=None, n_classes=n_classes) if results[1] is not None else go.Figure()
                        
                        # Cache the EchMo and CFAR masks for the animation
                        echmo_mask = results[2] if len(results) > 2 else None
                        cfar_mask = results[9] if len(results) > 9 else None
                        
                        # Build the initial overlay figure (opacity 0)
                        if echmo_mask is not None and cfar_mask is not None:
                            overlay_fig = create_overlay_figure(echmo_mask, cfar_mask, alpha=0.0, n_classes=n_classes)
                        else:
                            overlay_fig = go.Figure()
                        
                        # Use get_plot_for_chart so the 2D/3D view toggle works
                        outputs = [
                            get_plot_for_chart('raw', results[0], title=None),  # 0: raw_input_plot
                            gt_fig,                                     # 1: gt_plot
                            overlay_fig,                                # 2: pred_mask_plot (overlay)
                            get_plot_for_chart('echmo_lmf', results[3], title=None),  # 3: echmo_lmf_plot
                            get_plot_for_chart('trad_lmf', results[4], title=None),  # 4: trad_lmf_plot
                            get_plot_for_chart('echmo_mldc', results[5], title=None),  # 5: echmo_mldc_plot
                            get_plot_for_chart('trad_mti', results[6], title=None),  # 6: trad_mti_plot
                            get_plot_for_chart('echmo_gdf', results[7], title=None),  # 7: echmo_gdf_plot
                            get_plot_for_chart('trad_mtd', results[8], title=None),  # 8: trad_mtd_plot
                            results[10] if len(results) > 10 else "Error: No status",  # 9: status_text
                            has_gt,      # 10: has_gt_mask
                            0.0,         # 11: overlay_alpha
                            echmo_mask,  # 12: echmo_mask_cache
                            cfar_mask,   # 13: cfar_mask_cache
                            n_classes    # 14: n_classes_cache
                        ]
                        
                        return outputs
                    except Exception as e:
                        # On error return empty figures plus a message
                        error_msg = f"Error in run_processing: {str(e)}"
                        print(error_msg)
                        import traceback
                        traceback.print_exc()
                        empty_fig = go.Figure()
                        return [
                            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                            empty_fig, empty_fig, empty_fig, empty_fig,
                            error_msg, False, 0.0, None, None, 2
                        ]
                
                def update_gt_visibility(has_gt):
                    """Update GT plot visibility based on whether GT exists"""
                    return gr.update(visible=has_gt)
                
                # ==================== 2D/3D View Toggle Handlers ====================
                
                def toggle_raw_view_2d():
                    """Toggle Raw Echo to 2D view"""
                    try:
                        data = _view_cache.get_data('raw')
                        _view_cache.set_view_mode('raw', '2d')
                        plot = get_plot_for_chart('raw', data, title=None)
                        return (
                            plot,
                            gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                            gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                        )
                    except Exception as e:
                        print(f"Error in toggle_raw_view_2d: {e}")
                        return go.Figure(), gr.update(), gr.update()
                
                def toggle_raw_view_3d():
                    """Toggle Raw Echo to 3D view"""
                    try:
                        data = _view_cache.get_data('raw')
                        _view_cache.set_view_mode('raw', '3d')
                        plot = get_plot_for_chart('raw', data, title=None)
                        return (
                            plot,
                            gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                            gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                        )
                    except Exception as e:
                        print(f"Error in toggle_raw_view_3d: {e}")
                        return go.Figure(), gr.update(), gr.update()
                
                def toggle_echmo_lmf_view_2d():
                    data = _view_cache.get_data('echmo_lmf')
                    _view_cache.set_view_mode('echmo_lmf', '2d')
                    plot = get_plot_for_chart('echmo_lmf', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_echmo_lmf_view_3d():
                    data = _view_cache.get_data('echmo_lmf')
                    _view_cache.set_view_mode('echmo_lmf', '3d')
                    plot = get_plot_for_chart('echmo_lmf', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def toggle_echmo_mldc_view_2d():
                    data = _view_cache.get_data('echmo_mldc')
                    _view_cache.set_view_mode('echmo_mldc', '2d')
                    plot = get_plot_for_chart('echmo_mldc', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_echmo_mldc_view_3d():
                    data = _view_cache.get_data('echmo_mldc')
                    _view_cache.set_view_mode('echmo_mldc', '3d')
                    plot = get_plot_for_chart('echmo_mldc', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def toggle_echmo_gdf_view_2d():
                    data = _view_cache.get_data('echmo_gdf')
                    _view_cache.set_view_mode('echmo_gdf', '2d')
                    plot = get_plot_for_chart('echmo_gdf', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_echmo_gdf_view_3d():
                    data = _view_cache.get_data('echmo_gdf')
                    _view_cache.set_view_mode('echmo_gdf', '3d')
                    plot = get_plot_for_chart('echmo_gdf', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def toggle_trad_lmf_view_2d():
                    data = _view_cache.get_data('trad_lmf')
                    _view_cache.set_view_mode('trad_lmf', '2d')
                    plot = get_plot_for_chart('trad_lmf', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_trad_lmf_view_3d():
                    data = _view_cache.get_data('trad_lmf')
                    _view_cache.set_view_mode('trad_lmf', '3d')
                    plot = get_plot_for_chart('trad_lmf', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def toggle_trad_mti_view_2d():
                    data = _view_cache.get_data('trad_mti')
                    _view_cache.set_view_mode('trad_mti', '2d')
                    plot = get_plot_for_chart('trad_mti', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_trad_mti_view_3d():
                    data = _view_cache.get_data('trad_mti')
                    _view_cache.set_view_mode('trad_mti', '3d')
                    plot = get_plot_for_chart('trad_mti', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def toggle_trad_mtd_view_2d():
                    data = _view_cache.get_data('trad_mtd')
                    _view_cache.set_view_mode('trad_mtd', '2d')
                    plot = get_plot_for_chart('trad_mtd', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_trad_mtd_view_3d():
                    data = _view_cache.get_data('trad_mtd')
                    _view_cache.set_view_mode('trad_mtd', '3d')
                    plot = get_plot_for_chart('trad_mtd', data, title=None)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                # ==================== Event Bindings ====================
                
                data_source_dropdown.change(
                    fn=on_data_source_change,
                    inputs=data_source_dropdown,
                    outputs=[sequence_dropdown, seq_label, frame_label]
                )
                
                sequence_dropdown.change(
                    fn=on_sequence_change,
                    inputs=[data_source_dropdown, sequence_dropdown],
                    outputs=frame_dropdown
                )
                
                # Note: outputs order matches the interface component definition order
                run_event = run_button.click(
                    fn=run_processing,
                    inputs=[data_source_dropdown, sequence_dropdown, frame_dropdown],
                    outputs=[
                        raw_input_plot, gt_plot, pred_mask_plot,  # Swapped order: Raw → GT → Pred (Overlay)
                        echmo_lmf_plot, trad_lmf_plot, echmo_mldc_plot,
                        trad_mti_plot, echmo_gdf_plot, trad_mtd_plot,
                        status_text, has_gt_mask,
                        overlay_alpha, echmo_mask_cache, cfar_mask_cache, n_classes_cache  # cache state
                    ]
                )
                
                # Timer callback: advance the dynamic-opacity animation. The frame is
                # built by overlay_frame(), which keeps a clicked result on screen
                # until its pin expires.
                timer.tick(
                    fn=overlay_frame,
                    inputs=[overlay_alpha, echmo_mask_cache, cfar_mask_cache, n_classes_cache,
                            overlay_manual],
                    outputs=[pred_mask_plot, overlay_alpha]
                )
                
                # Clicking the overlay panel jumps straight to the other result and
                # pins it for OVERLAY_HOLD_SECONDS, so the next timer tick continues
                # from the clicked result instead of snapping back to the old one.
                def handle_overlay_click(alpha):
                    fig, phase, manual = overlay_flip(alpha)
                    if fig is None:
                        return gr.update(), phase, manual
                    return fig, phase, manual
                
                overlay_flip_trigger.click(
                    fn=handle_overlay_click,
                    inputs=[overlay_alpha],
                    outputs=[pred_mask_plot, overlay_alpha, overlay_manual]
                )
                
                has_gt_mask.change(
                    fn=update_gt_visibility,
                    inputs=has_gt_mask,
                    outputs=gt_plot
                )
                
                # 2D/3D View Toggle Event Bindings for Tab 1
                raw_view_2d.click(fn=toggle_raw_view_2d, inputs=[], outputs=[raw_input_plot, raw_view_2d, raw_view_3d])
                raw_view_3d.click(fn=toggle_raw_view_3d, inputs=[], outputs=[raw_input_plot, raw_view_2d, raw_view_3d])
                
                echmo_lmf_view_2d.click(fn=toggle_echmo_lmf_view_2d, inputs=[], outputs=[echmo_lmf_plot, echmo_lmf_view_2d, echmo_lmf_view_3d])
                echmo_lmf_view_3d.click(fn=toggle_echmo_lmf_view_3d, inputs=[], outputs=[echmo_lmf_plot, echmo_lmf_view_2d, echmo_lmf_view_3d])
                
                echmo_mldc_view_2d.click(fn=toggle_echmo_mldc_view_2d, inputs=[], outputs=[echmo_mldc_plot, echmo_mldc_view_2d, echmo_mldc_view_3d])
                echmo_mldc_view_3d.click(fn=toggle_echmo_mldc_view_3d, inputs=[], outputs=[echmo_mldc_plot, echmo_mldc_view_2d, echmo_mldc_view_3d])
                
                echmo_gdf_view_2d.click(fn=toggle_echmo_gdf_view_2d, inputs=[], outputs=[echmo_gdf_plot, echmo_gdf_view_2d, echmo_gdf_view_3d])
                echmo_gdf_view_3d.click(fn=toggle_echmo_gdf_view_3d, inputs=[], outputs=[echmo_gdf_plot, echmo_gdf_view_2d, echmo_gdf_view_3d])
                
                trad_lmf_view_2d.click(fn=toggle_trad_lmf_view_2d, inputs=[], outputs=[trad_lmf_plot, trad_lmf_view_2d, trad_lmf_view_3d])
                trad_lmf_view_3d.click(fn=toggle_trad_lmf_view_3d, inputs=[], outputs=[trad_lmf_plot, trad_lmf_view_2d, trad_lmf_view_3d])
                
                trad_mti_view_2d.click(fn=toggle_trad_mti_view_2d, inputs=[], outputs=[trad_mti_plot, trad_mti_view_2d, trad_mti_view_3d])
                trad_mti_view_3d.click(fn=toggle_trad_mti_view_3d, inputs=[], outputs=[trad_mti_plot, trad_mti_view_2d, trad_mti_view_3d])
                
                trad_mtd_view_2d.click(fn=toggle_trad_mtd_view_2d, inputs=[], outputs=[trad_mtd_plot, trad_mtd_view_2d, trad_mtd_view_3d])
                trad_mtd_view_3d.click(fn=toggle_trad_mtd_view_3d, inputs=[], outputs=[trad_mtd_plot, trad_mtd_view_2d, trad_mtd_view_3d])
                
                def auto_load_on_start():
                    empty_plots = [go.Figure()] * 9
                    if not available_datasets:
                        return ([gr.update()] * 4 + [True] + empty_plots
                                + ["No dataset found under datasets/ - download the assets first (README Step 1).",
                                   0.0, None, None, 2])
                    
                    dataset = available_datasets[0]
                    sequences = get_sequences(dataset)
                    if not sequences:
                        return ([gr.update(choices=sequences, value=None),
                                 gr.update(visible=True, choices=[], value=None),
                                 gr.update(value='<span class="dynamic-label">Sequence</span>'),
                                 gr.update(value='<span class="dynamic-label">Frame</span>'),
                                 True] + empty_plots
                                + [f"No sequence found in datasets/{dataset}", 0.0, None, None, 2])
                    
                    seq = sequences[0]
                    frames = get_frames(dataset, seq)
                    if not frames:
                        return ([gr.update(choices=sequences, value=seq),
                                 gr.update(visible=True, choices=[], value=None),
                                 gr.update(value='<span class="dynamic-label">Sequence</span>'),
                                 gr.update(value='<span class="dynamic-label">Frame</span>'),
                                 True] + empty_plots
                                + [f"No frame found for {dataset}/{seq}", 0.0, None, None, 2])
                    
                    frame = frames[0]
                    results = run_inference(dataset, seq, frame)
                    n_classes = _global_results.get('seg_n_classes', 2)
                    
                    # Cache the EChMo and CFAR masks
                    echmo_mask = results[2] if len(results) > 2 else None
                    cfar_mask = results[9] if len(results) > 9 else None
                    
                    # Build the initial overlay figure (opacity 0)
                    overlay_fig = go.Figure()
                    if echmo_mask is not None and cfar_mask is not None:
                        overlay_fig = create_overlay_figure(echmo_mask, cfar_mask, alpha=0.0, n_classes=n_classes)
                    
                    # Use get_plot_for_chart to support 2D/3D view toggle
                    plot_results = [
                        get_plot_for_chart('raw', results[0], title=None),
                        arr2plot(results[1], is_mask=True, title=None, n_classes=n_classes) if results[1] is not None else go.Figure(),  # Mask - 2D only
                        overlay_fig,  # Overlay - 2D only
                        get_plot_for_chart('echmo_lmf', results[3], title=None),
                        get_plot_for_chart('trad_lmf', results[4], title=None),
                        get_plot_for_chart('echmo_mldc', results[5], title=None),
                        get_plot_for_chart('trad_mti', results[6], title=None),
                        get_plot_for_chart('echmo_gdf', results[7], title=None),
                        get_plot_for_chart('trad_mtd', results[8], title=None),
                    ]
                    
                    has_gt = results[11] if len(results) > 11 else (results[1] is not None)  # Check results[1] after swap
                    
                    return [
                        gr.update(choices=sequences, value=seq, visible=True),
                        gr.update(visible=True, choices=frames, value=frame),
                        gr.update(value='<span class="dynamic-label">Sequence</span>'),
                        gr.update(value='<span class="dynamic-label">Frame</span>'),
                        has_gt
                    ] + plot_results + [results[10], 0.0, echmo_mask, cfar_mask, n_classes]
                
                demo.load(
                    fn=auto_load_on_start,
                    inputs=None,
                    outputs=[
                        sequence_dropdown, frame_dropdown, seq_label, frame_label, has_gt_mask,
                        raw_input_plot, gt_plot, pred_mask_plot,  # Order after the swap (Overlay)
                        echmo_lmf_plot, trad_lmf_plot, echmo_mldc_plot,
                        trad_mti_plot, echmo_gdf_plot, trad_mtd_plot,
                        status_text,  # status_text
                        overlay_alpha, echmo_mask_cache, cfar_mask_cache, n_classes_cache  # cache state
                    ]
                )
    
            # New Tab - Model Comparison
            with gr.Tab("🔥 Model Comparison", id="comparison"):
                # Comparison tab states
                cmp_has_gt_mask = gr.State(True)
                # Currently selected class index (-1 = show all)
                cmp_selected_class = gr.State(-1)
                # Cache the last inference result so class filtering can redraw instantly
                cmp_last_result = gr.State(None)
                
                # Hidden components for class filter JavaScript integration
                with gr.Row(visible=False):
                    class_filter_input = gr.Textbox(
                        value="-1",
                        elem_id="class-filter-input",
                        interactive=True
                    )
                    class_filter_trigger = gr.Button(
                        "Filter",
                        elem_id="class-filter-trigger",
                        visible=False
                    )
                
                # Control Bar
                with gr.Row(variant="panel", elem_classes="control-row"):
                    # Data Source
                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Data Source</span>')
                        cmp_data_source = gr.Dropdown(
                            choices=available_datasets,
                            value=available_datasets[0] if available_datasets else None,
                            show_label=False,
                            container=False,
                            elem_id="cmp-data-source",
                            elem_classes=["dark-dropdown"]
                        )
                    
                    # Sequence
                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Sequence</span>')
                        cmp_sequence = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )
                    
                    # Frame
                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Frame</span>')
                        cmp_frame = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )
                    
                    # Global 2D/3D switch - applies to every chart that supports it
                    with gr.Column(scale=0, min_width=200, elem_classes="control-item all-view-item"):
                        gr.HTML('<span class="dynamic-label">View (all charts)</span>')
                        gr.HTML(ALL_VIEW_TOGGLE_HTML)
                    
                    # Run Button
                    with gr.Column(scale=0, min_width=120):
                        cmp_run_btn = gr.Button("Run", variant="primary", elem_classes="run-btn")
                
                # Interactive legend bar for class filtering (absolutely positioned top-right)
                with gr.Row():
                    with gr.Column(scale=0):
                        gr.HTML('''<div style="width:0;"></div>''')
                    with gr.Column(scale=0, min_width=650, elem_classes="legend-btn-container"):
                        with gr.Row(variant="compact", elem_classes="legend-btn-inner"):
                            # BACKGROUND - clickable
                            cmp_bg_btn = gr.Button("BACKGROUND", interactive=True,
                                                  elem_classes=["legend-class-btn", "legend-c0"], min_width=115)
                            # Clickable class buttons (JavaScript adds the colour swatches)
                            cmp_target_btn = gr.Button("Target", interactive=True,
                                                      elem_classes=["legend-class-btn", "legend-c1"], min_width=75)
                            cmp_dftj_btn = gr.Button("DFTJ", interactive=True,
                                                    elem_classes=["legend-class-btn", "legend-c2"], min_width=65)
                            cmp_isrj_btn = gr.Button("ISRJ", interactive=True,
                                                    elem_classes=["legend-class-btn", "legend-c3"], min_width=65)
                            cmp_sij_btn = gr.Button("SIJ", interactive=True,
                                                   elem_classes=["legend-class-btn", "legend-c4"], min_width=60)
                            cmp_wj_btn = gr.Button("WJ", interactive=True,
                                                  elem_classes=["legend-class-btn", "legend-c5"], min_width=55)
                            cmp_nj_btn = gr.Button("NJ", interactive=True,
                                                  elem_classes=["legend-class-btn", "legend-c6"], min_width=55)
                            # Reset button
                            cmp_reset_btn = gr.Button("⟲", interactive=True,
                                                     elem_classes="legend-reset-btn", min_width=32)
                
                # Results Area - Row 1: Raw Echo, MTD and Ground Truth
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">Raw Echo Input</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                cmp_raw_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                cmp_raw_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            cmp_raw_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">MTD Input</div>')
                        with gr.Column(elem_classes="chart-with-toggle"):
                            with gr.Row(elem_classes="view-toggle-float"):
                                cmp_mtd_view_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
                                cmp_mtd_view_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                            cmp_mtd_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">Ground Truth Mask</div>')
                        cmp_gt_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"], visible=True)
                
                # Results Area - Row 2: EchMo, TMVA-Net, DeepLabv3+ (all 2D only - masks)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">EchMo</div>')
                        cmp_echmo_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">TMVA-Net</div>')
                        cmp_tmva_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">DeepLabv3+</div>')
                        cmp_deeplab_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                
                # Results Area - Row 3: U-Net, FCN-8s, ViT (all 2D only - masks)
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">U-Net</div>')
                        cmp_unet_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">FCN-8s</div>')
                        cmp_fcn_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    
                    with gr.Column(scale=1, min_width=240):
                        gr.HTML('<div class="col-header">ViT</div>')
                        cmp_vit_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                
                # Status
                cmp_status = gr.Textbox(
                    label="",
                    value="",
                    interactive=False,
                    elem_classes="status-bar",
                    show_label=False,
                    container=False,
                    lines=1
                )
                
                # ==================== Comparison Tab Event Handlers ====================
                
                # 2D/3D View Toggle Handlers for Tab 2
                def toggle_cmp_raw_view_2d():
                    """Toggle Comparison Raw Echo to 2D view"""
                    data = _view_cache.get_data('cmp_raw')
                    _view_cache.set_view_mode('cmp_raw', '2d')
                    plot = get_plot_for_chart('cmp_raw', data)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_cmp_raw_view_3d():
                    """Toggle Comparison Raw Echo to 3D view"""
                    data = _view_cache.get_data('cmp_raw')
                    _view_cache.set_view_mode('cmp_raw', '3d')
                    plot = get_plot_for_chart('cmp_raw', data)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def toggle_cmp_mtd_view_2d():
                    """Toggle Comparison MTD to 2D view"""
                    data = _view_cache.get_data('cmp_mtd')
                    _view_cache.set_view_mode('cmp_mtd', '2d')
                    plot = get_plot_for_chart('cmp_mtd', data)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                    )
                
                def toggle_cmp_mtd_view_3d():
                    """Toggle Comparison MTD to 3D view"""
                    data = _view_cache.get_data('cmp_mtd')
                    _view_cache.set_view_mode('cmp_mtd', '3d')
                    plot = get_plot_for_chart('cmp_mtd', data)
                    return (
                        plot,
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                        gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                    )
                
                def on_cmp_data_source_change(dataset_name):
                    """When comparison data source changes"""
                    if not dataset_name:
                        return gr.update(choices=[], value=None)
                    sequences = get_sequences(dataset_name)
                    return gr.update(choices=sequences, value=sequences[0] if sequences else None)
                
                def on_cmp_sequence_change(dataset_name, seq_name):
                    """When comparison sequence changes"""
                    if not dataset_name or not seq_name:
                        return gr.update(choices=[], value=None)
                    # Check that seq_name belongs to the current dataset
                    valid_sequences = get_sequences(dataset_name)
                    if seq_name not in valid_sequences:
                        # If seq_name is invalid, fall back to the first valid sequence
                        if valid_sequences:
                            seq_name = valid_sequences[0]
                        else:
                            return gr.update(choices=[], value=None)
                    frames = get_frames(dataset_name, seq_name)
                    return gr.update(choices=frames, value=frames[0] if frames else None)
                
                def run_comparison_processing(dataset_name, seq_name, frame_name):
                    """Run comparison inference with caching and 2D/3D view support"""
                    result, msg = run_comparison_inference(dataset_name, seq_name, frame_name)
                    
                    if result is None:
                        return [go.Figure()] * 9 + [msg, False, None, -1]
                    
                    n_classes = result.get('n_classes', 2)
                    has_gt = result['gt_mask'] is not None
                    
                    # Store the dataset name in the result for later class checks
                    result['dataset_name'] = dataset_name
                    
                    # Build the status message
                    display_msg = f"✓ Compared 6 models on {dataset_name}/{seq_name}/{frame_name}"
                    
                    # Cache raw data for 2D/3D view toggle
                    _view_cache.cache_data('cmp_raw', result['raw_echo'])
                    _view_cache.cache_data('cmp_mtd', result['mtd'])
                    
                    plots = [
                        get_plot_for_chart('cmp_raw', result['raw_echo']),  # Support 2D/3D
                        get_plot_for_chart('cmp_mtd', result['mtd']),  # Support 2D/3D
                        arr2plot(result['gt_mask'], is_mask=True, n_classes=n_classes) if result['gt_mask'] is not None else go.Figure(),  # Mask - 2D only
                        arr2plot(result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if result['predictions'].get('echmo') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if result['predictions'].get('unet') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if result['predictions'].get('fcn8s') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if result['predictions'].get('vit') is not None else go.Figure(),
                        display_msg,
                        has_gt,
                        result,  # cached result
                        -1       # Reset the selected class back to "all"
                    ]
                    return plots
                
                def apply_class_filter(filter_input, cached_result):
                    """Apply class filter to cached results - only updates last 6 model plots"""
                    # Parse the input
                    try:
                        parts = filter_input.split(',')
                        target_class_idx = int(parts[0]) if parts else -1
                    except:
                        target_class_idx = -1
                    
                    # Class name mapping
                    class_names = {
                        0: 'BACKGROUND',
                        1: 'Target', 
                        2: 'DFTJ',
                        3: 'ISRJ',
                        4: 'SIJ',
                        5: 'WJ',
                        6: 'NJ',
                        7: 'CFAR-Target'
                    }
                    
                    # Valid classes come from valid_classes in each dataset_meta.json
                    # NOTE: this is the UI class-filter whitelist, not the model output channel count nb_classes
                    dataset_valid_classes = {
                        name: assets.load_dataset_meta(name).get("valid_classes", [0, 1])
                        for name in assets.available_datasets()
                    }
                    
                    # No cached result yet - return an error
                    if cached_result is None:
                        return [go.Figure()] * 6 + ["⚠ No data. Please run inference first.", -1]
                    
                    # Get the dataset name
                    dataset_name = cached_result.get('dataset_name') or next(iter(assets.available_datasets()), None)
                    n_classes = cached_result.get('n_classes', 2)
                    
                    # If the target is "all" (-1), show the full segmentation
                    if target_class_idx == -1:
                        # Return the full segmentation figure
                        return [
                            arr2plot(cached_result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('echmo') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('unet') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('fcn8s') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('vit') is not None else go.Figure(),
                            "✓ Restored full segmentation view",
                            -1
                        ]
                    
                    # Validate the class against the dataset
                    valid_classes = dataset_valid_classes.get(dataset_name, [0, 1])
                    if target_class_idx not in valid_classes:
                        class_name = class_names.get(target_class_idx, f"Class {target_class_idx}")
                        # This dataset does not have that class - return the current figure unchanged
                        return [
                            arr2plot(cached_result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('echmo') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('unet') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('fcn8s') is not None else go.Figure(),
                            arr2plot(cached_result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('vit') is not None else go.Figure(),
                            f"⚠ {dataset_name} does not have category '{class_name}'. View unchanged.",
                            -1
                        ]
                    
                    # Class colour map (matches the legend buttons)
                    class_colors = {
                        0: '#440154',  # BACKGROUND - dark purple
                        1: '#FDE725',  # Target - yellow
                        2: '#B71C1C',  # DFTJ - deep red
                        3: '#E53935',  # ISRJ - red
                        4: '#EF9A9A',  # SIJ - light red
                        5: '#2563EB',  # WJ - blue
                        6: '#7DD3FC'   # BJ - light blue
                    }
                    
                    # Page theme background colour (for the 0 values of the binary mask)
                    theme_bg_color = '#1a0a2e'
                    
                    # Build the binary mask (using the class colour)
                    def create_binary_plot_colored(pred_mask, target_idx, class_name):
                        if pred_mask is None:
                            return go.Figure()
                        
                        # Pick the colour of the target class
                        target_color = class_colors.get(target_idx, '#FDE725')
                        
                        # Build the binary mask - uint8 avoids floating-point precision issues
                        binary_mask = (pred_mask == target_idx).astype(np.uint8)
                        
                        # Build a heatmap with custom colours
                        # 0 uses a transparent/dark background, 1 uses the class colour
                        colorscale = [
                            [0, 'rgba(24, 24, 38, 1)'],  # Theme background colour #181826
                            [1, target_color]
                        ]
                        
                        # Array dimensions (doppler, range)
                        doppler_cells, range_cells = binary_mask.shape
                        
                        # Compute ticks - consistent with arr2plot
                        range_step = max(1, range_cells // 8)
                        doppler_step = max(1, doppler_cells // 6)
                        range_ticks = list(range(0, range_cells, range_step))
                        doppler_ticks = list(range(0, doppler_cells, doppler_step))
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=binary_mask,
                            colorscale=colorscale,
                            showscale=False,
                            zmin=0,
                            zmax=1,
                            hovertemplate='%{y:.0f}, %{x:.0f}<extra></extra>'
                        ))
                        
                        # Layout identical to arr2plot
                        fig.update_layout(
                            margin=dict(l=0, r=0, t=0, b=10),
                            xaxis_showgrid=False,
                            yaxis_showgrid=False,
                            xaxis=dict(
                                title=dict(text='Range cell', font=dict(size=11, color='#ffffff')),
                                showticklabels=True,
                                tickmode='array',
                                tickvals=range_ticks,
                                ticktext=[str(i) for i in range_ticks],
                                tickfont=dict(size=9, color='#ffffff'),
                                side='bottom'
                            ),
                            yaxis=dict(
                                title=dict(text='Doppler cell', font=dict(size=11, color='#ffffff')),
                                showticklabels=True,
                                tickmode='array',
                                tickvals=doppler_ticks,
                                ticktext=[str(i) for i in doppler_ticks],
                                tickfont=dict(size=9, color='#ffffff'),
                                autorange='reversed'
                            ),
                            height=280,
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)'
                        )
                        
                        return fig
                    
                    class_name = class_names.get(target_class_idx, f"Class {target_class_idx}")
                    
                    # Return binary-mask figures (only the six model plots are updated)
                    return [
                        create_binary_plot_colored(cached_result['predictions'].get('echmo'), target_class_idx, class_name),
                        create_binary_plot_colored(cached_result['predictions'].get('tmva_net_sv'), target_class_idx, class_name),
                        create_binary_plot_colored(cached_result['predictions'].get('deeplabv3plus'), target_class_idx, class_name),
                        create_binary_plot_colored(cached_result['predictions'].get('unet'), target_class_idx, class_name),
                        create_binary_plot_colored(cached_result['predictions'].get('fcn8s'), target_class_idx, class_name),
                        create_binary_plot_colored(cached_result['predictions'].get('vit'), target_class_idx, class_name),
                        f"✓ Isolating class: {class_name} | Binary mask view",
                        target_class_idx
                    ]
                
                def update_cmp_gt_visibility(has_gt):
                    """Update GT plot visibility"""
                    return gr.update(visible=has_gt)
                
                # ==================== Comparison Tab Event Bindings ====================
                
                cmp_data_source.change(
                    fn=on_cmp_data_source_change,
                    inputs=cmp_data_source,
                    outputs=cmp_sequence
                )
                
                cmp_sequence.change(
                    fn=on_cmp_sequence_change,
                    inputs=[cmp_data_source, cmp_sequence],
                    outputs=cmp_frame
                )
                
                cmp_run_btn.click(
                    fn=run_comparison_processing,
                    inputs=[cmp_data_source, cmp_sequence, cmp_frame],
                    outputs=[
                        cmp_raw_plot, cmp_mtd_plot, cmp_gt_plot, cmp_echmo_plot, cmp_tmva_plot,
                        cmp_deeplab_plot, cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                        cmp_status, cmp_has_gt_mask, cmp_last_result, cmp_selected_class
                    ]
                )
                
                # 2D/3D View Toggle Event Bindings for Tab 2
                cmp_raw_view_2d.click(fn=toggle_cmp_raw_view_2d, inputs=[], outputs=[cmp_raw_plot, cmp_raw_view_2d, cmp_raw_view_3d])
                cmp_raw_view_3d.click(fn=toggle_cmp_raw_view_3d, inputs=[], outputs=[cmp_raw_plot, cmp_raw_view_2d, cmp_raw_view_3d])
                
                cmp_mtd_view_2d.click(fn=toggle_cmp_mtd_view_2d, inputs=[], outputs=[cmp_mtd_plot, cmp_mtd_view_2d, cmp_mtd_view_3d])
                cmp_mtd_view_3d.click(fn=toggle_cmp_mtd_view_3d, inputs=[], outputs=[cmp_mtd_plot, cmp_mtd_view_2d, cmp_mtd_view_3d])
                
                # Class-filter button events - plain Gradio Buttons
                def on_class_btn_click(btn_idx, current_selected, cached_result):
                    """Handle a class-button click"""
                    # Clicking the selected button again clears the selection (show all)
                    if current_selected == btn_idx:
                        return apply_class_filter("-1", cached_result)
                    return apply_class_filter(str(btn_idx), cached_result)
                
                # Bind the individual class buttons
                cmp_bg_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(0, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                cmp_target_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(1, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                cmp_dftj_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(2, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                cmp_isrj_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(3, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                cmp_sij_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(4, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                cmp_wj_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(5, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                cmp_nj_btn.click(
                    fn=lambda sel, cache: on_class_btn_click(6, sel, cache),
                    inputs=[cmp_selected_class, cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                
                # Reset button
                cmp_reset_btn.click(
                    fn=lambda cache: apply_class_filter("-1", cache),
                    inputs=[cmp_last_result],
                    outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                            cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                            cmp_status, cmp_selected_class]
                )
                
                cmp_has_gt_mask.change(
                    fn=update_cmp_gt_visibility,
                    inputs=cmp_has_gt_mask,
                    outputs=cmp_gt_plot
                )
                
                # Auto-load on start for comparison tab
                def cmp_auto_load():
                    if not available_datasets:
                        return [gr.update(choices=[], value=None)] * 2 + [go.Figure()] * 9 + ["No datasets", False, None, -1]
                    
                    dataset = available_datasets[0]
                    sequences = get_sequences(dataset)
                    if not sequences:
                        return [gr.update(choices=[], value=None), gr.update(choices=[], value=None)] + [go.Figure()] * 9 + ["No sequences", False, None, -1]
                    
                    seq = sequences[0]
                    frames = get_frames(dataset, seq)
                    if not frames:
                        return [gr.update(choices=sequences, value=seq), gr.update(choices=[], value=None)] + [go.Figure()] * 9 + ["No frames", False, None, -1]
                    
                    frame = frames[0]
                    result, msg = run_comparison_inference(dataset, seq, frame)
                    
                    if result is None:
                        return [gr.update(choices=sequences, value=seq), gr.update(choices=frames, value=frame)] + [go.Figure()] * 9 + [msg, False, None, -1]
                    
                    n_classes = result.get('n_classes', 2)
                    has_gt = result['gt_mask'] is not None
                    display_msg = f"✓ Compared 6 models on {dataset}/{seq}/{frame}"
                    
                    # Cache raw data for 2D/3D view toggle
                    _view_cache.cache_data('cmp_raw', result['raw_echo'])
                    _view_cache.cache_data('cmp_mtd', result['mtd'])
                    
                    plots = [
                        gr.update(choices=sequences, value=seq),
                        gr.update(choices=frames, value=frame),
                        get_plot_for_chart('cmp_raw', result['raw_echo']),  # Support 2D/3D
                        get_plot_for_chart('cmp_mtd', result['mtd']),  # Support 2D/3D
                        arr2plot(result['gt_mask'], is_mask=True, n_classes=n_classes) if result['gt_mask'] is not None else go.Figure(),  # Mask - 2D only
                        arr2plot(result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if result['predictions'].get('echmo') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if result['predictions'].get('unet') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if result['predictions'].get('fcn8s') is not None else go.Figure(),
                        arr2plot(result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if result['predictions'].get('vit') is not None else go.Figure(),
                        display_msg,
                        has_gt,
                        result,  # cached result
                        -1       # Nothing selected by default
                    ]
                    return plots
                
                demo.load(
                    fn=cmp_auto_load,
                    inputs=None,
                    outputs=[
                        cmp_sequence, cmp_frame,
                        cmp_raw_plot, cmp_mtd_plot, cmp_gt_plot, cmp_echmo_plot, cmp_tmva_plot,
                        cmp_deeplab_plot, cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                        cmp_status, cmp_has_gt_mask, cmp_last_result, cmp_selected_class
                    ]
                )
        
        # Close the tabs-wrapper div
        gr.HTML('</div>')
    
    return demo


# ==================== Main Entry ====================
if __name__ == "__main__":
    import sys
    
    # Get port from environment variable (for Hugging Face Spaces compatibility)
    # port = int(os.environ.get("PORT", 7860))
    port = int(os.environ.get("PORT", 8080))
    
    print("=" * 60, file=sys.stderr)
    print("  End-to-End Radar Echo Segmentation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Server: http://0.0.0.0:{port}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        head=LOADING_SCREEN_JS + VIEW_TOGGLE_JS + UI_HELPERS_JS + build_class_availability_js(),
        max_file_size="50mb"
    )
