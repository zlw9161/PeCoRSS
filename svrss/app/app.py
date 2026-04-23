"""
End-to-End Radar Echo Segmentation Visualization
"""

import os
import json
import numpy as np

# Force early pandas import and initialization to avoid circular import during Plotly serialization
import pandas
pandas.Series  # Force attribute access to ensure complete initialization

import torch
import gradio as gr
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from matplotlib.colors import ListedColormap
from scipy.io import loadmat
import io
import base64
from PIL import Image
import plotly.graph_objs as go
import matplotlib.colors as mcolors
from collections import OrderedDict
import traceback

# Import all models for comparison
from models.echmo import EchMo
from models.unet import UNet
from models.deeplabv3plus import DeepLabV3Plus_Res101, DeepLabV3
from models.fcn8s import FCN8s
from models.tmva_net_sv import TMVANet_SV
from models.vit import ViT


# =============================================================================
# Section 1: Configuration & Constants
# =============================================================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


MODEL_PATHS = {
    'NuaaSim': {
        'echmo': ('echmo_e300_lr0.0001_ce_3', 'echmo.pt'),
        'unet': ('unet_e300_lr0.0001_ce_0', 'unet.pt'),
        'deeplabv3plus': ('deeplabv3plus_e300_lr0.0001_ce_1', 'deeplabv3plus.pt'),
        'fcn8s': ('fcn8s_e300_lr0.0001_ce_1', 'fcn8s.pt'),
        'tmva_net_sv': ('tmva_net_sv_e300_lr0.0001_ce_0', 'tmva_net_sv.pt'),
        'vit': ('vit_e300_lr0.0001_ce_0', 'vit.pt'),
    },
    'LSFJamming': {
        'echmo': ('echmo_e300_lr0.0001_lsf_wce_0', 'echmo.pt'),
        'unet': ('unet_e300_lr0.0001_lsf_wce_0', 'unet.pt'),
        'deeplabv3plus': ('deeplabv3plus_e300_lr0.0001_lsf_wce_4', 'deeplabv3plus.pt'),
        'fcn8s': ('fcn8s_e300_lr0.0001_lsf_wce_2', 'fcn8s.pt'),
        'tmva_net_sv': ('tmva_net_sv_e300_lr0.0001_lsf_wce_0', 'tmva_net_sv.pt'),
        'vit': ('vit_e300_lr0.0001_lsf_wce_0', 'vit.pt'),
    }
}

DEFAULT_MODEL = 'echmo'

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

SEG_COLORS_2 = [
    "#440154",     # Background
    "#FDE725",     # Main target
]

SEG_COLORS_9 = [
    "#440154",     # 0. Background
    "#FDE725",     # 1. Main target
    "#B71C1C",     # 2. Dark red
    "#E53935",     # 3. Red
    "#EF9A9A",     # 4. Light red
    "#2563EB",     # 5. Dodger blue
    "#7DD3FC",     # 6. Light sky blue
    "#45CA77",     # 7. Grass green
    "#84E354"      # 8. Yellow green
]


# =============================================================================
# Section 2: AppState (Global State Singleton)
# =============================================================================

class ChartViewCache:
    """Manage 2D/3D view states for charts"""
    def __init__(self):
        self._view_states = {}
        self._data_cache = {}

    def get_view_mode(self, chart_key):
        return self._view_states.get(chart_key, '2d')

    def set_view_mode(self, chart_key, mode):
        self._view_states[chart_key] = mode

    def cache_data(self, chart_key, data):
        if data is not None:
            self._data_cache[chart_key] = data

    def get_data(self, chart_key):
        return self._data_cache.get(chart_key)

    def toggle_view(self, chart_key):
        current = self.get_view_mode(chart_key)
        new_mode = '3d' if current == '2d' else '2d'
        self.set_view_mode(chart_key, new_mode)
        return new_mode

    def clear(self):
        self._view_states.clear()
        self._data_cache.clear()


class AppState:
    """Global application state singleton — centralized management of model_cache, results, view_cache"""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.model_cache = {}
            cls._instance.global_results = {}
            cls._instance.view_cache = ChartViewCache()
        return cls._instance

    def clear(self):
        self.model_cache.clear()
        self.global_results.clear()
        self.view_cache.clear()



# =============================================================================
# Section 3: DataLoader — dataset/sequence/frame loading and file upload handling
# =============================================================================

class DataLoader:
    """Data loading responsibility class"""

    @staticmethod
    def load_dataset_json(dataset_name):
        json_path = f"datasets/{dataset_name}/light_dataset_frame_oriented.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}

    @staticmethod
    def load_echo_data(dataset_name, seq_name, frame_name):
        echo_path = f"datasets/{dataset_name}/{seq_name}/range_doppler_echo/{frame_name}.npy"
        if os.path.exists(echo_path):
            return np.load(echo_path)
        return None

    @staticmethod
    def load_annotation(dataset_name, seq_name, frame_name):
        annot_path = f"datasets/{dataset_name}/{seq_name}/annotations/dense/{frame_name}/range_doppler.npy"
        if os.path.exists(annot_path):
            return np.load(annot_path)
        return None

    @staticmethod
    def load_transmit_signal(dataset_path, dataset_name):
        tx_path = os.path.join(dataset_path, 'coefs', 's_tx.mat')
        if not os.path.exists(tx_path):
            raise FileNotFoundError(f"Transmit signal not found: {tx_path}")
        mat_data = loadmat(tx_path)
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

    @staticmethod
    def get_available_datasets():
        datasets = []
        for dataset_name in ['NuaaSim', 'LSFJamming']:
            json_path = f"datasets/{dataset_name}/light_dataset_frame_oriented.json"
            if os.path.exists(json_path):
                datasets.append(dataset_name)
        return datasets

    @staticmethod
    def get_sequences(dataset_name):
        json_path = f"datasets/{dataset_name}/light_dataset_frame_oriented.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return list(data.keys())
        return []

    @staticmethod
    def get_frames(dataset_name, seq_name):
        if not seq_name:
            return []
        json_path = f"datasets/{dataset_name}/light_dataset_frame_oriented.json"
        if os.path.exists(json_path):
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data.get(seq_name, [])
        return []


# =============================================================================
# Section 4: ModelManager — model loading, caching, and inference
# =============================================================================

class ModelManager:
    """Model management responsibility class"""

    def __init__(self):
        self.app_state = AppState()

    def load_model(self, dataset_name, model_type=DEFAULT_MODEL):
        if dataset_name not in MODEL_PATHS:
            raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(MODEL_PATHS.keys())}")
        if model_type not in MODEL_PATHS[dataset_name]:
            raise ValueError(f"Model type '{model_type}' not found for dataset '{dataset_name}'")

        model_folder, model_filename = MODEL_PATHS[dataset_name][model_type]
        model_path = os.path.join('models', dataset_name, model_folder, 'results', model_filename)
        config_path = os.path.join('models', dataset_name, model_folder, 'config.json')
        dataset_path = os.path.join('datasets', dataset_name)

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, 'r') as f:
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

    def load_comparison_models(self, dataset_name):
        models_dict = {}
        model_configs = {
            'echmo': {'class': EchMo, 'extra_args': True},
            'unet': {'class': UNet, 'extra_args': False},
            'deeplabv3plus': {'class': DeepLabV3Plus_Res101, 'extra_args': False},
            'fcn8s': {'class': FCN8s, 'extra_args': False},
            'tmva_net_sv': {'class': TMVANet_SV, 'extra_args': True},
            'vit': {'class': ViT, 'extra_args': False},
        }

        for model_name, (folder, filename) in MODEL_PATHS[dataset_name].items():
            config_path = os.path.join('models', dataset_name, folder, 'config.json')
            model_path = os.path.join('models', dataset_name, folder, 'results', filename)

            if not os.path.exists(config_path) or not os.path.exists(model_path):
                print(f"Warning: {model_name} model not found, skipping...")
                continue

            with open(config_path, 'r') as f:
                cfg = json.load(f)

            model_cfg = model_configs[model_name]
            if model_name == 'echmo':
                class EchMo_CFG:
                    pass
                m_cfg = EchMo_CFG()
                m_cfg.task_name = cfg.get('task_name', 'Segmentation')
                m_cfg.coef_path = 'coefs'
                m_cfg.dataset_path = os.path.join('datasets', dataset_name)
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
                    n_frames=1,
                    in_ch=1,
                    w=cfg['w_size'],
                    h=cfg['h_size']
                )
            elif model_name == 'vit':
                model = model_cfg['class'](
                    n_classes=cfg['nb_classes'],
                    in_chans=1
                )
            else:
                model = model_cfg['class'](
                    n_classes=cfg['nb_classes'],
                    n_frames=1
                )

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

    def process_with_model(self, model, echo_data):
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

    def get_or_load_model(self, dataset_name):
        """Retrieve or load model from cache"""
        model_key = dataset_name
        if model_key not in self.app_state.model_cache:
            model, config = self.load_model(dataset_name)
            self.app_state.model_cache[model_key] = (model, config)
        return self.app_state.model_cache[model_key]


# =============================================================================
# Section 5: SignalProcessor — traditional signal processing (LMF/MTI/MTD/CFAR)
# =============================================================================

class SignalProcessor:
    """Traditional signal processing responsibility class"""

    @staticmethod
    def traditional_lmf_mti_mtd(raw_data, dataset_name=None, dataset_path=None, tx_signal=None):
        if raw_data.shape[0] == 2:
            raw_cmplx = raw_data[0] + 1j * raw_data[1]
        else:
            raw_cmplx = raw_data

        signal_len = raw_cmplx.shape[1]

        if tx_signal is not None:
            s_tx = tx_signal
            print(f"[DEBUG] Using uploaded transmit signal shape={s_tx.shape}")
        elif dataset_path is not None and dataset_name is not None:
            s_tx = DataLoader.load_transmit_signal(dataset_path, dataset_name)
        else:
            raise ValueError("Either tx_signal or (dataset_path and dataset_name) must be provided")

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

        mf_padded = np.zeros(signal_len, dtype=np.complex64)
        mf_padded[:N_mf] = mf
        mf_fft = np.conj(np.fft.fft(mf_padded))

        fft_raw = np.fft.fft(raw_cmplx, axis=1)
        lmf_fft = fft_raw * mf_fft
        lmf_data = np.fft.ifft(lmf_fft, axis=1)

        print(f"[DEBUG] LMF output max at: {np.unravel_index(np.argmax(np.abs(lmf_data)), lmf_data.shape)}")

        mti_data = np.zeros_like(lmf_data)
        mti_data[1:] = lmf_data[1:] - lmf_data[:-1]

        window = np.hamming(raw_cmplx.shape[0]).reshape(-1, 1)
        mti_windowed = mti_data * window

        if dataset_name == "NuaaSim":
            mtd_data = np.fft.fft(mti_windowed, axis=0)
            print(f"[DEBUG] Dataset=nuaasim: MTD only FFT (no fftshift)")
        else:
            mtd_data = np.fft.fftshift(np.fft.fft(mti_windowed, axis=0), axes=0)
            print(f"[DEBUG] Dataset={dataset_name}: MTD = FFT + fftshift")

        cfar_mask = SignalProcessor.ca_cfar_2d(mtd_data, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-4)
        print(f"[DEBUG] CFAR detected {np.sum(cfar_mask)} targets")

        return lmf_data, mti_data, mtd_data, cfar_mask

    @staticmethod
    def ca_cfar_2d(mtd_data, guard_cells=(2, 2), training_cells=(4, 4), pfa=1e-4):
        data = np.abs(mtd_data)
        doppler_bins, range_bins = data.shape

        train_doppler, train_range = training_cells
        guard_doppler, guard_range = guard_cells

        N = (2 * train_doppler + 2 * guard_doppler + 1) * (2 * train_range + 2 * guard_range + 1) - \
            (2 * guard_doppler + 1) * (2 * guard_range + 1)

        alpha = N * (pfa ** (-1 / N) - 1)
        cfar_mask = np.zeros_like(data, dtype=bool)

        for d in range(doppler_bins):
            for r in range(range_bins):
                d_start = max(0, d - train_doppler - guard_doppler)
                d_end = min(doppler_bins, d + train_doppler + guard_doppler + 1)
                r_start = max(0, r - train_range - guard_range)
                r_end = min(range_bins, r + train_range + guard_range + 1)

                window = data[d_start:d_end, r_start:r_end].copy()

                gd_start = max(0, d - guard_doppler) - d_start
                gd_end = min(doppler_bins, d + guard_doppler + 1) - d_start
                gr_start = max(0, r - guard_range) - r_start
                gr_end = min(range_bins, r + guard_range + 1) - r_start

                if gd_end > gd_start and gr_end > gr_start:
                    window[gd_start:gd_end, gr_start:gr_end] = 0

                train_sum = np.sum(window)
                train_count = np.count_nonzero(window)

                if train_count > 0:
                    threshold = alpha * train_sum / train_count
                    if data[d, r] > threshold:
                        cfar_mask[d, r] = True

        return cfar_mask

    @staticmethod
    def os_cfar_2d(mtd_data, guard_cells=(2, 2), training_cells=(4, 4), rank=0.75, pfa=1e-4):
        data = np.abs(mtd_data)
        doppler_bins, range_bins = data.shape

        train_doppler, train_range = training_cells
        guard_doppler, guard_range = guard_cells

        cfar_mask = np.zeros_like(data, dtype=bool)

        for d in range(doppler_bins):
            for r in range(range_bins):
                d_start = max(0, d - train_doppler - guard_doppler)
                d_end = min(doppler_bins, d + train_doppler + guard_doppler + 1)
                r_start = max(0, r - train_range - guard_range)
                r_end = min(range_bins, r + train_range + guard_range + 1)

                window = data[d_start:d_end, r_start:r_end].copy()

                gd_start = max(0, d - guard_doppler) - d_start
                gd_end = min(doppler_bins, d + guard_doppler + 1) - d_start
                gr_start = max(0, r - guard_range) - r_start
                gr_end = min(range_bins, r + guard_range + 1) - r_start

                if gd_end > gd_start and gr_end > gr_start:
                    window[gd_start:gd_end, gr_start:gr_end] = np.nan

                train_values = window[~np.isnan(window)]

                if len(train_values) > 0:
                    sorted_values = np.sort(train_values)
                    k = int(rank * len(sorted_values))
                    threshold = sorted_values[min(k, len(sorted_values) - 1)]
                    N = len(train_values)
                    alpha = N * (pfa ** (-1 / N) - 1)
                    threshold = alpha * threshold

                    if data[d, r] > threshold:
                        cfar_mask[d, r] = True

        return cfar_mask


# =============================================================================
# Section 6: Visualizer — arr2plot/arr2plot_3d/overlay/animation
# =============================================================================

class Visualizer:
    """Visualization responsibility class"""

    @staticmethod
    def create_figure_no_colorbar(data, title, cmap='viridis', mask_mode=False, n_classes=2):
        fig, ax = plt.subplots(figsize=(10, 4), facecolor=COLORS['background'])
        ax.set_facecolor(COLORS['background'])

        vmin, vmax = np.percentile(data, (1, 99))

        if mask_mode and n_classes <= 9:
            if n_classes == 2:
                hex_colors = SEG_COLORS_2
            else:
                hex_colors = SEG_COLORS_9[:n_classes]

            colors = [mcolors.to_rgb(c) for c in hex_colors]
            cmap_obj = ListedColormap(colors)
            im = ax.imshow(data, cmap=cmap_obj, aspect='auto', vmin=0, vmax=n_classes - 1, interpolation='none')
        else:
            im = ax.imshow(data, cmap='viridis', aspect='auto', interpolation='nearest', vmin=vmin, vmax=vmax)

        ax.set_title(title, fontsize=10, color=COLORS['secondary'], fontweight='600', pad=8)
        ax.tick_params(axis='both', labelsize=7, colors=COLORS['text_dim'])

        for spine in ax.spines.values():
            spine.set_color(COLORS['border'])
            spine.set_linewidth(0.5)

        plt.tight_layout()
        return fig

    @staticmethod
    def create_overlay_figure(echmo_mask, cfar_mask, alpha=0.5, n_classes=2, scan_x=None):
        if echmo_mask.ndim == 3:
            echmo_mask = echmo_mask[0] if echmo_mask.shape[0] == 1 else echmo_mask.argmax(axis=0)
        if cfar_mask.ndim == 3:
            cfar_mask = cfar_mask[0]

        echmo_mask = np.asarray(echmo_mask, dtype=np.float32)
        cfar_mask = np.asarray(cfar_mask, dtype=bool)
        doppler_cells, range_cells = echmo_mask.shape

        fig = go.Figure()

        if n_classes == 2:
            hex_colors = SEG_COLORS_2
        else:
            hex_colors = SEG_COLORS_9[:n_classes]

        cfar_vis = np.zeros((doppler_cells, range_cells), dtype=np.float32)
        if np.sum(cfar_mask) > 0:
            y_coords, x_coords = np.where(cfar_mask)
            for y, x in zip(y_coords, x_coords):
                y0, y1 = max(0, y - 2), min(doppler_cells - 1, y + 2)
                x0, x1 = max(0, x - 2), min(range_cells - 1, x + 2)
                cfar_vis[y0:y1 + 1, x0:x1 + 1] = 1.0

        fig.add_trace(go.Heatmap(
            z=cfar_vis.tolist(),
            colorscale=[[0, SEG_COLORS_2[0]], [1, '#45CA77']],
            showscale=False,
            hoverinfo='skip',
            name='CFAR Detection'
        ))

        z_echmo = echmo_mask.astype(np.float32).tolist()
        colorscale_echmo = [[float(i) / (n_classes - 1), hex_colors[i]] for i in range(n_classes)]

        fig.add_trace(go.Heatmap(
            z=z_echmo,
            colorscale=colorscale_echmo,
            showscale=False,
            opacity=alpha,
            hoverinfo='skip',
            name='EchMo Prediction'
        ))

        if scan_x is not None:
            fig.add_vline(
                x=scan_x,
                line=dict(color='#00ccff', width=3, dash='solid'),
                opacity=0.9
            )
            fig.add_vline(
                x=scan_x,
                line=dict(color='#00ccff', width=12),
                opacity=0.3
            )

        range_step = max(1, range_cells // 8)
        doppler_step = max(1, doppler_cells // 6)
        range_ticks = list(range(0, range_cells, range_step))
        doppler_ticks = list(range(0, doppler_cells, doppler_step))

        fig.update_layout(
            xaxis=dict(
                title=dict(text='Range cell', font=dict(size=11, color='#ffffff')),
                tickmode='array',
                tickvals=range_ticks,
                ticktext=[str(i) for i in range_ticks],
                tickfont=dict(size=9, color='#ffffff'),
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False
            ),
            yaxis=dict(
                title=dict(text='Doppler cell', font=dict(size=11, color='#ffffff')),
                tickmode='array',
                tickvals=doppler_ticks,
                ticktext=[str(i) for i in doppler_ticks],
                tickfont=dict(size=9, color='#ffffff'),
                gridcolor='rgba(255,255,255,0.1)',
                zeroline=False,
                autorange='reversed'
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=60, r=20, t=40, b=40),
            height=320,
            dragmode='zoom',
            selectdirection='any'
        )

        return fig

    @staticmethod
    def fig_to_pil(fig):
        buf = io.BytesIO()
        fig.savefig(buf, format='png', facecolor=fig.get_facecolor(),
                    edgecolor='none', bbox_inches='tight', dpi=100)
        buf.seek(0)
        img = Image.open(buf)
        plt.close(fig)
        return img

    @staticmethod
    def tensor_to_numpy(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.detach().cpu().numpy()
        return tensor

    @staticmethod
    def arr2plot(arr, colorscale='Viridis', is_mask=False, title=None, n_classes=2):
        if arr is None:
            return go.Figure()
        if hasattr(arr, 'size') and arr.size == 0:
            return go.Figure()
        if is_mask and arr.ndim == 3:
            arr = arr[0] if arr.shape[0] == 1 else arr.argmax(axis=0)

        if arr.ndim >= 2:
            doppler_cells, range_cells = arr.shape
        else:
            doppler_cells, range_cells = arr.shape[0], 1

        if is_mask:
            if n_classes == 2:
                color_list = SEG_COLORS_2
            else:
                color_list = SEG_COLORS_9

            colorscale = []
            for class_idx in range(n_classes):
                hex_color = color_list[min(class_idx, len(color_list) - 1)]
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

    @staticmethod
    def arr2plot_3d(arr, title=None, colorscale='Viridis'):
        try:
            if arr is None:
                return go.Figure()
            if hasattr(arr, 'size') and arr.size == 0:
                return go.Figure()

            if arr.ndim > 2:
                arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
            if arr.ndim < 2:
                arr = arr.reshape(1, -1)

            doppler_cells, range_cells = arr.shape
            x_coords = np.arange(range_cells)
            y_coords = np.arange(doppler_cells)

            fig = go.Figure(data=[go.Surface(
                x=x_coords,
                y=y_coords,
                z=arr,
                colorscale=colorscale,
                showscale=False,
                hovertemplate='Range: %{x}<br>Doppler: %{y}<br>Amplitude: %{z:.4f}<extra></extra>',
                contours={'x': {'show': False}, 'y': {'show': False}, 'z': {'show': False}}
            )])

            range_step = max(1, range_cells // 6)
            doppler_step = max(1, doppler_cells // 6)

            fig.update_layout(
                margin=dict(l=0, r=0, t=40 if title else 10, b=0),
                scene=dict(
                    xaxis=dict(
                        title=dict(text='Range cell', font=dict(size=10, color='#ffffff')),
                        tickfont=dict(size=9, color='#ffffff'),
                        tickmode='linear', dtick=range_step,
                        gridcolor='rgba(255,255,255,0.1)',
                        showbackground=False
                    ),
                    yaxis=dict(
                        title=dict(text='Doppler cell', font=dict(size=10, color='#ffffff')),
                        tickfont=dict(size=9, color='#ffffff'),
                        tickmode='linear', dtick=doppler_step,
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
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
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
            traceback.print_exc()
            return go.Figure()

    @staticmethod
    def arr2plot_3d_optimized(arr, title=None, colorscale='Viridis', zmin=None, zmax=None):
        try:
            if arr is None:
                return go.Figure()
            if hasattr(arr, 'size') and arr.size == 0:
                return go.Figure()

            if arr.ndim > 2:
                arr = arr[0] if arr.shape[0] == 1 else arr.mean(axis=0)
            if arr.ndim < 2:
                arr = arr.reshape(1, -1)

            doppler_cells, range_cells = arr.shape
            x_coords = np.arange(range_cells)
            y_coords = np.arange(doppler_cells)

            if zmin is None:
                zmin = np.percentile(arr, 0)
            if zmax is None:
                zmax = np.percentile(arr, 100)

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
                    aspectmode='manual',
                    aspectratio=dict(x=2.5, y=1.0, z=0.5)
                ),
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
            print(f"Error in arr2plot_3d_optimized: {e}")
            return go.Figure()

    @staticmethod
    def get_plot_for_chart(chart_key, data, title=None, is_mask=False, n_classes=2):
        try:
            if data is None:
                return go.Figure()
            AppState().view_cache.cache_data(chart_key, data)
            view_mode = AppState().view_cache.get_view_mode(chart_key)
            if view_mode == '3d' and not is_mask:
                return Visualizer.arr2plot_3d_optimized(data, title=title)
            else:
                return Visualizer.arr2plot(data, is_mask=is_mask, title=title, n_classes=n_classes)
        except Exception as e:
            print(f"Error in get_plot_for_chart for {chart_key}: {e}")
            traceback.print_exc()
            return go.Figure()


# =============================================================================
# Section 7: ViewController — 2D/3D view toggle and caching
# =============================================================================

class ViewController:
    """View control responsibility class"""

    def __init__(self):
        self.app_state = AppState()

    def toggle_chart_view(self, chart_key, current_data, title=None, is_mask=False, n_classes=2):
        new_mode = self.app_state.view_cache.toggle_view(chart_key)
        data = current_data if current_data is not None else self.app_state.view_cache.get_data(chart_key)
        if data is None:
            return go.Figure(), new_mode
        if new_mode == '3d' and not is_mask:
            fig = Visualizer.arr2plot_3d(data, title=title)
        else:
            fig = Visualizer.arr2plot(data, is_mask=is_mask, title=title, n_classes=n_classes)
        return fig, new_mode

    def make_view_toggler(self, chart_key):
        """Factory function: generate 2D/3D toggle function pair for specified chart"""
        def toggle_2d():
            try:
                data = self.app_state.view_cache.get_data(chart_key)
                self.app_state.view_cache.set_view_mode(chart_key, '2d')
                plot = Visualizer.get_plot_for_chart(chart_key, data, title=None)
                return (
                    plot,
                    gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"]),
                    gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
                )
            except Exception as e:
                print(f"Error in toggle_2d for {chart_key}: {e}")
                return go.Figure(), gr.update(), gr.update()

        def toggle_3d():
            try:
                data = self.app_state.view_cache.get_data(chart_key)
                self.app_state.view_cache.set_view_mode(chart_key, '3d')
                plot = Visualizer.get_plot_for_chart(chart_key, data, title=None)
                return (
                    plot,
                    gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-left"]),
                    gr.update(elem_classes=["view-toggle-btn-compact", "view-toggle-right", "view-active"])
                )
            except Exception as e:
                print(f"Error in toggle_3d for {chart_key}: {e}")
                return go.Figure(), gr.update(), gr.update()

        return toggle_2d, toggle_3d


# =============================================================================
# Section 8: InferenceService — end-to-end inference business logic
# =============================================================================

class InferenceService:
    """Inference service — integrates model inference and traditional signal processing"""

    def __init__(self):
        self.data_loader = DataLoader()
        self.model_manager = ModelManager()
        self.signal_processor = SignalProcessor()
        self.visualizer = Visualizer()
        self.app_state = AppState()

    def run_inference(self, dataset_name, seq_name, frame_name):
        if not dataset_name or not seq_name or not frame_name:
            msg = "Please select data source, sequence, and frame."
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]

        if dataset_name not in MODEL_PATHS:
            msg = f"Configuration not found for {dataset_name}"
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]

        dataset_path = os.path.join('datasets', dataset_name)
        if not os.path.exists(dataset_path):
            msg = f"Dataset path not found: {dataset_path}"
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]

        try:
            model, config = self.model_manager.get_or_load_model(dataset_name)
        except Exception as e:
            msg = f"Error loading model: {str(e)}"
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]

        n_classes = config.get('nb_classes', 2)

        echo_data = self.data_loader.load_echo_data(dataset_name, seq_name, frame_name)
        if echo_data is None:
            msg = "Echo data not found"
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]

        annotation = self.data_loader.load_annotation(dataset_name, seq_name, frame_name)

        try:
            model_output, lmf_out, mldc_out, gdf_out = self.model_manager.process_with_model(model, echo_data)
        except Exception as e:
            msg = f"Error during model inference: {str(e)}"
            print("[run_inference]", msg)
            return [None] * 10 + [msg, True]

        try:
            trad_lmf, trad_mti, trad_mtd, trad_cfar = self.signal_processor.traditional_lmf_mti_mtd(
                echo_data,
                dataset_name=dataset_name,
                dataset_path=dataset_path
            )
        except Exception as e:
            msg = f"Error during traditional processing: {str(e)}"
            print("[run_inference]", msg)
            traceback.print_exc()
            return [None] * 10 + [msg, True]

        if echo_data.shape[0] == 2:
            echo_amp = np.abs(echo_data[0] + 1j * echo_data[1])
        else:
            echo_amp = np.abs(echo_data)
        if echo_amp.ndim == 3:
            echo_amp = echo_amp[0]

        lmf_batch0 = lmf_out[0]
        lmf_amp = self.visualizer.tensor_to_numpy(torch.abs(lmf_batch0).mean(dim=0))

        trad_lmf_amp = np.abs(trad_lmf)
        if trad_lmf_amp.ndim == 3:
            trad_lmf_amp = trad_lmf_amp[0]

        mldc_batch0 = mldc_out[0]
        mldc_amp = self.visualizer.tensor_to_numpy(torch.abs(mldc_batch0).mean(dim=0))

        trad_mti_amp = np.abs(trad_mti)
        if trad_mti_amp.ndim == 3:
            trad_mti_amp = trad_mti_amp[0]

        gdf_batch0 = gdf_out[0]
        gdf_amp = self.visualizer.tensor_to_numpy(torch.abs(gdf_batch0).mean(dim=0))

        trad_mtd_amp = np.abs(trad_mtd)
        if trad_mtd_amp.ndim == 3:
            trad_mtd_amp = trad_mtd_amp[0]

        pred_mask = self.visualizer.tensor_to_numpy(torch.argmax(model_output, dim=1)[0])

        self.app_state.global_results['seg_data'] = pred_mask
        self.app_state.global_results['seg_n_classes'] = n_classes
        self.app_state.global_results['trad_rd_data'] = trad_mtd_amp
        self.app_state.global_results['cfar_mask'] = trad_cfar

        gt_mask = None
        has_gt = False
        if annotation is not None:
            has_gt = True
            if annotation.ndim == 3:
                gt_mask = annotation.argmax(axis=0)
            else:
                gt_mask = annotation
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]

        status_msg = f"✓ Processed: {dataset_name} / {seq_name} / {frame_name}"
        if not has_gt:
            status_msg += " | Ground truth not found"

        return [
            echo_amp,
            gt_mask,
            pred_mask,
            lmf_amp,
            trad_lmf_amp,
            mldc_amp,
            trad_mti_amp,
            gdf_amp,
            trad_mtd_amp,
            trad_cfar,
            status_msg,
            has_gt
        ]

    def run_comparison_inference(self, dataset_name, seq_name, frame_name):
        if not dataset_name or not seq_name or not frame_name:
            return None, "Please select data source, sequence, and frame."

        echo_data = self.data_loader.load_echo_data(dataset_name, seq_name, frame_name)
        if echo_data is None:
            return None, "Echo data not found"

        annotation = self.data_loader.load_annotation(dataset_name, seq_name, frame_name)

        if echo_data.shape[0] == 2:
            real_part = echo_data[0]
            imag_part = echo_data[1]
            echo_cmplx = real_part + 1j * imag_part
        else:
            echo_cmplx = echo_data

        x_complex = torch.from_numpy(echo_cmplx).unsqueeze(0).unsqueeze(0).to(DEVICE)
        x_complex = torch.complex(x_complex.real.float(), x_complex.imag.float())

        dataset_path = os.path.join('datasets', dataset_name)
        if dataset_name == 'NuaaSim':
            mtd_path = os.path.join(dataset_path, seq_name, 'range_doppler_mtdData_amp', f'{frame_name}.npy')
            if os.path.exists(mtd_path):
                mtd_amp = np.load(mtd_path)
            else:
                mtd_amp = np.abs(echo_cmplx)
        else:
            mtd_path = os.path.join(dataset_path, seq_name, 'range_doppler_MTD', f'{frame_name}.npy')
            if os.path.exists(mtd_path):
                mtd_data = np.load(mtd_path)
                mtd_amp = np.abs(mtd_data)
            else:
                mtd_amp = np.abs(echo_cmplx)

        if mtd_amp.ndim == 3:
            mtd_amp = mtd_amp[0]

        models_dict = self.model_manager.load_comparison_models(dataset_name)

        results = {}
        n_classes = 2

        with torch.no_grad():
            for model_name, (model, cfg) in models_dict.items():
                try:
                    n_classes = cfg.get('nb_classes', 2)

                    if model_name == 'echmo':
                        output, _, _, _ = model(x_complex)
                    elif model_name == 'tmva_net_sv':
                        x_tmva = np.expand_dims(mtd_amp, axis=0)
                        x_tmva = np.expand_dims(x_tmva, axis=0)
                        x_tmva = torch.from_numpy(x_tmva).unsqueeze(0).to(DEVICE).float()
                        output = model(x_tmva)
                    else:
                        x_other = torch.from_numpy(mtd_amp).unsqueeze(0).unsqueeze(0).to(DEVICE).float()
                        output = model(x_other)

                    pred_mask = torch.argmax(output, dim=1)[0].cpu().numpy()
                    results[model_name] = pred_mask
                except Exception as e:
                    print(f"Error running {model_name}: {e}")
                    traceback.print_exc()
                    results[model_name] = None

        echo_amp = np.abs(echo_cmplx)
        if echo_amp.ndim == 3:
            echo_amp = echo_amp[0]

        gt_mask = None
        if annotation is not None:
            if annotation.ndim == 3:
                gt_mask = annotation.argmax(axis=0)
            else:
                gt_mask = annotation
            if gt_mask.ndim == 3:
                gt_mask = gt_mask[0]

        return {
            'raw_echo': echo_amp,
            'mtd': mtd_amp,
            'gt_mask': gt_mask,
            'predictions': results,
            'n_classes': n_classes
        }, f"✓ Compared {len([r for r in results.values() if r is not None])} models on {dataset_name}/{seq_name}/{frame_name}"


# =============================================================================
# Section 9: UIBuilder — Gradio UI assembly and component factory
# =============================================================================

def create_chart_with_toggle(title):
    """Component factory: create charts with 2D/3D toggle buttons (must be called within Gradio Column context)"""
    if title:
        gr.HTML(f'<div class="col-header-sub" style="margin-top: 8px;">{title}</div>')
    with gr.Column(elem_classes="chart-with-toggle"):
        with gr.Row(elem_classes="view-toggle-float"):
            btn_2d = gr.Button("2D", elem_classes=["view-toggle-btn-compact", "view-toggle-left", "view-active"])
            btn_3d = gr.Button("3D", elem_classes=["view-toggle-btn-compact", "view-toggle-right"])
        plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
    return plot, btn_2d, btn_3d


def create_interface():
    available_datasets = DataLoader.get_available_datasets()
    if not available_datasets:
        available_datasets = ['NuaaSim', 'LSFJamming']

    all_data_sources = available_datasets
    inference = InferenceService()
    view_ctl = ViewController()
    visualizer = Visualizer()
    app_state = AppState()

    with gr.Blocks() as demo:
        # Inject JavaScript (view toggle script injected via demo.launch head)
        gr.HTML(CLASS_FILTER_JS)

        # Header
        gr.HTML('''
        <div class="header-container">
            <div class="header-title-wrapper">
                <div class="header-title">End-to-End Radar Echo Segmentation</div>
            </div>
        </div>
        ''')

        # Global style container
        gr.HTML(LEGEND_STYLE_HTML)

        with gr.Tabs():
            # =====================================================================
            # Tab 1: Single Model (EchMo vs Traditional)
            # =====================================================================
            with gr.Tab("⚔️ EchMo vs. Traditional RSP", id="single"):
                # Static legend
                gr.HTML(STATIC_LEGEND_HTML)

                has_gt_mask = gr.State(True)

                # Control Bar
                with gr.Row(variant="panel", elem_classes="control-row"):
                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Data Source</span>')
                        data_source_dropdown = gr.Dropdown(
                            choices=all_data_sources,
                            value=all_data_sources[0] if all_data_sources else None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )

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

                    with gr.Column(scale=0, min_width=120):
                        run_button = gr.Button("Run", variant="primary", elem_classes="run-btn")

                # Dynamic transparency state
                overlay_alpha = gr.State(0.0)
                echmo_mask_cache = gr.State(None)
                cfar_mask_cache = gr.State(None)
                n_classes_cache = gr.State(2)

                # Main Content
                with gr.Row(equal_height=True):
                    # Column 1: Segmentation Results
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">Segmentation Results: Before & After</div>')
                        raw_input_plot, raw_view_2d, raw_view_3d = create_chart_with_toggle("Raw Echo Input")
                        gr.HTML('<div class="col-header-sub">Ground Truth</div>')
                        gt_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"], visible=True)
                        gr.HTML('<div class="col-header-sub">EchMo + CFAR Overlay</div>')
                        pred_mask_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                        timer = gr.Timer(value=0.05, active=True)

                    # Column 2: EchMo Intermediate Features
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">EchMo Intermediate Features</div>')
                        echmo_lmf_plot, echmo_lmf_view_2d, echmo_lmf_view_3d = create_chart_with_toggle("EchMo LMF")
                        echmo_mldc_plot, echmo_mldc_view_2d, echmo_mldc_view_3d = create_chart_with_toggle("EchMo MLDC")
                        echmo_gdf_plot, echmo_gdf_view_2d, echmo_gdf_view_3d = create_chart_with_toggle("EchMo GDF")

                    # Column 3: Traditional RSP
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">Traditional RSP Intermediate Results</div>')
                        trad_lmf_plot, trad_lmf_view_2d, trad_lmf_view_3d = create_chart_with_toggle("Traditional LMF")
                        trad_mti_plot, trad_mti_view_2d, trad_mti_view_3d = create_chart_with_toggle("Traditional MTI")
                        trad_mtd_plot, trad_mtd_view_2d, trad_mtd_view_3d = create_chart_with_toggle("Traditional MTD")

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
                    sequences = DataLoader.get_sequences(data_source)
                    return {
                        sequence_dropdown: gr.update(choices=sequences, value=sequences[0] if sequences else None, visible=True),
                        seq_label: gr.update(value='<span class="dynamic-label">Sequence</span>'),
                        frame_label: gr.update(value='<span class="dynamic-label">Frame</span>')
                    }

                def on_sequence_change(data_source, seq_name):
                    if not seq_name:
                        return gr.update(choices=[], value=None)
                    frames = DataLoader.get_frames(data_source, seq_name)
                    return gr.update(choices=frames, value=frames[0] if frames else None)

                def run_processing(data_source, sequence, frame):
                    try:
                        results = inference.run_inference(data_source, sequence, frame)

                        has_gt = results[11] if len(results) > 11 else False
                        n_classes = app_state.global_results.get('seg_n_classes', 2)

                        gt_fig = visualizer.arr2plot(results[1], is_mask=True, title=None, n_classes=n_classes) if results[1] is not None else go.Figure()

                        echmo_mask = results[2] if len(results) > 2 else None
                        cfar_mask = results[9] if len(results) > 9 else None

                        if echmo_mask is not None and cfar_mask is not None:
                            overlay_fig = visualizer.create_overlay_figure(echmo_mask, cfar_mask, alpha=0.0, n_classes=n_classes)
                        else:
                            overlay_fig = go.Figure()

                        outputs = [
                            visualizer.get_plot_for_chart('raw', results[0], title=None),
                            gt_fig,
                            overlay_fig,
                            visualizer.get_plot_for_chart('echmo_lmf', results[3], title=None),
                            visualizer.get_plot_for_chart('trad_lmf', results[4], title=None),
                            visualizer.get_plot_for_chart('echmo_mldc', results[5], title=None),
                            visualizer.get_plot_for_chart('trad_mti', results[6], title=None),
                            visualizer.get_plot_for_chart('echmo_gdf', results[7], title=None),
                            visualizer.get_plot_for_chart('trad_mtd', results[8], title=None),
                            results[10] if len(results) > 10 else "Error: No status",
                            has_gt,
                            0.0,
                            echmo_mask,
                            cfar_mask,
                            n_classes
                        ]
                        return outputs
                    except Exception as e:
                        error_msg = f"Error in run_processing: {str(e)}"
                        print(error_msg)
                        traceback.print_exc()
                        empty_fig = go.Figure()
                        return [
                            empty_fig, empty_fig, empty_fig, empty_fig, empty_fig,
                            empty_fig, empty_fig, empty_fig, empty_fig,
                            error_msg, False, 0.0, None, None, 2
                        ]

                def update_overlay_animation(alpha, echmo_mask, cfar_mask, n_classes):
                    if echmo_mask is None or cfar_mask is None:
                        return go.Figure(), alpha
                    overlay_fig = visualizer.create_overlay_figure(echmo_mask, cfar_mask, alpha=alpha, n_classes=n_classes)
                    return overlay_fig, alpha

                def update_gt_visibility(has_gt):
                    return gr.update(visible=has_gt)

                def animate_overlay(alpha, echmo_mask, cfar_mask, n_classes):
                    if echmo_mask is None or cfar_mask is None:
                        return go.Figure(), alpha

                    if echmo_mask.ndim == 3:
                        temp_mask = echmo_mask[0] if echmo_mask.shape[0] == 1 else echmo_mask.argmax(axis=0)
                    else:
                        temp_mask = echmo_mask
                    _, range_cells = temp_mask.shape

                    scan_x = None

                    if alpha < 0.433:
                        new_alpha = 0.0
                    elif alpha < 0.500:
                        t = (alpha - 0.433) / 0.067
                        new_alpha = t
                        scan_x = t * range_cells
                    elif alpha < 0.933:
                        new_alpha = 1.0
                    else:
                        t = (alpha - 0.933) / 0.067
                        new_alpha = 1.0 - t
                        scan_x = (1.0 - t) * range_cells

                    next_alpha = (alpha + 0.01667) % 1.0
                    overlay_fig = visualizer.create_overlay_figure(echmo_mask, cfar_mask, alpha=new_alpha, n_classes=n_classes, scan_x=scan_x)
                    return overlay_fig, next_alpha

                # ==================== Event Bindings (Unified) ====================

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

                run_event = run_button.click(
                    fn=run_processing,
                    inputs=[data_source_dropdown, sequence_dropdown, frame_dropdown],
                    outputs=[
                        raw_input_plot, gt_plot, pred_mask_plot,
                        echmo_lmf_plot, trad_lmf_plot, echmo_mldc_plot,
                        trad_mti_plot, echmo_gdf_plot, trad_mtd_plot,
                        status_text, has_gt_mask,
                        overlay_alpha, echmo_mask_cache, cfar_mask_cache, n_classes_cache
                    ]
                )

                timer.tick(
                    fn=animate_overlay,
                    inputs=[overlay_alpha, echmo_mask_cache, cfar_mask_cache, n_classes_cache],
                    outputs=[pred_mask_plot, overlay_alpha]
                )

                has_gt_mask.change(
                    fn=update_gt_visibility,
                    inputs=has_gt_mask,
                    outputs=gt_plot
                )

                # 2D/3D View Toggle Event Bindings — loop binding to eliminate duplicate code
                tab1_charts = [
                    (raw_input_plot, raw_view_2d, raw_view_3d, 'raw'),
                    (echmo_lmf_plot, echmo_lmf_view_2d, echmo_lmf_view_3d, 'echmo_lmf'),
                    (echmo_mldc_plot, echmo_mldc_view_2d, echmo_mldc_view_3d, 'echmo_mldc'),
                    (echmo_gdf_plot, echmo_gdf_view_2d, echmo_gdf_view_3d, 'echmo_gdf'),
                    (trad_lmf_plot, trad_lmf_view_2d, trad_lmf_view_3d, 'trad_lmf'),
                    (trad_mti_plot, trad_mti_view_2d, trad_mti_view_3d, 'trad_mti'),
                    (trad_mtd_plot, trad_mtd_view_2d, trad_mtd_view_3d, 'trad_mtd'),
                ]

                for plot, btn_2d, btn_3d, key in tab1_charts:
                    t2d, t3d = view_ctl.make_view_toggler(key)
                    btn_2d.click(fn=t2d, inputs=[], outputs=[plot, btn_2d, btn_3d])
                    btn_3d.click(fn=t3d, inputs=[], outputs=[plot, btn_2d, btn_3d])

                # Auto-load on start
                def auto_load_on_start():
                    if not available_datasets:
                        return [gr.update()] * 3 + [True] + [go.Figure()] * 9 + ["No datasets", False, 0.0, None, None, 2]

                    dataset = available_datasets[0]
                    sequences = DataLoader.get_sequences(dataset)
                    if not sequences:
                        return [gr.update(choices=sequences), gr.update(visible=True, choices=[], value=None),
                               gr.update(value='<span class="dynamic-label">Sequence</span>'),
                               gr.update(value='<span class="dynamic-label">Frame</span>'),
                               True] + [go.Figure()] * 9 + ["No sequences", False, 0.0, None, None, 2]

                    seq = sequences[0]
                    frames = DataLoader.get_frames(dataset, seq)
                    if not frames:
                        return [gr.update(choices=sequences, value=seq), gr.update(visible=True, choices=[], value=None),
                               gr.update(value='<span class="dynamic-label">Sequence</span>'),
                               gr.update(value='<span class="dynamic-label">Frame</span>'),
                               True] + [go.Figure()] * 9 + ["No frames", False, 0.0, None, None, 2]

                    frame = frames[0]
                    results = inference.run_inference(dataset, seq, frame)
                    n_classes = app_state.global_results.get('seg_n_classes', 2)

                    echmo_mask = results[2] if len(results) > 2 else None
                    cfar_mask = results[9] if len(results) > 9 else None

                    overlay_fig = go.Figure()
                    if echmo_mask is not None and cfar_mask is not None:
                        overlay_fig = visualizer.create_overlay_figure(echmo_mask, cfar_mask, alpha=0.0, n_classes=n_classes)

                    plot_results = [
                        visualizer.get_plot_for_chart('raw', results[0], title=None),
                        visualizer.arr2plot(results[1], is_mask=True, title=None, n_classes=n_classes) if results[1] is not None else go.Figure(),
                        overlay_fig,
                        visualizer.get_plot_for_chart('echmo_lmf', results[3], title=None),
                        visualizer.get_plot_for_chart('trad_lmf', results[4], title=None),
                        visualizer.get_plot_for_chart('echmo_mldc', results[5], title=None),
                        visualizer.get_plot_for_chart('trad_mti', results[6], title=None),
                        visualizer.get_plot_for_chart('echmo_gdf', results[7], title=None),
                        visualizer.get_plot_for_chart('trad_mtd', results[8], title=None),
                    ]

                    has_gt = results[11] if len(results) > 11 else (results[1] is not None)

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
                        raw_input_plot, gt_plot, pred_mask_plot,
                        echmo_lmf_plot, trad_lmf_plot, echmo_mldc_plot,
                        trad_mti_plot, echmo_gdf_plot, trad_mtd_plot,
                        status_text,
                        overlay_alpha, echmo_mask_cache, cfar_mask_cache, n_classes_cache
                    ]
                )

            # =====================================================================
            # Tab 2: Model Comparison
            # =====================================================================
            with gr.Tab("🔥 Model Comparison", id="comparison"):
                cmp_has_gt_mask = gr.State(True)
                cmp_selected_class = gr.State(-1)
                cmp_last_result = gr.State(None)

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
                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Data Source</span>')
                        cmp_data_source = gr.Dropdown(
                            choices=available_datasets,
                            value=available_datasets[0] if available_datasets else None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )

                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Sequence</span>')
                        cmp_sequence = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )

                    with gr.Column(scale=2, elem_classes="control-item"):
                        gr.HTML('<span class="dynamic-label">Frame</span>')
                        cmp_frame = gr.Dropdown(
                            choices=[],
                            value=None,
                            show_label=False,
                            container=False,
                            elem_classes=["dark-dropdown"]
                        )

                    with gr.Column(scale=0, min_width=120):
                        cmp_run_btn = gr.Button("Run", variant="primary", elem_classes="run-btn")

                # Interactive legend bar
                with gr.Row():
                    with gr.Column(scale=0):
                        gr.HTML('''<div style="width:0;"></div>''')
                    with gr.Column(scale=0, min_width=650, elem_classes="legend-btn-container"):
                        with gr.Row(variant="compact", elem_classes="legend-btn-inner"):
                            cmp_bg_btn = gr.Button("BACKGROUND", interactive=True,
                                                   elem_classes=["legend-class-btn", "legend-c0"], min_width=115)
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
                            cmp_reset_btn = gr.Button("⟲", interactive=True,
                                                     elem_classes="legend-reset-btn", min_width=32)

                # Results Area - Row 1
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">Raw Echo Input</div>')
                        cmp_raw_plot, cmp_raw_view_2d, cmp_raw_view_3d = create_chart_with_toggle("")
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">MTD Input</div>')
                        cmp_mtd_plot, cmp_mtd_view_2d, cmp_mtd_view_3d = create_chart_with_toggle("")
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">Ground Truth Mask</div>')
                        cmp_gt_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"], visible=True)

                # Results Area - Row 2
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">EchMo</div>')
                        cmp_echmo_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">TMVA-Net</div>')
                        cmp_tmva_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">DeepLabv3+</div>')
                        cmp_deeplab_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])

                # Results Area - Row 3
                with gr.Row(equal_height=True):
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">U-Net</div>')
                        cmp_unet_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    with gr.Column(scale=1, min_width=320):
                        gr.HTML('<div class="col-header">FCN-8s</div>')
                        cmp_fcn_plot = gr.Plot(label="", show_label=False, container=True, elem_classes=["plot-container"])
                    with gr.Column(scale=1, min_width=320):
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

                def on_cmp_data_source_change(dataset_name):
                    if not dataset_name:
                        return gr.update(choices=[], value=None)
                    sequences = DataLoader.get_sequences(dataset_name)
                    return gr.update(choices=sequences, value=sequences[0] if sequences else None)

                def on_cmp_sequence_change(dataset_name, seq_name):
                    if not dataset_name or not seq_name:
                        return gr.update(choices=[], value=None)
                    valid_sequences = DataLoader.get_sequences(dataset_name)
                    if seq_name not in valid_sequences:
                        if valid_sequences:
                            seq_name = valid_sequences[0]
                        else:
                            return gr.update(choices=[], value=None)
                    frames = DataLoader.get_frames(dataset_name, seq_name)
                    return gr.update(choices=frames, value=frames[0] if frames else None)

                def run_comparison_processing(dataset_name, seq_name, frame_name):
                    result, msg = inference.run_comparison_inference(dataset_name, seq_name, frame_name)

                    if result is None:
                        return [go.Figure()] * 9 + [msg, False, None, -1]

                    n_classes = result.get('n_classes', 2)
                    has_gt = result['gt_mask'] is not None
                    result['dataset_name'] = dataset_name
                    display_msg = f"✓ Compared 6 models on {dataset_name}/{seq_name}/{frame_name}"

                    app_state.view_cache.cache_data('cmp_raw', result['raw_echo'])
                    app_state.view_cache.cache_data('cmp_mtd', result['mtd'])

                    plots = [
                        visualizer.get_plot_for_chart('cmp_raw', result['raw_echo']),
                        visualizer.get_plot_for_chart('cmp_mtd', result['mtd']),
                        visualizer.arr2plot(result['gt_mask'], is_mask=True, n_classes=n_classes) if result['gt_mask'] is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if result['predictions'].get('echmo') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if result['predictions'].get('unet') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if result['predictions'].get('fcn8s') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if result['predictions'].get('vit') is not None else go.Figure(),
                        display_msg,
                        has_gt,
                        result,
                        -1
                    ]
                    return plots

                def apply_class_filter(filter_input, cached_result):
                    try:
                        parts = filter_input.split(',')
                        target_class_idx = int(parts[0]) if parts else -1
                    except Exception:
                        target_class_idx = -1

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

                    dataset_valid_classes = {
                        'NuaaSim': [0, 1],
                        'LSFJamming': [0, 1, 2, 3, 4, 5, 6]
                    }

                    if cached_result is None:
                        return [go.Figure()] * 6 + ["⚠ No data. Please run inference first.", -1]

                    dataset_name = cached_result.get('dataset_name', 'NuaaSim')
                    n_classes = cached_result.get('n_classes', 2)

                    if target_class_idx == -1:
                        return [
                            visualizer.arr2plot(cached_result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('echmo') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('unet') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('fcn8s') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('vit') is not None else go.Figure(),
                            "✓ Restored full segmentation view",
                            -1
                        ]

                    valid_classes = dataset_valid_classes.get(dataset_name, [0, 1])
                    if target_class_idx not in valid_classes:
                        class_name = class_names.get(target_class_idx, f"Class {target_class_idx}")
                        return [
                            visualizer.arr2plot(cached_result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('echmo') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('unet') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('fcn8s') is not None else go.Figure(),
                            visualizer.arr2plot(cached_result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if cached_result['predictions'].get('vit') is not None else go.Figure(),
                            f"⚠ {dataset_name} does not have category '{class_name}'. View unchanged.",
                            -1
                        ]

                    class_colors = {
                        0: '#440154',
                        1: '#FDE725',
                        2: '#B71C1C',
                        3: '#E53935',
                        4: '#EF9A9A',
                        5: '#2563EB',
                        6: '#7DD3FC'
                    }

                    def create_binary_plot_colored(pred_mask, target_idx, class_name):
                        if pred_mask is None:
                            return go.Figure()
                        target_color = class_colors.get(target_idx, '#FDE725')
                        binary_mask = (pred_mask == target_idx).astype(np.uint8)
                        colorscale = [
                            [0, 'rgba(24, 24, 38, 1)'],
                            [1, target_color]
                        ]
                        doppler_cells, range_cells = binary_mask.shape
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

                # 2D/3D View Toggle for Tab 2 — loop binding
                tab2_charts = [
                    (cmp_raw_plot, cmp_raw_view_2d, cmp_raw_view_3d, 'cmp_raw'),
                    (cmp_mtd_plot, cmp_mtd_view_2d, cmp_mtd_view_3d, 'cmp_mtd'),
                ]

                for plot, btn_2d, btn_3d, key in tab2_charts:
                    t2d, t3d = view_ctl.make_view_toggler(key)
                    btn_2d.click(fn=t2d, inputs=[], outputs=[plot, btn_2d, btn_3d])
                    btn_3d.click(fn=t3d, inputs=[], outputs=[plot, btn_2d, btn_3d])

                # Class filter button events
                def on_class_btn_click(btn_idx, current_selected, cached_result):
                    if current_selected == btn_idx:
                        return apply_class_filter("-1", cached_result)
                    return apply_class_filter(str(btn_idx), cached_result)

                class_buttons = [
                    (cmp_bg_btn, 0),
                    (cmp_target_btn, 1),
                    (cmp_dftj_btn, 2),
                    (cmp_isrj_btn, 3),
                    (cmp_sij_btn, 4),
                    (cmp_wj_btn, 5),
                    (cmp_nj_btn, 6),
                ]

                for btn, idx in class_buttons:
                    btn.click(
                        fn=lambda sel, cache, i=idx: on_class_btn_click(i, sel, cache),
                        inputs=[cmp_selected_class, cmp_last_result],
                        outputs=[cmp_echmo_plot, cmp_tmva_plot, cmp_deeplab_plot,
                                cmp_unet_plot, cmp_fcn_plot, cmp_vit_plot,
                                cmp_status, cmp_selected_class]
                    )

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
                    sequences = DataLoader.get_sequences(dataset)
                    if not sequences:
                        return [gr.update(choices=[], value=None), gr.update(choices=[], value=None)] + [go.Figure()] * 9 + ["No sequences", False, None, -1]

                    seq = sequences[0]
                    frames = DataLoader.get_frames(dataset, seq)
                    if not frames:
                        return [gr.update(choices=sequences, value=seq), gr.update(choices=[], value=None)] + [go.Figure()] * 9 + ["No frames", False, None, -1]

                    frame = frames[0]
                    result, msg = inference.run_comparison_inference(dataset, seq, frame)

                    if result is None:
                        return [gr.update(choices=sequences, value=seq), gr.update(choices=frames, value=frame)] + [go.Figure()] * 9 + [msg, False, None, -1]

                    n_classes = result.get('n_classes', 2)
                    has_gt = result['gt_mask'] is not None
                    display_msg = f"✓ Compared 6 models on {dataset}/{seq}/{frame}"

                    app_state.view_cache.cache_data('cmp_raw', result['raw_echo'])
                    app_state.view_cache.cache_data('cmp_mtd', result['mtd'])

                    plots = [
                        gr.update(choices=sequences, value=seq),
                        gr.update(choices=frames, value=frame),
                        visualizer.get_plot_for_chart('cmp_raw', result['raw_echo']),
                        visualizer.get_plot_for_chart('cmp_mtd', result['mtd']),
                        visualizer.arr2plot(result['gt_mask'], is_mask=True, n_classes=n_classes) if result['gt_mask'] is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('echmo'), is_mask=True, n_classes=n_classes) if result['predictions'].get('echmo') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('tmva_net_sv'), is_mask=True, n_classes=n_classes) if result['predictions'].get('tmva_net_sv') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('deeplabv3plus'), is_mask=True, n_classes=n_classes) if result['predictions'].get('deeplabv3plus') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('unet'), is_mask=True, n_classes=n_classes) if result['predictions'].get('unet') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('fcn8s'), is_mask=True, n_classes=n_classes) if result['predictions'].get('fcn8s') is not None else go.Figure(),
                        visualizer.arr2plot(result['predictions'].get('vit'), is_mask=True, n_classes=n_classes) if result['predictions'].get('vit') is not None else go.Figure(),
                        display_msg,
                        has_gt,
                        result,
                        -1
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

        # Close tabs-wrapper div
        gr.HTML('</div>')

    return demo


# =============================================================================
# Section 10: CSS / JS Constants
# =============================================================================

LEGEND_STYLE_HTML = """
<style>
.tabs-wrapper { position: relative; }
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
.legend-btn-container {
    position: absolute !important;
    top: -230px !important;
    right: 0px !important;
    z-index: 1000 !important;
    width: auto !important;
    min-width: 1200px !important;
    max-width: 1200px !important;
}
.legend-btn-inner {
    display: flex !important;
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
.legend-reset-btn {
    width: 28px !important;
    height: 28px !important;
    min-width: 28px !important;
    max-width: 28px !important;
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
.legend-class-btn .lucide-icon {
    display: none !important;
}
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
.legend-c1::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #FDE725; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-c2::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #B71C1C; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-c3::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #E53935; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-c4::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #EF9A9A; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-c5::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #2563EB; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-c6::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #7DD3FC; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-c0::before { content: ""; display: inline-block; width: 10px; height: 10px; background: #440154; border-radius: 2px; margin-right: 5px; vertical-align: middle; border: 1px solid rgba(255,255,255,0.3); }
.legend-class-btn { 
    display: inline-flex !important; 
    align-items: center !important; 
    justify-content: center !important;
    flex-direction: row !important;
    white-space: nowrap !important;
}
.legend-class-btn > div,
.legend-class-btn > div > div { 
    display: inline-flex !important; 
    align-items: center !important; 
    flex-direction: row !important;
    gap: 0 !important;
}
.legend-class-btn::before {
    flex-shrink: 0 !important;
    align-self: center !important;
}
.legend-color-box {
    flex-shrink: 0 !important;
    align-self: center !important;
}
</style>
<div class="tabs-wrapper">
"""

STATIC_LEGEND_HTML = """
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
"""

# =============================================================================
# Section 11: Custom CSS
# =============================================================================

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Exo+2:wght@400;500;600;700&display=swap');

/* ===== File Upload - Compact Style ===== */
.compact-file-upload {
    height: 42px !important;
    min-height: 42px !important;
    max-height: 42px !important;
}
.compact-file-upload .wrap,
.compact-file-upload .file-preview,
.compact-file-upload .file-thumb,
.compact-file-upload .file-info,
.compact-file-upload .icon,
.compact-file-upload svg,
.compact-file-upload .gr-file-preview,
.compact-file-upload [class*="preview"],
.compact-file-upload [class*="thumb"] {
    display: none !important;
    opacity: 0 !important;
    visibility: hidden !important;
    height: 0 !important;
    width: 0 !important;
    overflow: hidden !important;
}
.compact-file-upload .gr-file {
    display: none !important;
}
.compact-file-upload input[type="file"] {
    position: absolute !important;
    top: 0 !important;
    left: 0 !important;
    width: 100% !important;
    height: 100% !important;
    opacity: 0 !important;
    cursor: pointer !important;
    z-index: 100 !important;
}
.compact-file-upload .custom-upload-box {
    position: relative !important;
    height: 42px !important;
    border: 2px dashed #7c3aed !important;
    border-radius: 8px !important;
    background: #181826 !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    color: #94a3b8 !important;
    font-size: 0.9rem !important;
    transition: all 0.3s ease !important;
    overflow: hidden !important;
    white-space: nowrap !important;
    padding: 0 12px !important;
}
.compact-file-upload:hover .custom-upload-box {
    border-color: #FFC40E !important;
    background: #232345 !important;
}
.compact-file-upload.has-file .custom-upload-box {
    border-style: solid !important;
    border-color: #7c3aed !important;
    background: #1a0a2e !important;
    justify-content: flex-start !important;
    padding: 0 36px 0 12px !important;
}
.compact-file-upload .placeholder-text {
    display: block !important;
    pointer-events: none !important;
}
.compact-file-upload.has-file .placeholder-text {
    display: none !important;
}
.compact-file-upload .file-info {
    display: none !important;
    align-items: center !important;
    width: 100% !important;
    height: 100% !important;
    overflow: hidden !important;
}
.compact-file-upload.has-file .file-info {
    display: flex !important;
}
.compact-file-upload .file-name {
    flex: 1 !important;
    white-space: nowrap !important;
    overflow: hidden !important;
    text-overflow: ellipsis !important;
    font-weight: 500 !important;
    color: #1a1a2e !important;
    font-size: 0.85rem !important;
    min-width: 0 !important;
}
.compact-file-upload .file-size {
    margin-left: 8px !important;
    color: #6b7280 !important;
    font-size: 0.75rem !important;
    white-space: nowrap !important;
    flex-shrink: 0 !important;
}
.compact-file-upload .remove-btn {
    position: absolute !important;
    right: 8px !important;
    top: 50% !important;
    transform: translateY(-50%) !important;
    width: 24px !important;
    height: 24px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    background: transparent !important;
    border: none !important;
    color: #6b7280 !important;
    font-size: 1.4rem !important;
    cursor: pointer !important;
    line-height: 1 !important;
    padding: 0 !important;
    font-family: Arial, sans-serif !important;
    pointer-events: auto !important;
    z-index: 101 !important;
    flex-shrink: 0 !important;
}
.compact-file-upload .remove-btn:hover {
    color: #dc2626 !important;
}

/* ===== Splash loading animation ===== */
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
.loading-progress-area {
    width: 400px;
    position: relative;
    z-index: 2147483647;
}
.progress-container {
    width: 100%;
    height: 8px;
    background: rgba(0, 0, 0, 0.4);
    border: 2px solid rgba(255, 255, 255, 0.5);
    border-radius: 10px;
    padding: 2px;
    box-shadow: 0 0 20px rgba(124, 58, 237, 0.3), inset 0 2px 4px rgba(0, 0, 0, 0.5);
    position: relative;
    z-index: 2147483647;
}
.progress-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, #f59e0b 0%, #FFC40E 30%, #fff5a8 50%, #FFC40E 70%, #f59e0b 100%);
    background-size: 200% 100%;
    border-radius: 6px;
    box-shadow: 0 0 10px rgba(255, 196, 14, 0.6), 0 0 20px rgba(255, 196, 14, 0.3);
    position: relative;
    z-index: 2147483647;
    overflow: hidden;
    transition: width 0.1s linear;
}
@keyframes barGlow {
    0%, 100% { box-shadow: 0 0 10px rgba(255, 196, 14, 0.6), 0 0 20px rgba(255, 196, 14, 0.3); }
    50% { box-shadow: 0 0 20px rgba(255, 196, 14, 0.9), 0 0 40px rgba(255, 196, 14, 0.5); }
}
.progress-bar::after {
    content: '';
    position: absolute;
    top: 0;
    left: -50%;
    width: 50%;
    height: 100%;
    background: linear-gradient(90deg, transparent 0%, rgba(255, 255, 255, 0.6) 50%, transparent 100%);
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

/* ===== Global Container ===== */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: linear-gradient(135deg, #1a0a2e 0%, #2a004f 50%, #0d0618 100%) !important;
    color: #e2e8f0 !important;
    min-height: 100vh !important;
}

/* ===== Header ===== */
.header-container {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 20px 30px 15px;
    position: relative;
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
    background: linear-gradient(90deg, #FFC40E 0%, #fff5a8 25%, #FFC40E 50%, #fff5a8 75%, #FFC40E 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    background-clip: text;
    -webkit-text-fill-color: transparent;
    animation: shimmer 3s linear infinite, fadeInDown 1s ease-out;
}
@keyframes shimmer {
    0% { background-position: 200% center; }
    100% { background-position: -200% center; }
}
@keyframes fadeInDown {
    0% { opacity: 0; transform: translateY(-20px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes fadeInUp {
    0% { opacity: 0; transform: translateY(15px); }
    100% { opacity: 1; transform: translateY(0); }
}
@keyframes pulse {
    0%, 100% { opacity: 0.8; }
    50% { opacity: 1; }
}
@media (max-width: 768px) {
    .header-container { flex-direction: column; padding: 15px 20px; }
    .header-title { font-size: 1.4rem; }
}

/* ===== Tab Nav ===== */
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
.gradio-container .tab-nav *,
.gradio-container [class*="tab"] button {
    color: #c4b5fd !important;
}
.gradio-container .tab-nav .selected,
.gradio-container [class*="tab"] [class*="selected"],
.gradio-container [class*="tab"] [aria-selected="true"] {
    color: #FFC40E !important;
}

/* ===== Control Row ===== */
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

/* ===== Dropdown ===== */
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
    min-width: 450px !important;
    max-width: 500px !important;
}
.dark-dropdown:hover {
    border-color: #FFC40E !important;
    box-shadow: 0 0 8px rgba(255, 196, 14, 0.4) !important;
}
.dark-dropdown [role="listbox"] {
    width: 100% !important;
    height: 100% !important;
    background: #1a0a2e !important;
    border: none !important;
    padding: 0 !important;
    margin: 0 !important;
}
[role="listbox"] {
    background: rgba(26, 10, 46, 0.05) !important;
    backdrop-filter: blur(10px); 
    border: none !important;
    border-radius: 6px !important;
    width: auto !important;
    height: auto !important;
}
[role="listbox"] [role="option"] {
    color: #ffffff !important;
}
[role="listbox"] [role="option"]:hover {
    background: #2d1b4e !important;
    color: #FFC40E !important;
}
[role="listbox"] [aria-selected="true"] {
    background: rgba(26, 10, 46, 0.55) !important;
    color: #FFC40E !important;
}

/* ===== Hide loading indicators ===== */
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

/* ===== Run Button ===== */
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

/* ===== Column Header ===== */
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
    .control-row { flex-wrap: wrap !important; }
    .control-item { min-width: 200px !important; }
}

/* ===== 3D View Toggle System ===== */
.chart-with-toggle {
    position: relative !important;
    margin-bottom: 8px !important;
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    overflow: visible !important;
    max-width: 100% !important;
    height: 280px !important;
    min-height: 280px !important;
}
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
.plot-container .js-plotly-plot,
.plot-container .plotly,
.plot-container > div {
    margin: 0 auto !important;
    max-width: 100% !important;
}
.plot-container svg,
.plot-container .main-svg {
    max-width: 100% !important;
    height: auto !important;
}
.view-toggle-float {
    position: absolute !important;
    top: 4px !important;
    right: auto !important;
    left: 50px !important;
    z-index: 100 !important;
    height: 28px !important;
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
.view-toggle-float > div,
.view-toggle-float .flex,
.view-toggle-float [class*="flex"] {
    width: auto !important;
    min-width: auto !important;
    flex: 0 0 auto !important;
    display: flex !important;
    flex-direction: row !important;
}
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
.view-toggle-btn-compact.view-toggle-left,
.view-toggle-btn-compact.view-toggle-left > button {
    border-radius: 2px 0 0 2px !important;
    border-right: 1px solid rgba(124, 58, 237, 0.2) !important;
}
.view-toggle-btn-compact.view-toggle-right,
.view-toggle-btn-compact.view-toggle-right > button {
    border-radius: 0 2px 2px 0 !important;
}
.view-toggle-btn-compact:hover,
.view-toggle-btn-compact:hover > button,
.view-toggle-btn-compact > button:hover {
    color: #ffffff !important;
    background: rgba(124, 58, 237, 0.2) !important;
}
.view-toggle-btn-compact:not(.view-active),
.view-toggle-btn-compact:not(.view-active) > button,
.view-toggle-btn-compact:not(.view-active) > button:active,
.view-toggle-float .view-toggle-btn-compact:not(.view-active) {
    background: transparent !important;
    color: rgba(255, 255, 255, 0.6) !important;
    border-color: rgba(124, 58, 237, 0.3) !important;
}
.view-toggle-btn-compact.view-active,
.view-toggle-btn-compact.view-active > button,
.view-toggle-btn-compact.view-active > button:active,
.view-toggle-float .view-toggle-btn-compact.view-active {
    background: #FFC40E !important;
    color: #1a0a2e !important;
    border-color: #FFC40E !important;
}
.col-header-sub {
    font-size: 0.8rem !important;
    color: #FFC40E !important;
    font-weight: 500 !important;
    padding: 4px 8px !important;
    margin: 4px 0 !important;
    background: transparent !important;
    border: none !important;
    text-align: center !important;
    height: 24px !important;
    line-height: 16px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
}
.plot-container {
    position: relative !important;
    border: none !important;
    box-shadow: none !important;
    background: transparent !important;
}
.chart-3d-active {
    box-shadow: 0 0 0 1px rgba(255, 196, 14, 0.4), 0 0 15px rgba(255, 196, 14, 0.15) !important;
    border-radius: 6px !important;
}
.equal-height > div,
[equal_height="true"] > div {
    display: flex !important;
    flex-direction: column !important;
}
.equal-height .col-header-sub:first-of-type {
    margin-top: 8px !important;
}
"""


# =============================================================================
# Section 12: JavaScript Constants
# =============================================================================

VIEW_TOGGLE_JS = """
<script>
(function() {
    window.ChartViewManager = {
        viewStates: {},
        getView: function(chartKey) { return this.viewStates[chartKey] || '2d'; },
        setView: function(chartKey, mode) {
            this.viewStates[chartKey] = mode;
            this.updateButtonState(chartKey, mode);
        },
        toggleView: function(chartKey) {
            var current = this.getView(chartKey);
            var newMode = current === '2d' ? '3d' : '2d';
            this.setView(chartKey, newMode);
            return newMode;
        },
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
    window.toggleView = function(chartKey, mode) {
        window.ChartViewManager.setView(chartKey, mode);
        var eventName = 'view_toggle_' + chartKey;
        if (window.gradioApp && window.gradioApp.submit) {
            window.gradioApp.submit(eventName, mode);
        }
    };
    function initViewStates() {
        try {
            var saved = localStorage.getItem('chartViewStates');
            if (saved) { window.ChartViewManager.viewStates = JSON.parse(saved); }
        } catch(e) { console.log('No saved view states'); }
    }
    function saveViewStates() {
        try { localStorage.setItem('chartViewStates', JSON.stringify(window.ChartViewManager.viewStates)); }
        catch(e) { console.log('Failed to save view states'); }
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initViewStates);
    } else { initViewStates(); }
    window.addEventListener('beforeunload', saveViewStates);

    function setup3DResetHandler() {
        document.addEventListener('dblclick', function(e) {
            var plotlyDiv = e.target.closest('.js-plotly-plot');
            if (plotlyDiv && plotlyDiv.data && plotlyDiv.data[0] && plotlyDiv.data[0].type === 'surface') {
                Plotly.relayout(plotlyDiv, {
                    'scene.camera': { eye: {x: 0, y: -2.2, z: 1.3}, up: {x: 0, y: 0, z: 1}, center: {x: 0, y: 0, z: 0} }
                });
            }
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setup3DResetHandler);
    } else { setup3DResetHandler(); }

    function center3DPlots() {
        var plots = document.querySelectorAll('.js-plotly-plot');
        plots.forEach(function(plot) {
            if (plot.data && plot.data[0] && plot.data[0].type === 'surface') {
                var container = plot.closest('.chart-with-toggle') || plot.parentElement;
                if (container) {
                    container.style.display = 'flex';
                    container.style.justifyContent = 'center';
                    container.style.alignItems = 'center';
                    plot.style.margin = '0 auto';
                }
            }
        });
    }
    var plotObserver = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) {
                    if (node.classList && node.classList.contains('js-plotly-plot')) {
                        setTimeout(center3DPlots, 100);
                    }
                    var plots = node.querySelectorAll ? node.querySelectorAll('.js-plotly-plot') : [];
                    if (plots.length > 0) { setTimeout(center3DPlots, 100); }
                }
            });
        });
    });
    if (document.body) {
        plotObserver.observe(document.body, { childList: true, subtree: true });
        setTimeout(center3DPlots, 500);
    } else {
        document.addEventListener('DOMContentLoaded', function() {
            plotObserver.observe(document.body, { childList: true, subtree: true });
            setTimeout(center3DPlots, 500);
        });
    }
    window.addEventListener('resize', center3DPlots);
})();
</script>
"""

LOADING_SCREEN_JS = """
<script>
(function() {
    window.AppLoader = {
        screen: null,
        bar: null,
        status: null,
        percent: null,
        progress: 0,
        targetProgress: 0,
        isComplete: false,
        loadStartTime: Date.now(),
        minDisplayTime: 2500,
        create: function() {
            if (document.getElementById('loading-screen')) return;
            this.screen = document.createElement('div');
            this.screen.id = 'loading-screen';
            this.screen.style.cssText = 'position:fixed;top:0;left:0;width:100vw;height:100vh;background:#1a0a2e;display:flex;flex-direction:column;justify-content:center;align-items:center;z-index:2147483647;transition:opacity 0.5s ease-out;';
            var title = document.createElement('div');
            title.textContent = 'End-to-End Radar Echo Segmentation';
            title.style.cssText = 'color:#FFC40E;font-size:2.2rem;font-weight:800;margin-bottom:16px;text-align:center;letter-spacing:2px;font-family:"Exo 2",sans-serif;text-shadow:0 0 20px rgba(255,196,14,0.5);';
            var progressWrap = document.createElement('div');
            progressWrap.style.cssText = 'width:400px;';
            var container = document.createElement('div');
            container.style.cssText = 'width:100%;height:8px;background:rgba(0,0,0,0.4);border:2px solid rgba(255,255,255,0.5);border-radius:10px;padding:2px;box-shadow:0 0 20px rgba(124,58,237,0.3);';
            this.bar = document.createElement('div');
            this.bar.style.cssText = 'height:100%;width:0%;background:linear-gradient(90deg,#f59e0b 0%,#FFC40E 30%,#fff5a8 50%,#FFC40E 70%,#f59e0b 100%);border-radius:6px;box-shadow:0 0 10px rgba(255,196,14,0.6);transition:width 0.2s ease-out;';
            container.appendChild(this.bar);
            progressWrap.appendChild(container);
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
            this.screen.appendChild(progressWrap);
            document.body.appendChild(this.screen);
            this.animate();
        },
        animate: function() {
            var self = this;
            function updateFrame() {
                var diff = self.targetProgress - self.progress;
                if (Math.abs(diff) > 0.1) { self.progress += diff * 0.1; }
                else { self.progress = self.targetProgress; }
                if (self.bar) self.bar.style.width = self.progress + '%';
                if (self.percent) self.percent.textContent = Math.floor(self.progress) + '%';
                if (self.status) {
                    if (self.progress < 30) self.status.textContent = 'Initializing...';
                    else if (self.progress < 60) self.status.textContent = 'Loading models...';
                    else if (self.progress < 85) self.status.textContent = 'Processing data...';
                    else if (self.progress < 100) self.status.textContent = 'Rendering charts...';
                    else self.status.textContent = 'Ready!';
                }
                if (self.isComplete && self.progress >= 99 && !self.isHiding) {
                    var elapsed = Date.now() - self.loadStartTime;
                    if (elapsed >= self.minDisplayTime) { self.hide(); return; }
                }
                if (!self.isHiding) requestAnimationFrame(updateFrame);
            }
            requestAnimationFrame(updateFrame);
        },
        setProgress: function(p) { this.targetProgress = Math.min(Math.max(p, 0), 100); },
        setModelLoading: function() { this.setProgress(30); },
        setProcessing: function() { this.setProgress(60); },
        setRendering: function() { this.setProgress(85); },
        complete: function() { this.isComplete = true; this.setProgress(100); },
        hide: function() {
            var self = this;
            if (this.isHiding || !this.screen) return;
            this.isHiding = true;
            this.screen.style.opacity = '0';
            setTimeout(function() {
                if (self.screen) { self.screen.remove(); self.screen = null; }
                window.dispatchEvent(new CustomEvent('appLoaderComplete'));
            }, 500);
        }
    };
    function observeGradioLoading() {
        var checkCount = 0;
        var maxChecks = 150;
        var lastPlotCount = 0;
        var stableCount = 0;
        var checkInterval = setInterval(function() {
            checkCount++;
            var plots = document.querySelectorAll('.js-plotly-plot');
            var hasPlots = plots.length >= 8;
            if (hasPlots) {
                if (plots.length === lastPlotCount && plots.length > 0) stableCount++;
                else stableCount = 0;
                lastPlotCount = plots.length;
                var allHaveData = true;
                for (var i = 0; i < plots.length; i++) {
                    var plot = plots[i];
                    if (!plot.data || plot.data.length === 0) {
                        if (!plot._fullLayout) { allHaveData = false; break; }
                    }
                }
                window.AppLoader.setProgress(90 + Math.min(stableCount, 8));
                if (stableCount >= 3 && allHaveData && plots.length >= 8) {
                    clearInterval(checkInterval);
                    window.AppLoader.complete();
                    return;
                }
            } else {
                window.AppLoader.setProgress(75 + Math.min(checkCount * 0.5, 10));
            }
            if (checkCount >= maxChecks) { clearInterval(checkInterval); window.AppLoader.complete(); }
        }, 100);
    }
    function init() {
        window.AppLoader.create();
        window.AppLoader.setProgress(10);
        setTimeout(function() {
            window.AppLoader.setProgress(25);
            setTimeout(observeGradioLoading, 500);
        }, 500);
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else { init(); }
})();
</script>
"""

CLASS_FILTER_JS = """
<script>
(function() {
    'use strict';
    window.selectedClassIndex = -1;
    const classColors = {
        0: '#440154', 1: '#FDE725', 2: '#B71C1C', 3: '#E53935',
        4: '#EF9A9A', 5: '#2563EB', 6: '#7DD3FC'
    };
    function initLegendButtonColors() {
        var colorMap = [
            { match: 'Target', color: '#FDE725' },
            { match: 'DFTJ', color: '#B71C1C' },
            { match: 'ISRJ', color: '#E53935' },
            { match: 'SIJ', color: '#EF9A9A' },
            { match: 'WJ', color: '#2563EB' },
            { match: 'BJ', color: '#7DD3FC' },
            { match: 'BACKGROUND', color: '#440154' }
        ];
        var buttons = document.querySelectorAll('.legend-class-btn');
        if (buttons.length === 0) {
            buttons = document.querySelectorAll('[class*="legend-c"]');
        }
        buttons.forEach(function(btn) {
            var text = btn.textContent.trim();
            if (btn.dataset.colorInit === 'true') return;
            for (var i = 0; i < colorMap.length; i++) {
                var item = colorMap[i];
                if (text === item.match || text.includes(item.match)) {
                    var colorBox = document.createElement('span');
                    colorBox.className = 'legend-color-box';
                    colorBox.style.cssText = 'display:inline-block;width:10px;height:10px;background:' + item.color + ';border-radius:2px;margin-right:5px;vertical-align:middle;border:1px solid rgba(255,255,255,0.3);flex-shrink:0;';
                    var innerDiv = btn.querySelector('div');
                    if (innerDiv && innerDiv.firstChild) {
                        innerDiv.insertBefore(colorBox, innerDiv.firstChild);
                    } else {
                        btn.insertBefore(colorBox, btn.firstChild);
                    }
                    btn.dataset.colorInit = 'true';
                    break;
                }
            }
        });
    }
    function initColorWithDelay() {
        [100, 300, 500, 1000, 2000, 3000].forEach(function(delay) {
            setTimeout(initLegendButtonColors, delay);
        });
    }
    function setupTabObserver() {
        var tabBtns = document.querySelectorAll('[role="tab"]');
        tabBtns.forEach(function(tab) {
            tab.addEventListener('click', function() {
                setTimeout(initLegendButtonColors, 300);
                setTimeout(initLegendButtonColors, 600);
            });
        });
    }
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() {
            initColorWithDelay();
            setTimeout(setupTabObserver, 1000);
        });
    } else {
        initColorWithDelay();
        setTimeout(setupTabObserver, 1000);
    }
    var colorObserver = new MutationObserver(function() {
        initLegendButtonColors();
    });
    setTimeout(function() {
        if (document.body) {
            colorObserver.observe(document.body, { childList: true, subtree: true });
        }
    }, 500);
    function updateButtonStates(activeIndex) {
        var buttons = document.querySelectorAll('.legend-btn');
        var resetBtn = document.getElementById('legend-reset');
        buttons.forEach(function(btn) {
            var idx = parseInt(btn.dataset.class);
            if (idx === activeIndex) { btn.classList.add('active'); }
            else { btn.classList.remove('active'); }
        });
        if (resetBtn) {
            if (activeIndex >= 0) { resetBtn.classList.add('has-active'); }
            else { resetBtn.classList.remove('has-active'); }
        }
    }
    function triggerGradioFilter(classIdx) {
        var filterInput = document.getElementById('class-filter-input');
        if (filterInput) {
            filterInput.value = classIdx.toString();
            filterInput.dispatchEvent(new Event('input', { bubbles: true }));
            var triggerBtn = document.getElementById('class-filter-trigger');
            if (triggerBtn) { triggerBtn.click(); }
        }
    }
    window.filterByClass = function(classIdx, btnElement) {
        if (classIdx === 0) return;
        if (window.selectedClassIndex === classIdx) {
            window.resetClassFilter();
            return;
        }
        window.selectedClassIndex = classIdx;
        updateButtonStates(classIdx);
        triggerGradioFilter(classIdx);
    };
    window.resetClassFilter = function() {
        window.selectedClassIndex = -1;
        updateButtonStates(-1);
        triggerGradioFilter(-1);
    };
})();
</script>
"""


# =============================================================================
# Section 13: Main Entry
# =============================================================================

if __name__ == "__main__":
    import sys

    port = int(os.environ.get("PORT", 8080))
    server_name = os.environ.get("SERVER_NAME", "0.0.0.0")

    print("=" * 60, file=sys.stderr)
    print("  End-to-End Radar Echo Segmentation", file=sys.stderr)
    print("=" * 60, file=sys.stderr)
    print(f"  Server: http://{server_name}:{port}", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    demo = create_interface()
    demo.launch(
        server_name=server_name,
        server_port=port,
        share=False,
        show_error=True,
        theme=gr.themes.Base(),
        css=CUSTOM_CSS,
        head=LOADING_SCREEN_JS + VIEW_TOGGLE_JS,
        max_file_size="50mb"
    )
