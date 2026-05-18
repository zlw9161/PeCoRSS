"""Classes to load datasets"""
import os
import warnings
import numpy as np
import torch
from skimage import transform
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from svrss.loaders.dataset import CWRdata, PDRdata
from svrss.utils.paths import Paths


class SequenceDataset(Dataset):
    """DataLoader class for dataset sequences"""

    def __init__(self, dataset):
        self.dataset = dataset
        self.seq_names = list(self.dataset.keys())

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        return seq_name, self.dataset[seq_name]


def _frame_id_from_entry(frame_entry):
    """Resolve frame id whether JSON stores plain str or legacy [frame_id, ...] rows."""
    if isinstance(frame_entry, (list, tuple)):
        return frame_entry[0]
    return frame_entry


class PeCoRSVDataset(Dataset):
    """DataLoader class for Radar sequences from different sources
    Load frames, only for semantic segmentation
    Specific to load several frames at the same time (sub sequences)
    Aggregated Tensor Input + Multiple Output

    PARAMETERS
    ----------
    dataset: SequenceCarradaDataset object
    annotation_type: str
        Supported annotations are 'sparse', 'dense'
    path_to_frames: str
        Path to the frames of a given sequence (folder of the sequence)
    process_signal: boolean
        Load signal w/ or w/o processing (power, log transform)
    n_frame: int
        Number of frames used for each sample
    transformations: list of functions
        Preprocessing or data augmentation functions
        Default: None
    add_temp: boolean
        Formating the input tensors as sequences
        Default: False
    """

    def __init__(self, dataset, dataset_type, annotation_type, signal_type, path_to_frames, process_signal,
                 n_frames, cmplx_flag=False, transformations=None, add_temp=False, mulmask=False):
        self.dataset = dataset
        self.dataset_type = dataset_type
        self.annotation_type = annotation_type
        self.signal_type = signal_type
        self.path_to_frames = Path(path_to_frames)
        self.process_signal = process_signal
        self.n_frames = n_frames
        self.cmplx_flag = cmplx_flag
        self.transformations = transformations
        self.add_temp = add_temp
        self.mulmask = bool(mulmask)
        self._warned_bad_samples = set()
        # self.dataset = self.dataset[self.n_frames-1:]  # remove n first frames
        if self.dataset_type in ['CarradaRAC_RD', 'CarradaRAC_RA']:
            self.path_to_annots = self.path_to_frames / 'new_annotations' / self.annotation_type
        else:
            self.path_to_annots = self.path_to_frames / 'annotations' / self.annotation_type

    def transform(self, frame, is_vflip=False, is_hflip=False):
        """
        Method to apply preprocessing / data augmentation functions

        PARAMETERS
        ----------
        frame: dict
            Contains the matrices and the masks on which we want to apply the transformations
        is_vfilp: boolean
            If you want to apply a vertical flip
            Default: False
        is_hfilp: boolean
            If you want to apply a horizontal flip
            Default: False

        RETURNS
        -------
        frame: dict
        """
        if self.transformations is not None:
            for function in self.transformations:
                if isinstance(function, VFlip):
                    if is_vflip:
                        frame = function(frame)
                    else:
                        continue
                if isinstance(function, HFlip):
                    if is_hflip:
                        frame = function(frame)
                    else:
                        continue
                if not isinstance(function, VFlip) and not isinstance(function, HFlip):
                    frame = function(frame)
        return frame

    def __len__(self):
        """Number of frames per sequence"""
        return len(self.dataset[self.n_frames-1:])

    def __getitem__(self, idx):
        return self._getitem_with_retry(idx, max_retries=5)

    def _getitem_with_retry(self, idx, max_retries=5):
        last_exc = None
        for retry in range(max_retries + 1):
            cur_idx = (idx + retry) % len(self)
            try:
                return self._load_item(cur_idx)
            except Exception as exc:
                last_exc = exc
                if self.dataset_type != 'LSFJamming':
                    raise
                bad_key = (str(self.path_to_frames), cur_idx, str(exc))
                if bad_key not in self._warned_bad_samples:
                    warnings.warn(
                        f"[LSFJamming] skip corrupted sample at idx={cur_idx} in {self.path_to_frames}: {exc}",
                        RuntimeWarning,
                    )
                    self._warned_bad_samples.add(bad_key)
                continue
        raise RuntimeError(
            f"Failed to fetch a valid sample after {max_retries + 1} retries for {self.path_to_frames}."
        ) from last_exc

    def _load_item(self, idx):
        init_frame_entry = self.dataset[idx + self.n_frames - 1]
        init_frame_name = _frame_id_from_entry(init_frame_entry)
        if self.dataset_type in ['Carrada_RD', 'Carrada_RA', 'CarradaRAC_RD', 'CarradaRAC_RA']:
            frame_id = int(str(init_frame_name))
            frame_names = [str(f_id).zfill(6) for f_id in range(frame_id-self.n_frames+1, frame_id+1)]
        else:
            frame_names = [
                str(_frame_id_from_entry(self.dataset[f_id]))
                for f_id in range(idx, idx + self.n_frames)
            ]
        matrices = []
        if self.mulmask:
            masks = []
        else:
            mask = np.load(
                os.path.join(self.path_to_annots, init_frame_name, self.signal_type + ".npy")
            )
            if self.dataset_type == 'LSFJamming':
                mask = torch.nn.functional.one_hot(torch.from_numpy(mask).long(), num_classes=7)
                mask = mask.permute(2, 0, 1).float().numpy()
        for frame_name in frame_names:
            if self.dataset_type in ['Carrada_RD', 'Carrada_RA', 'CarradaRAC_RD', 'CarradaRAC_RA']:
                if self.process_signal:
                    matrix = np.load(
                        os.path.join(
                            self.path_to_frames,
                            self.signal_type + "_processed",
                            frame_name + ".npy",
                        )
                    )
                else:
                    matrix = np.load(
                        os.path.join(
                            self.path_to_frames,
                            self.signal_type + "_raw",
                            frame_name + ".npy",
                        )
                    )
            elif self.dataset_type == 'XPDdata':
                matrix = np.load(
                    os.path.join(
                        self.path_to_frames,
                        self.signal_type + "_mtdData_amp",
                        frame_name + ".npy",
                    )
                )
            elif self.dataset_type == 'NuaaSim':
                if self.cmplx_flag:
                    matrix = np.load(
                        os.path.join(
                            self.path_to_frames,
                            self.signal_type + "_echo",
                            frame_name + ".npy",
                        )
                    )
                    # if len(rd_matrix.size) == 4:
                    #     rd_matrix = rd_matrix[0,...]
                else:
                    matrix = np.load(
                        os.path.join(
                            self.path_to_frames,
                            self.signal_type + "_mtdData_amp",
                            frame_name + ".npy",
                        )
                    )
            elif self.dataset_type == 'LSFJamming':
                if self.cmplx_flag:
                    rd_matrix_temp = np.load(os.path.join(self.path_to_frames,
                                                          self.signal_type + '_echo',
                                                          frame_name + '.npy'))
                    matrix = rd_matrix_temp[0, ...] + 1j * rd_matrix_temp[1, ...]
                else:
                    rd_matrix_temp = np.load(os.path.join(self.path_to_frames,
                                                          self.signal_type + '_MTD',
                                                          frame_name + '.npy'))
                    matrix = np.abs(rd_matrix_temp[0, ...] + 1j * rd_matrix_temp[1, ...])
            else:
                matrix = np.load(
                    os.path.join(
                        self.path_to_frames, self.signal_type + "_numpy", frame_name + ".npy"
                    )
                )

            matrices.append(matrix)
            if self.mulmask:
                mask_i = np.load(
                    os.path.join(self.path_to_annots, frame_name, self.signal_type + ".npy")
                )
                if self.dataset_type == 'LSFJamming':
                    mask_i = torch.nn.functional.one_hot(torch.from_numpy(mask_i).long(), num_classes=7)
                    mask_i = mask_i.permute(2, 0, 1).float().numpy()
                masks.append(mask_i)

        # Apply the same transfo to all representations
        if np.random.uniform(0, 1) > 0.5:
            is_vflip = True
        else:
            is_vflip = False
        if np.random.uniform(0, 1) > 0.5:
            is_hflip = True
        else:
            is_hflip = False

        matrix = np.array([np.array(sub_list) for sub_list in matrices])
        if self.mulmask:
            # Keep the same mask layout as mvrss: [C, T, H, W] before batching.
            mask = np.stack(masks).transpose(1, 0, 2, 3)
        frame0 = {"matrix": matrix, "mask": mask}
        frame0 = self.transform(frame0, is_vflip=is_vflip, is_hflip=is_hflip)
        if self.add_temp:
            if isinstance(self.add_temp, bool):
                frame0["matrix"] = np.expand_dims(frame0["matrix"], axis=0)
            else:
                assert isinstance(self.add_temp, int)
                frame0["matrix"] = np.expand_dims(frame0["matrix"], axis=self.add_temp)

        if self.signal_type == "range_doppler":
            return {"rd_matrix": frame0["matrix"], "rd_mask": frame0["mask"]}
        if self.signal_type == "range_angle":
            return {"ra_matrix": frame0["matrix"], "ra_mask": frame0["mask"]}
        if self.signal_type == "angle_doppler":
            return {"ad_matrix": frame0["matrix"], "ad_mask": frame0["mask"]}

        raise ValueError(f"Unsupported signal_type: {self.signal_type}")


def _peco_dataset_type_from_cfg(cfg: dict) -> str:
    """Mirror model_ddp_sv.ModelDDP._peco_dataset_type for dataloader construction."""
    d = (cfg.get("dataset") or "").strip()
    dl = d.lower()
    if dl in ("cwr", "cwrdata"):
        return "CWRdata"
    if dl in ("pdr", "pdrdata"):
        return "PDRdata"
    if dl == "carrada":
        st = cfg.get("signal_type")
        if st == "range_doppler":
            return "Carrada_RD"
        if st == "range_angle":
            return "Carrada_RA"
        return "Carrada"
    return d


def _sequence_frames_root_from_cfg(cfg: dict, seq_name: str) -> Path:
    """Mirror model_ddp_sv.ModelDDP._sequence_frames_root using string sequence name."""
    paths = Paths().get()
    dl = (cfg.get("dataset") or "").strip().lower()
    if dl in ("cwr", "cwrdata"):
        return paths["CWRdata"] / seq_name
    if dl in ("pdr", "pdrdata"):
        return paths["PDRdata"] / seq_name
    if dl == "carrada" or dl.startswith("carrada"):
        return paths["carrada"] / seq_name
    if dl in ("xpddata", "xpd"):
        return paths["XPDdata"] / seq_name
    if dl in ("rdsyn",):
        return paths["RDSyn"] / seq_name
    if dl in ("nuaasim",):
        return paths["NuaaSim"] / seq_name
    raise KeyError(
        f"_sequence_frames_root_from_cfg: unsupported cfg['dataset']={cfg.get('dataset')!r}"
    )


class SVGlobalWindowTrainDataset(Dataset):
    """All sliding-window samples over the train split, indexed as a flat list.

    Each item matches ``PeCoRSVDataset(seq, ...)[window_idx]`` for one sequence. Combined with
    ``DataLoader(..., shuffle=True)`` or ``DistributedSampler(..., shuffle=True)``, window order
    is randomized each epoch while still visiting every window exactly once per epoch (each index
    owned by one rank under DDP).

    Call :meth:`update_training_episode` at the start of each training epoch so augmentations
    and ``add_temp`` stay in sync with :class:`ModelDDP`.
    """

    def __init__(self, split_dict: dict, cfg: dict):
        self.cfg = cfg
        self.split_dict = split_dict
        self.n_frames = int(cfg["nb_input_channels"])
        self.annot_type = cfg["annot_type"]
        self.signal_type = cfg["signal_type"]
        self.process_signal = cfg["process_signal"]
        self.cmplx_flag = cfg.get("cmplx_flag", False)
        self.mulmask = bool(cfg.get("mulmask", False))
        self._transformations = None
        self._add_temp = False

        pe_type = _peco_dataset_type_from_cfg(cfg)
        self._peco_by_seq = {}
        self._pairs = []

        for seq_name in sorted(split_dict.keys()):
            seq = split_dict[seq_name]
            nwin = len(seq) - self.n_frames + 1
            if nwin <= 0:
                continue
            path_to_frames = _sequence_frames_root_from_cfg(cfg, seq_name)
            self._peco_by_seq[seq_name] = PeCoRSVDataset(
                seq,
                pe_type,
                self.annot_type,
                self.signal_type,
                path_to_frames,
                self.process_signal,
                self.n_frames,
                self.cmplx_flag,
                None,
                False,
                self.mulmask,
            )
            for widx in range(nwin):
                self._pairs.append((seq_name, widx))

    def __len__(self):
        return len(self._pairs)

    def update_training_episode(self, transformations, add_temp):
        """Refresh augmentation pipeline and temporal axis flag on cached per-sequence loaders."""
        self._transformations = transformations
        self._add_temp = add_temp
        for peco in self._peco_by_seq.values():
            peco.transformations = transformations
            peco.add_temp = add_temp

    def __getitem__(self, idx):
        seq_name, widx = self._pairs[idx]
        return self._peco_by_seq[seq_name][widx]


class Rescale:
    """Rescale the image in a sample to a given size.

    PARAMETERS
    ----------
    output_size: tuple or int
        Desired output size. If tuple, output is
        matched to output_size. If int, smaller of image edges is matched
        to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, frame):
        matrix, rd_mask, ra_mask = frame['matrix'], frame['rd_mask'], frame['ra_mask']
        h, w = matrix.shape[1:]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        # transform.resize induce a smoothing effect on the values
        # transform only the input data
        matrix = transform.resize(matrix, (matrix.shape[0], new_h, new_w))
        return {'matrix': matrix, 'rd_mask': rd_mask, 'ra_mask': ra_mask}


class Flip:
    """
    Randomly flip the matrix with a proba p
    """

    def __init__(self, proba):
        assert proba <= 1.
        self.proba = proba

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        h_flip_proba = np.random.uniform(0, 1)
        if h_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=-2).copy()
            mask = np.flip(mask, axis=-2).copy()
        v_flip_proba = np.random.uniform(0, 1)
        if v_flip_proba < self.proba:
            matrix = np.flip(matrix, axis=-1).copy()
            mask = np.flip(mask, axis=-1).copy()
        return {'matrix': matrix, 'mask': mask}


class HFlip:
    """
    Randomly horizontal flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=-2).copy()
        mask = np.flip(mask, axis=-2).copy()
        return {'matrix': matrix, 'mask': mask}


class VFlip:
    """
    Randomly vertical flip the matrix with a proba p
    """

    def __init__(self):
        pass

    def __call__(self, frame):
        matrix, mask = frame['matrix'], frame['mask']
        matrix = np.flip(matrix, axis=-1).copy()
        mask = np.flip(mask, axis=-1).copy()
        return {'matrix': matrix, 'mask': mask}


def test_sequence():
    dataset = CWRdata().get('Train')
    dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                            shuffle=False, num_workers=0)
    for i, data in enumerate(dataloader):
        seq_name, seq = data
        if i == 0:
            seq = [subseq[0] for subseq in seq]
            assert seq_name[0] == '2020年11月11日15时50分44秒_mtd_mid'
            assert 'MTD_Scan_1_27' in seq
        else:
            break


def test_XPDdata():
    import svrss.loaders.dataset as ds
    if not hasattr(ds, "XPDdata"):
        print("skip test_XPDdata: XPDdata not in dataset.py")
        return
    XPDdata = ds.XPDdata
    paths = Paths().get()
    n_frames = 3
    dataset = XPDdata().get('Train')
    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['XPDdata'] / seq_name[0]
        frame_dataloader = DataLoader(PeCoRSVDataset(seq,
                                                     'XPDdata',
                                                     'dense',
                                                     'range_doppler',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            assert list(frame['rd_matrix'].shape[2:]) == [64, 148]
            print(frame['rd_matrix'].shape)
            # assert list(frame['ra_matrix'].shape[2:]) == [256, 256]
            # assert list(frame['ad_matrix'].shape[2:]) == [256, 64]
            assert frame['rd_matrix'].shape[1] == n_frames
            assert list(frame['rd_mask'].shape[2:]) == [64, 148]
            print(frame['rd_mask'].shape)
            # assert list(frame['ra_mask'].shape[2:]) == [256, 256]
        break


def test_Nuaa():
    import svrss.loaders.dataset as ds
    if not hasattr(ds, "NuaaSim"):
        print("skip test_Nuaa: NuaaSim not in dataset.py")
        return
    NuaaSim = ds.NuaaSim
    paths = Paths().get()
    n_frames = 3
    dataset = NuaaSim().get('Train')
    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['NuaaSim'] / seq_name[0]
        frame_dataloader = DataLoader(PeCoRSVDataset(seq,
                                                     'NuaaSim',
                                                     'dense',
                                                     'range_doppler',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames,
                                                     cmplx_flag=False),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            assert list(frame['rd_matrix'].shape[2:]) == [128, 1024]
            print(frame['rd_matrix'].shape)
            # assert list(frame['ra_matrix'].shape[2:]) == [256, 256]
            # assert list(frame['ad_matrix'].shape[2:]) == [256, 64]
            assert frame['rd_matrix'].shape[1] == n_frames
            assert list(frame['rd_mask'].shape[2:]) == [128, 1024]
            print(frame['rd_mask'].shape)
            # assert list(frame['ra_mask'].shape[2:]) == [256, 256]
        break


def test_subflip():
    paths = Paths().get()
    n_frames = 3
    dataset = CWRdata().get('Train')
    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data
        path_to_frames = paths['CWRdata'] / seq_name[0]
        frame_dataloader = DataLoader(PeCoRSVDataset(seq,
                                                     'CWRdata',
                                                     'dense',
                                                     'range_doppler',
                                                     path_to_frames,
                                                     True,
                                                     n_frames),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            rd_matrix = frame['rd_matrix'][0].cpu().detach().numpy()
            rd_mask = frame['rd_mask'][0].cpu().detach().numpy()
            rd_frame_test = {'matrix': rd_matrix,
                             'mask': rd_mask}
            rd_frame_vflip = VFlip()(rd_frame_test)
            rd_matrix_vflip = rd_frame_vflip['matrix']
            rd_frame_hflip = HFlip()(rd_frame_test)
            rd_matrix_hflip = rd_frame_hflip['matrix']
            assert rd_matrix[0][0][0] == rd_matrix_vflip[0][0][-1]
            assert rd_matrix[0][0][-1] == rd_matrix_vflip[0][0][0]
            assert rd_matrix[0][0][0] == rd_matrix_hflip[0][-1][0]
            assert rd_matrix[0][-1][0] == rd_matrix_hflip[0][0][0]
        break


def test_snr_eval():
    import svrss.loaders.dataset as ds
    if not hasattr(ds, "NuaaSim"):
        print("skip test_snr_eval: NuaaSim not in dataset.py")
        return
    NuaaSim = ds.NuaaSim
    paths = Paths().get()
    n_frames = 1
    dataset = NuaaSim().get('Train')
    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=True, num_workers=0)
    for _, data in enumerate(seq_dataloader):
        seq_name, seq = data

        path_to_frames = paths['NuaaSim'] / seq_name[0]
        frame_dataloader = DataLoader(PeCoRSVDataset(seq,
                                                     'NuaaSim',
                                                     'dense',
                                                     'range_doppler',
                                                     path_to_frames,
                                                     process_signal=True,
                                                     n_frames=n_frames,
                                                     cmplx_flag=False),
                                      shuffle=False,
                                      batch_size=1,
                                      num_workers=0)
        for _, frame in enumerate(frame_dataloader):
            data = frame['rd_matrix']
            mask = torch.argmax(frame['rd_mask'],dim=1)
            signal_power = ((data*mask)**2).sum()/mask.sum()
            noise_power = ((data*(1-mask))**2).sum()/(1-mask).sum()
            SNR = 10*np.log10(signal_power/noise_power)
            print(SNR)




    
if __name__ == '__main__':
    test_sequence()