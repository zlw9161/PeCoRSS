"""Class to load datasets"""
import json
from svrss.utils.paths import Paths
import numpy as np

class LSFJamming:
    """Class to load LSFJamming dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.lsfdata = self.paths['LSFJamming']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.7, 0.2, 0.1]
        self._split()

    def _load_data_seq_ref(self):
        path = self.lsfdata / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.lsfdata / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.data_seq_ref.keys():
            num_frames = len(self.annotations[sequence])
            # self.annotations[sequence] = np.random.permutation(num_frames)
            self.train[sequence] = self.annotations[sequence][0:int(num_frames * cum_ratio[0])]
            self.validation[sequence] = self.annotations[sequence] \
                [int(num_frames * cum_ratio[0]):int(num_frames * cum_ratio[1])]
            self.test[sequence] = self.annotations[sequence][int(num_frames * cum_ratio[1]):]
            # print('Test', sequence, self.test[sequence][:-1])

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))

class CWRdata:
    """Class to load CWRdata dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.cwrdata = self.paths['CWRdata']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.split_type = 'seq'
        self.ratio_spilt = [0.7, 0.1, 0.2]
        self._split()

    def _load_data_seq_ref(self):
        path = self.cwrdata / 'data_seq_ref_-10dB.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.cwrdata / 'light_dataset_frame_oriented_-10dB.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.annotations.keys():
            if self.split_type == 'seq':
                split = self.data_seq_ref[sequence]['split']
                num_frames = len(self.annotations[sequence])
                if split == 'Train':
                    self.train[sequence] = self.annotations[sequence]
                if 'Validation' in split:
                    self.validation[sequence] = self.annotations[sequence]
                if 'Test' in split:
                    self.test[sequence] = self.annotations[sequence]
            elif self.split_type == 'ratio':
                if '机场人车' not in sequence:
                    num_frames = len(self.annotations[sequence])
                    # self.annotations[sequence] = np.random.permutation(num_frames)
                    self.train[sequence] = self.annotations[sequence][0:int(num_frames * cum_ratio[0])]
                    self.validation[sequence] = self.annotations[sequence] \
                        [int(num_frames * cum_ratio[0]):int(num_frames * cum_ratio[1])]
                    self.test[sequence] = self.annotations[sequence][int(num_frames * cum_ratio[1]):]
                else:
                    num_frames = len(self.annotations[sequence]) - 200
                    self.train[sequence] = self.annotations[sequence][0:int(200 * cum_ratio[0])] + \
                                        self.annotations[sequence][200:200 + int(num_frames * cum_ratio[0])]

                    self.validation[sequence] = self.annotations[sequence][
                                                int(200 * cum_ratio[0]):int(200 * cum_ratio[1])] + \
                                                self.annotations[sequence][
                                                200 + int(num_frames * cum_ratio[0]):200 + int(num_frames * cum_ratio[1])]

                    self.test[sequence] = self.annotations[sequence][int(200 * cum_ratio[1]):200] + \
                                        self.annotations[sequence][200 + int(num_frames * cum_ratio[1]):]

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


class NuaaSim:
    """Class to load PDRdata dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.pdrdata = self.paths['NuaaSim']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.7, 0.2, 0.1]
        self._split()

    def _load_data_seq_ref(self):
        path = self.pdrdata / 'data_seq_ref_SLO.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.pdrdata / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.data_seq_ref.keys():
            num_frames = len(self.annotations[sequence])
            # self.annotations[sequence] = np.random.permutation(num_frames)
            self.train[sequence] = self.annotations[sequence][0:int(num_frames * cum_ratio[0])]
            self.validation[sequence] = self.annotations[sequence] \
                [int(num_frames * cum_ratio[0]):int(num_frames * cum_ratio[1])]
            self.test[sequence] = self.annotations[sequence][int(num_frames * cum_ratio[1]):]
            # print('Test', sequence, self.test[sequence][:-1])

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


class PDRdata:
    """Class to load PDRdata dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.pdrdata = self.paths['PDRdata']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.8, 0.1, 0.1]
        self._split()

    def _load_data_seq_ref(self):
        path = self.pdrdata / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.pdrdata / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):

        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.annotations.keys():
            num_frames = len(self.annotations[sequence])
            # self.annotations[sequence] = np.random.permutation(num_frames)
            self.train[sequence] = self.annotations[sequence][0:int(num_frames * cum_ratio[0])]
            self.validation[sequence] = self.annotations[sequence] \
                [int(num_frames * cum_ratio[0]):int(num_frames * cum_ratio[1])]
            self.test[sequence] = self.annotations[sequence][int(num_frames * cum_ratio[1]):]

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


class RDSyn:
    """Class to load Synthesis or Custom RD dataset"""

    def __init__(self):
        self.paths = Paths().get()
        self.rdsyn = self.paths['RDSyn']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.split_type = 'ratio'
        self.ratio_spilt = [0.7, 0.1, 0.2]
        self._split()

    def _load_data_seq_ref(self):
        path = self.rdsyn / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.rdsyn / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='GBK') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.annotations.keys():
            if self.split_type == 'seq':
                split = self.data_seq_ref[sequence]['split']
                num_frames = len(self.annotations[sequence])
                if split == 'Train':
                    self.train[sequence] = self.annotations[sequence]
                if 'Validation' in split:
                    self.validation[sequence] = self.annotations[sequence]
                if 'Test' in split:
                    self.test[sequence] = self.annotations[sequence]
            elif self.split_type == 'ratio':
                num_frames = len(self.annotations[sequence])
                self.train[sequence] = self.annotations[sequence][0:int(num_frames * cum_ratio[0])]
                self.validation[sequence] = self.annotations[sequence] \
                    [int(num_frames * cum_ratio[0]):int(num_frames * cum_ratio[1])]
                self.test[sequence] = self.annotations[sequence][int(num_frames * cum_ratio[1]):]

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


class XPDdata:
    """Class to load XPDdata dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.xpddata = self.paths['XPDdata']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self.ratio_spilt = [0.7, 0.1, 0.2]
        self._split()

    def _load_data_seq_ref(self):
        path = self.xpddata / 'data_seq_ref.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.xpddata / 'light_dataset_frame_oriented.json'
        with open(path, 'r', encoding='UTF-8') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        cum_ratio = np.cumsum(self.ratio_spilt)
        for sequence in self.data_seq_ref.keys():
            num_frames = len(self.annotations[sequence])
            self.train[sequence] = self.annotations[sequence][0:int(num_frames * cum_ratio[0])]
            self.validation[sequence] = self.annotations[sequence] \
                [int(num_frames * cum_ratio[0]):int(num_frames * cum_ratio[1])]
            self.test[sequence] = self.annotations[sequence][int(num_frames * cum_ratio[1]):]

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


class Carrada:
    """Class to load CARRADA dataset"""

    def __init__(self):
        self.paths = Paths().get()
        # self.warehouse = self.paths['warehouse']
        self.carrada = self.paths['Carrada']
        self.data_seq_ref = self._load_data_seq_ref()
        self.annotations = self._load_dataset_ids()
        self.train = dict()
        self.validation = dict()
        self.test = dict()
        self._split()

    def _load_data_seq_ref(self):
        path = self.carrada / 'data_seq_ref.json'
        with open(path, 'r') as fp:
            data_seq_ref = json.load(fp)
        return data_seq_ref

    def _load_dataset_ids(self):
        path = self.carrada / 'light_dataset_frame_oriented.json'
        with open(path, 'r') as fp:
            annotations = json.load(fp)
        return annotations

    def _split(self):
        for sequence in self.annotations.keys():
            split = self.data_seq_ref[sequence]['split']
            if split == 'Train':
                self.train[sequence] = self.annotations[sequence]
            elif split == 'Validation':
                self.validation[sequence] = self.annotations[sequence]
            elif split == 'Test':
                self.test[sequence] = self.annotations[sequence]
            else:
                raise TypeError('Type {} is not supported for splits.'.format(split))

    def get(self, split):
        """Method to get the corresponding split of the dataset"""
        if split == 'Train':
            return self.train
        if split == 'Validation':
            return self.validation
        if split == 'Test':
            return self.test
        raise TypeError('Type {} is not supported for splits.'.format(split))


def test_cwr():
    """Method to test the cwrdata dataset"""
    dataset = CWRdata().get('Test')
    print(dataset['2020年11月11日15时50分44秒_mtd_mid'])


def test_pdr():
    """Method to test the pdrdata dataset"""
    dataset = PDRdata().get('Train')
    print(dataset['20191210_145742_Ku5km_SeaBoat'])


def test_xpd():
    """Method to test the pdrdata dataset"""
    dataset = XPDdata().get('Train')
    print(dataset['seq0'])


def test_nuaa():
    """Method to test the rdsyn dataset"""
    dataset_trn = NuaaSim().get('Train')
    dataset_val = NuaaSim().get('Validation')
    dataset_tst = NuaaSim().get('Test')
    # assert '0410_SNR10' in dataset.keys()
    # assert '0410_SNR15' in dataset.keys()
    # assert '0410_SNR20' in dataset.keys()
    # print("train:", dataset_trn['0410_SNR20'])
    # print("valid:", dataset_val['0410_SNR15'])
    # print("test:", dataset_trn['0410_SNR10'])
    # print("train:", dataset_trn['data_1000m_Hover_10dB'])
    # print("valid:", dataset_val['data_3000m_Hover_10dB'])
    print("test:", dataset_tst['data_5000m_Hover_10dB'])
    print(NuaaSim().data_seq_ref)


if __name__ == '__main__':
    test_nuaa()
