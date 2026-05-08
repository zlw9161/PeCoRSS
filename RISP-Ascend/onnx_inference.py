#ACL inference packages
import torch
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import imageio

import onnxruntime as ort

"""Main script to inference using a pretrained model"""
import argparse
import json

import numpy as np
from pathlib import Path
import os
from PIL import Image
from scipy.stats import hmean

import time 

 
from models import deeplabv3plus_resnet101, CWRNetPKC, CWRNet

# PKC part
class Evaluator:
    """Class to evaluate a model with quantitative metrics
    using a ground truth mask and a predicted mask.

    PARAMETERS
    ----------
    num_class: int
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_pixel_prec_class(self, harmonic_mean=False):
        """Pixel Precision"""
        prec_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=0)
        prec_by_class = np.nan_to_num(prec_by_class)
        if harmonic_mean:
            prec = hmean(prec_by_class)
        else:
            prec = np.mean(prec_by_class)
        return prec, prec_by_class

    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall"""
        recall_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=1)
        recall_by_class = np.nan_to_num(recall_by_class)
        if harmonic_mean:
            recall = hmean(recall_by_class)
        else:
            recall = np.mean(recall_by_class)
        return recall, recall_by_class

    def get_pixel_acc_class(self, harmonic_mean=False):
        """Pixel Accuracy"""
        acc_by_class = np.diag(self.confusion_matrix).sum() / (np.nansum(self.confusion_matrix, axis=1)
                                                               + np.nansum(self.confusion_matrix, axis=0)
                                                               + np.diag(self.confusion_matrix).sum()
                                                               - 2 * np.diag(self.confusion_matrix))
        acc_by_class = np.nan_to_num(acc_by_class)
        if harmonic_mean:
            acc = hmean(acc_by_class)
        else:
            acc = np.mean(acc_by_class)
        return acc, acc_by_class

    def get_miou_class(self, harmonic_mean=False):
        """Mean Intersection over Union"""
        miou_by_class = np.diag(self.confusion_matrix) / (np.nansum(self.confusion_matrix, axis=1)
                                                          + np.nansum(self.confusion_matrix, axis=0)
                                                          - np.diag(self.confusion_matrix))
        miou_by_class = np.nan_to_num(miou_by_class)
        if harmonic_mean:
            miou = hmean(miou_by_class)
        else:
            miou = np.mean(miou_by_class)
        return miou, miou_by_class

    def get_dice_class(self, harmonic_mean=False):
        """Dice"""
        _, prec_by_class = self.get_pixel_prec_class()
        _, recall_by_class = self.get_pixel_recall_class()
        # Add epsilon term to avoid /0
        dice_by_class = 2 * prec_by_class * recall_by_class / (prec_by_class + recall_by_class + 1e-8)
        if harmonic_mean:
            dice = hmean(dice_by_class)
        else:
            dice = np.mean(dice_by_class)
        return dice, dice_by_class

    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  labels=list(range(self.num_class)))
        return matrix

    def add_batch(self, labels, predictions):
        """Method to add ground truth and predicted masks by batch
        and update the global confusion matrix (entire dataset)

        PARAMETERS
        ----------
        labels: torch tensor or numpy array
            Ground truth masks
        predictions: torch tensor or numpy array
            Predicted masks
        """
        assert labels.shape == predictions.shape
        self.confusion_matrix += self._generate_matrix(labels, predictions)

    def reset(self):
        """Method to reset the confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def update(self, confusion_matrix):
        self.confusion_matrix = confusion_matrix.numpy()

    def get(self):
        return torch.from_numpy(self.confusion_matrix)


def normalize(data, file_path='kurad/config_files/cwr_rd_stats_all.json'):
    with open(file_path, 'r') as fp:
        rd_stats = json.load(fp)
    min_value = torch.tensor(rd_stats['min_val'])
    max_value = torch.tensor(rd_stats['max_val'])

    norm_data = torch.div(torch.sub(data, min_value),
                          torch.sub(max_value, min_value))
    return norm_data

def normalize_np(data, file_path='kurad/config_files/cwr_rd_stats_all.json'):
    with open(file_path, 'r') as fp:
        rd_stats = json.load(fp)
    min_value = np.array(rd_stats['min_val'])
    max_value = np.array(rd_stats['max_val'])
    norm_data = (data - min_value) / (max_value - min_value)
    return norm_data


def mask_to_img(mask, num_class):
    """Generate colors per class, only <=7 classes are supported"""
    # red, green, blue, yellow, cyan, white
    colors = [[255, 0, 0],
              [0, 255, 0],
              [0, 0, 255],
              [255, 255, 0],
              [0, 255, 255],
              [255, 255, 255]
              ]
    mask_img = np.zeros((mask.shape[0],
                         mask.shape[1], 3), dtype=np.uint8)
    for i in range(1, num_class):
        mask_img[mask == i] = colors[i - 1]
    mask_img = Image.fromarray(mask_img)
    return mask_img


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='cwr', help='Name of pretrained model, from [cwr, deeplabv3plus, cwrpkc].')
    parser.add_argument('--stats-path', default='./config/rd_stats_all.json', dest="stats_path",
                        help='Path to *stats_all.json.')
    parser.add_argument('--input-path', default='./input', dest="input_path", help='Input path.')
    parser.add_argument('--output-path', default='./output', dest="output_path", help='Output path.')
    parser.add_argument('--device', default='cuda', help='Use gpu or cpu.')
    parser.add_argument("--no-quali", dest="no_quali", action='store_true', help="Not output predicted images")
    args = parser.parse_args()
    print(args)
    return args

# ONNX inference 



def main():
    # t1 = time.time()
    args = parse_args()

    cfg_path = './config/{}.json'.format(args.model)

    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(args.device)

    output_list = []

    ## Model
    # if args.model == 'deeplabv3plus':
    #     model = deeplabv3plus_resnet101(n_classes=cfg['nb_classes'],
    #                                     n_frames=cfg['nb_input_channels'])
    # elif args.model == 'cwrpkc':
    #     model = CWRNetPKC(n_classes=cfg['nb_classes'],
    #                       n_frames=cfg['nb_input_channels'])
    # elif args.model == 'cwr':
    #     model = CWRNet(n_classes=cfg['nb_classes'],
    #                    n_frames=cfg['nb_input_channels'])
    # else:
    #     raise ValueError('model {} is not supported in test.py yet.'.format(args.model))
    #
    # weight_path = './weight/{}.pt'.format(args.model)
    #
    if cfg['model'] in ['deeplabv3plus']:
        add_temp = False
    else:
        add_temp = True
    #
    # ## Prepare model
    # saved_model = torch.load(weight_path, map_location=torch.device('cpu'))
    # model.load_state_dict(saved_model, strict=True)
    # model.to(device)

    #Dataset params
    if cfg['dataset'] == 'CWRdata':
        num_class = 4
    elif cfg['dataset'] == 'PWRdata':
        num_class = 5
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))

    #ACL init
    device = 0
    model_path = './deeplab_model.onnx'
    print('Now inference ', model_path)

    if not os.path.exists(model_path):
        raise Exception("the model is not exist")
    options = ort.SessionOptions()

    providers = [
        (
            "CANNExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "npu_mem_limit": 2 * 1024 * 1024 * 1024,
                "op_select_impl_mode": "high_performance",
                "optypelist_for_implmode": "Gelu",
                "enable_cann_graph": True
            },
        )
        ,"CPUExecutionProvider",
    ]
    session = ort.InferenceSession(model_path, sess_options=options, providers=providers)

    #PKC inference

    dataset_fpath = Path(args.input_path)
    seq_names = os.listdir(dataset_fpath)

    rd_metrics = Evaluator(num_class=num_class)
    folder_path = Path(args.output_path)

    seq_names = sorted(seq_names)
    time_count = 0
    time_total = 0

    for seq_name in seq_names:
        # if seq_name == '2020年11月29日11时29分55秒_mtd_机场人车_true':
        #     continue
        # if seq_name == '2021年01月25日13时22分47秒_mtd_S弯':
        #     continue

        inp_frame_fpath = dataset_fpath / seq_name / 'range_doppler_numpy'
        inp_mask_fpath = dataset_fpath / seq_name / 'annotations' / 'dense'

        mask_fpath = folder_path / seq_name / 'mask'
        output_fpath = folder_path / seq_name / 'output'
        mask_fpath.mkdir(parents=True, exist_ok=True)
        output_fpath.mkdir(parents=True, exist_ok=True)

        frame_names = os.listdir(inp_frame_fpath)
        frame_names = [frame_name.split('.')[0] for frame_name in frame_names]
        # frame_names = sorted(frame_names, key=lambda x: int(x.split('_')[2]))
        frame_names = sorted(frame_names)
        # print(frame_names)

        # print('len: ', len(frame_names[cfg['nb_input_channels'] - 1:]))

        # 遍历每个待测帧
        for idx_frame in range(len(frame_names[cfg['nb_input_channels'] - 1:])):
            init_frame_name = frame_names[idx_frame + cfg['nb_input_channels'] - 1]
            rd_mask = np.load(inp_mask_fpath / init_frame_name / 'range_doppler.npy')

            rd_matrices = []
            for frame_name in frame_names[idx_frame:idx_frame + cfg['nb_input_channels']]:
                rd_matrix = np.load(os.path.join(inp_frame_fpath, frame_name + '.npy'))
                rd_matrices.append(rd_matrix)

            rd_matrix = np.dstack(rd_matrices)
            rd_matrix = np.rollaxis(rd_matrix, axis=-1)
            if add_temp:
                rd_matrix = np.expand_dims(rd_matrix, axis=0)

            # 增加batch维度
            rd_matrix = np.expand_dims(rd_matrix, axis=0)
            rd_mask = np.expand_dims(rd_mask, axis=0)

            # 先numpy torch上互转为了debug方便，后续提速全在numpy上操作
            # rd_matrix = torch.from_numpy(rd_matrix)
            # rd_matrix = rd_matrix.float()
            # rd_matrix = normalize(rd_matrix, file_path=args.stats_path)
            # rd_matrix = rd_matrix.numpy()

            rd_matrix = normalize_np(rd_matrix, file_path=args.stats_path)

            # ACL procedure
            inputs = {session.get_inputs()[0].name: rd_matrix.astype(np.float32)}
            # inference
            t1 = time.time()
            outputs = session.run(None, inputs)[0]
            t2 = time.time()
            time_total = time_total + t2 - t1
            time_count = time_count + 1

            #debug
            output_list.append(outputs)

            # 先numpy， torch上互转为了debug方便，后续提速全在numpy上操作
            
            # outputs = torch.from_numpy(outputs).float()
            # rd_metrics.add_batch(np.argmax(rd_mask, axis=1), torch.argmax(outputs, axis=1).cpu())

            rd_metrics.add_batch(np.argmax(rd_mask, axis=1), np.argmax(outputs, axis=1))
            # debug
            print(init_frame_name)
            prec, prec_by_class = rd_metrics.get_pixel_prec_class()
            print('precision: ', prec)

            if not args.no_quali:
                # outputs = torch.argmax(outputs, axis=1).cpu().numpy()
                outputs = np.argmax(outputs, axis= 1)
                masks = np.argmax(rd_mask, axis=1)

                for i in range(outputs.shape[0]):
                    mask_img = mask_to_img(masks[i], num_class)
                    mask_path = mask_fpath / 'mask_{}.png'.format(init_frame_name)
                    mask_img.save(mask_path)
                    output_img = mask_to_img(outputs[i], num_class)
                    output_path = output_fpath / 'output_{}.png'.format(init_frame_name)
                    output_img.save(output_path)
        # print('This is Test mode, remove config_path in acl.init() and below break in official release!')
        # if idx_frame >= 3:
        #     break

    acc, acc_by_class = rd_metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = rd_metrics.get_pixel_prec_class()
    recall, recall_by_class = rd_metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = rd_metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = rd_metrics.get_dice_class()
    confusion_matrix = rd_metrics.get().tolist()

    # output_list = np.array(output_list)
    # np.save('outputs_debug.npy', output_list)
    # t2 = time.time()



    print(f'Acc      : {acc}')
    print(f'Precision: {prec}')
    print(f'Recall   : {recall}')
    print(f'mIoU     : {miou}')
    print(f'Dice     : {dice}')
    print(f'confusion_matrix     : {confusion_matrix}')

    print('Avg running time:', time_total/time_count)

    print("*****run finish******")

if __name__ == '__main__':
    main()
