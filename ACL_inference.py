#ACL inference packages
import torch
from sklearn.metrics import confusion_matrix
import os
import numpy as np
import acl
import imageio

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

# ACL inference part

# ACL hyperparams
NPY_FLOAT32 = 11
ACL_MEMCPY_HOST_TO_HOST = 0
ACL_MEMCPY_HOST_TO_DEVICE = 1
ACL_MEMCPY_DEVICE_TO_HOST = 2
ACL_MEMCPY_DEVICE_TO_DEVICE = 3
ACL_MEM_MALLOC_HUGE_FIRST = 0
ACL_DEVICE, ACL_HOST = 0, 1
ACL_SUCCESS = 0

# ACL class
class Model_Inference(object):
    def __init__(self, device_id,
                 model_path,
                 cropping_size=(64, 64),
                 overlap_size=(16, 0), img_size=(1024, 64)):
        self.device_id = device_id  # int
        self.context = None  # pointer
        self.stream = None

        self.model_id = None  # pointer
        self.model_path = model_path  # string
        self.model_desc = None  # pointer when using
        self.input_dataset = None
        self.output_dataset = None
        self.input_buffer = None
        self.output_buffer = None
        self.input_buffer_size = None
        self.image_bytes = None
        self.image_name = None
        self.dir = None
        self.image = None
        self.runMode_ = acl.rt.get_run_mode()
        self.bytes_input = None
        self.cropping_size = cropping_size
        self.overlap_size = overlap_size
        self.process_idx_debug = 0
        self.output_patch_list = []
        self.img_size = img_size
        self.config_path = './acl.json'

    def init_resource(self):
        # init acl resource
        ret = acl.init()
        if ret != ACL_SUCCESS:
            print('acl init failed, errorCode is', ret)

        ret = acl.rt.set_device(self.device_id)
        if ret != ACL_SUCCESS:
            print('set device failed, errorCode is', ret)

        self.context, ret = acl.rt.create_context(self.device_id)
        if ret != ACL_SUCCESS:
            print('create context failed, errorCode is', ret)

        self.stream, ret = acl.rt.create_stream()
        if ret != ACL_SUCCESS:
            print('create stream failed, errorCode is', ret)

        # load model from file
        self.model_id, ret = acl.mdl.load_from_file(self.model_path)
        if ret != ACL_SUCCESS:
            print('load model failed, errorCode is', ret)

        # create description of model
        self.model_desc = acl.mdl.create_desc()
        ret = acl.mdl.get_desc(self.model_desc, self.model_id)
        if ret != ACL_SUCCESS:
            print('get desc failed, errorCode is', ret)

        # create data set of input
        self.input_dataset = acl.mdl.create_dataset()
        input_index = 0
        self.input_buffer_size = acl.mdl.get_input_size_by_index(self.model_desc, input_index)
        self.input_buffer, ret = acl.rt.malloc(self.input_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
        input_data = acl.create_data_buffer(self.input_buffer, self.input_buffer_size)
        self.input_dataset, ret = acl.mdl.add_dataset_buffer(self.input_dataset, input_data)
        if ret != ACL_SUCCESS:
            print('acl.mdl.add_dataset_buffer failed, errorCode is', ret)

        # create data set of output
        self.output_dataset = acl.mdl.create_dataset()
        output_index = 0
        output_buffer_size = acl.mdl.get_output_size_by_index(self.model_desc, output_index)
        self.output_buffer, ret = acl.rt.malloc(output_buffer_size, ACL_MEM_MALLOC_HUGE_FIRST)
        output_data = acl.create_data_buffer(self.output_buffer, output_buffer_size)
        self.output_dataset, ret = acl.mdl.add_dataset_buffer(self.output_dataset, output_data)
        if ret != ACL_SUCCESS:
            print('acl.mdl.add_dataset_buffer failed, errorCode is', ret)

    def process_input(self, input_data):
        image = input_data.astype('float16')
        self.image = image
        self.image_bytes = np.frombuffer(image.tobytes())


    def inference(self):
        # pass image data to input data buffer
        if self.runMode_ == ACL_DEVICE:
            kind = ACL_MEMCPY_DEVICE_TO_DEVICE
        else:
            kind = ACL_MEMCPY_HOST_TO_DEVICE
        if "bytes_to_ptr" in dir(acl.util):
            bytes_data = self.image_bytes.tobytes()
            self.bytes_input = bytes_data
            ptr = acl.util.bytes_to_ptr(bytes_data)
        else:
            ptr = acl.util.numpy_to_ptr(self.image_bytes)
        ret = acl.rt.memcpy(self.input_buffer,
                            self.input_buffer_size,
                            ptr,
                            self.input_buffer_size,
                            kind)

        if ret != ACL_SUCCESS:
            print('memcpy failed, errorCode is', ret)

        # inference
        ret = acl.mdl.execute(self.model_id,
                              self.input_dataset,
                              self.output_dataset)
        if ret != ACL_SUCCESS:
            print('execute failed, errorCode is', ret)

    def get_result(self):
        # get result from output data set
        output_index = 0
        output_data_buffer = acl.mdl.get_dataset_buffer(self.output_dataset, output_index)
        output_data_buffer_addr = acl.get_data_buffer_addr(output_data_buffer)
        output_data_size = acl.get_data_buffer_size(output_data_buffer)
        ptr, ret = acl.rt.malloc_host(output_data_size)

        # # copy device output data to host
        # if self.runMode_ == ACL_DEVICE:
        #     kind = ACL_MEMCPY_DEVICE_TO_HOST
        # else:
        #     kind = ACL_MEMCPY_HOST_TO_HOST

        

        ret = acl.rt.memcpy(ptr,
                            output_data_size,
                            output_data_buffer_addr,
                            output_data_size,
                            ACL_MEMCPY_DEVICE_TO_HOST)
        if ret != ACL_SUCCESS:
            print('memcpy failed, errorCode is', ret)

        index = 0
        dims, ret = acl.mdl.get_cur_output_dims(self.model_desc, index)

        if ret != ACL_SUCCESS:
            print('get output dims failed, errorCode is', ret)
        out_dim = dims['dims']

        if "ptr_to_bytes" in dir(acl.util):
            bytes_data = acl.util.ptr_to_bytes(ptr, output_data_size)
            data = np.frombuffer(bytes_data, dtype=np.float32).reshape(out_dim)
        else:
            data = acl.util.ptr_to_numpy(ptr, out_dim, NPY_FLOAT32)

        self.output_patch_list = data

    def release_resource(self):
        # release resource includes acl resource, data set and unload model
        acl.rt.free(self.input_buffer)
        acl.mdl.destroy_dataset(self.input_dataset)

        acl.rt.free(self.output_buffer)
        acl.mdl.destroy_dataset(self.output_dataset)
        ret = acl.mdl.unload(self.model_id)
        if ret != ACL_SUCCESS:
            print('unload model failed, errorCode is', ret)

        if self.model_desc:
            acl.mdl.destroy_desc(self.model_desc)
            self.model_desc = None

        if self.stream:
            ret = acl.rt.destroy_stream(self.stream)
            if ret != ACL_SUCCESS:
                print('destroy stream failed, errorCode is', ret)
            self.stream = None

        if self.context:
            ret = acl.rt.destroy_context(self.context)
            if ret != ACL_SUCCESS:
                print('destroy context failed, errorCode is', ret)
            self.context = None

        ret = acl.rt.reset_device(self.device_id)
        if ret != ACL_SUCCESS:
            print('reset device failed, errorCode is', ret)

        ret = acl.finalize()
        if ret != ACL_SUCCESS:
            print('finalize failed, errorCode is', ret)

def main():
    t1 = time.time()
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
    model_path = './pkcin4any.om'
    print('Now inference ', model_path)

    if not os.path.exists(model_path):
        raise Exception("the model is not exist")
    net = Model_Inference(device, model_path)
    net.init_resource()

    #PKC inference

    dataset_fpath = Path(args.input_path)
    seq_names = os.listdir(dataset_fpath)

    rd_metrics = Evaluator(num_class=num_class)
    folder_path = Path(args.output_path)

    


    seq_names = sorted(seq_names)
    count = 0
    inference_time = 0

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
    #        print(init_frame_name)
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
            net.process_input(rd_matrix)
            # inference
            time_stamp_1 = time.time()
            net.inference()
            net.get_result()
            time_stamp_2 = time.time()
            inference_time = inference_time + time_stamp_2 - time_stamp_1

            outputs = net.output_patch_list
            count = count + 1

            #debug
            output_list.append(outputs)

            # 先numpy， torch上互转为了debug方便，后续提速全在numpy上操作
            
            # outputs = torch.from_numpy(outputs).float()
            # rd_metrics.add_batch(np.argmax(rd_mask, axis=1), torch.argmax(outputs, axis=1).cpu())

            rd_metrics.add_batch(np.argmax(rd_mask, axis=1), np.argmax(outputs, axis=1))

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

    output_list = np.array(output_list)
    np.save('outputs_debug.npy', output_list)
    t2 = time.time()

    print(f'Acc      : {acc}')
    print(f'Precision: {prec}')
    print(f'Recall   : {recall}')
    print(f'mIoU     : {miou}')
    print(f'Dice     : {dice}')
    print(f'confusion_matrix     : {confusion_matrix}')

    print('Total running time:', t2 - t1)
    print('time per iter:', inference_time/count)
    print("*****run finish******")
    net.release_resource()

if __name__ == '__main__':
    main()
