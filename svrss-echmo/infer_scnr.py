import os
import numpy as np
import torch
import cv2
from skimage import transform
from pathlib import Path
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from svrss.utils.functions import transform_masks_viz, get_metrics, normalize, define_loss, get_transformations, get_qualitatives
from svrss.loaders.dataset import CWRdata, XPDdata, RDSyn, NuaaSim
from svrss.utils.paths import Paths
from svrss.models import  PKCIn4ANY
from svrss.utils.metrics import Evaluator
from svrss.loaders.dataloaders import SequenceDataset,PeCoRSVDataset
from svrss.utils.Detector import CFAR_2D
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
def netPara():
    return {
        'n_classes':2,
        'n_frames':1,
        'w':128,
        'h':1024,
    }

def mask2bbox(mask, frmID, seq_path):
    _, _, stats, centroids = cv2.connectedComponentsWithStats(image=mask.astype(np.uint8),
                                                              connectivity=8)
    stats = stats[stats[:,4].argsort()]
    bbox_dict = {}
    bbox_dict['frmPath'] = os.path.join(seq_path, frmID + '.npy')
    idx = 0
    for b in stats[:-1, :]:
        tgtName = 'tgtID_' + str(idx)
        frmID = 'frmID_' + frmID
        bbox_dict[frmID] = {}
        bbox_dict[frmID][tgtName] = {'label': None, 'bbox': {}}
        bbox_dict[frmID][tgtName]['label'] = 'UAV'
        bbox_dict[frmID][tgtName]['bbox']['point'] = (int(b[0]-1), int(b[1]-1)) # start from 1
        bbox_dict[frmID][tgtName]['bbox']['width'] = int(b[2])
        bbox_dict[frmID][tgtName]['bbox']['height'] = int(b[3])
        bbox_dict[frmID][tgtName]['bbox']['area'] = int(b[4])
        idx = idx + 1

    return stats[:-1], bbox_dict

def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        # json.dump(data, f, indent=2, ensure_ascii=False)
        json.dump(data, f)

def cfar_eval():
    det = CFAR_2D('CA',referenceLen=8,protectLen=2,factor=1)
    pf = 1.81e-05

    paths = Paths().get()
    n_frames = 1
    dataset = NuaaSim().get('Test')
    signal_type = 'range_doppler'

    rd_metrics = Evaluator(num_class=2)
    criterion = define_loss('NuaaSim', 'range_doppler', 'ce', 'cpu')

    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=False, num_workers=0)
    running_losses = list()
    rd_running_losses = list()
    rd_running_global_losses = [list(), list()]

    clutter_amplitude = []
    cluter_threshold = []
    TP = []

    for k, data in enumerate(seq_dataloader):
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

        for i, frame in enumerate(frame_dataloader):
            print(path_to_frames)
            data = frame['rd_matrix'].squeeze()
            rd_mask = frame['rd_mask'][0,1,:,:].squeeze()
            rd_output, threshold = det.filter(data)

            #25
            if int(rd_mask.sum()) != 25:
                continue
            TP.append(rd_output[rd_mask == 1])

            # assert (int(rd_mask.sum()) != 25)
            # print(f'rd_mask is {rd_mask.sum()} \n  frame id is {k}_{i}')

            clutter_amplitude.append(data[rd_mask == 0])
            cluter_threshold.append(abs(threshold[rd_mask == 0]))

            print(i)

        #     if (k+1) * i >=1000:
        #         break
        #
        # if (k + 1) * i >= 1000:
        #     break

    clutter_amplitude = torch.stack((clutter_amplitude)).flatten()
    cluter_threshold = torch.stack((cluter_threshold)).flatten()
    TP = torch.stack((TP)).flatten()
    pd = TP.sum()/TP.numel()

    print(f'pd : {pd}')
    print(f'number of samples : {cluter_threshold.numel()}')


    ratio = torch.div(clutter_amplitude, cluter_threshold)
    k_value = torch.round(torch.mul(pf, clutter_amplitude.shape[0])).long()
    # if k_value == 0:
    #     k_value += 1
    # else:
    #     k_value = k_value.item()

    # N = 100
    # offset = torch.arange(0,int(3.5e8),step=int(3.5e8/N))
    # sub_ratios = torch.empty(0)
    # sub_indexs = torch.empty(0)
    #
    # for i in range(N-1):
    #     print(f'{i}/{N-1}')
    #     if offset[i+1]>= ratio.numel():
    #         sub_ratio, sub_index = torch.topk(ratio[offset[i]:], k=k_value.item())
    #         sub_ratios = torch.cat((sub_ratios,sub_ratio))
    #         sub_indexs = torch.cat((sub_indexs,sub_index + offset[i]))
    #         break
    #     else:
    #         sub_ratio,sub_index = torch.topk(ratio[offset[i]:offset[i+1]], k=k_value.item())
    #         sub_ratios = torch.cat((sub_ratios, sub_ratio))
    #         sub_indexs = torch.cat((sub_indexs, sub_index + offset[i]))
    #
    #
    # ratio,idx = torch.topk(sub_ratios,k=k_value.item())
    # index = sub_indexs[idx].long()
    ratio,index = torch.topk(ratio, k=k_value.item())

    clutter_amplitude = clutter_amplitude[index]
    cluter_threshold = cluter_threshold[index]

    # print(f'order is {k_value.item()}')
    print(f'order is {k_value}')
    factor = clutter_amplitude[-1]/cluter_threshold[-1]
    print(f'threshold factor: {factor}')


    TP = []
    FN = []
    det = CFAR_2D('CA', referenceLen=8, protectLen=2, factor=factor)

    rd_metrics = Evaluator(num_class=2)

    for k, data in enumerate(seq_dataloader):
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
        for i, frame in enumerate(frame_dataloader):
            print('round2:{}'.format(path_to_frames))
            data = frame['rd_matrix'].squeeze()
            rd_mask = frame['rd_mask']
            rd_output, threshold = det.filter(data)
            rd_output = rd_output[None, None, :, :]
            rd_output = torch.cat((1 - rd_output, rd_output), dim=1)

            # 25
            if int(rd_mask[0,1,...].sum()) != 25:
                continue

            rd_metrics.add_batch(torch.argmax(rd_mask, axis=1),
                                      torch.argmax(rd_output, axis=1))
            rd_losses = [c(rd_output, torch.argmax(rd_mask, axis=1))
                         for c in criterion]
            rd_loss = torch.mean(torch.stack(rd_losses))
            loss = rd_loss

            running_losses.append(loss.data.cpu().numpy()[()])
            rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
            rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
            rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
            print(i)

    matrix = rd_metrics.confusion_matrix
    pd = matrix[1][1] / sum(matrix[1])
    pf = matrix[0][1] / sum(matrix[0])

    print(f'pd:{pd} \n pf:{pf}')
    test_results = dict()
    test_results['range_doppler'] = get_metrics(rd_metrics,
                                                np.mean(rd_running_losses),
                                                [np.mean(sub_loss) for sub_loss in rd_running_global_losses])

    test_results['global_acc'] = test_results[signal_type]['acc']
    test_results['global_prec'] = test_results[signal_type]['prec']
    dice = test_results[signal_type]['dice']
    test_results['global_recall'] = test_results[signal_type]['recall']
    miou = test_results[signal_type]['miou']

    print(f'dice:{dice} \n miou:{miou}')

    rd_metrics.reset()


def net_eval():
    paths = Paths().get()
    n_frames = 1
    device = 'cuda'
    dataset = NuaaSim().get('Test')
    signal_type = 'range_doppler'
    pkc = PKCIn4ANY(**netPara())
    pkc.load_state_dict(torch.load('/home/liwenzhang/PeCoLab/logs/NuaaSim/pkcin4any/pkcin4any_e300_lr0.0001_ce_2/results/pkcin4any.pt'))
    pkc.to(device)

    rd_metrics = Evaluator(num_class=2)
    criterion = define_loss('NuaaSim', 'range_doppler', 'ce', 'cuda')

    seq_dataloader = DataLoader(SequenceDataset(dataset), batch_size=1,
                                shuffle=False, num_workers=0)
    running_losses = list()
    rd_running_losses = list()
    rd_running_global_losses = [list(), list()]

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
        for i, frame in enumerate(frame_dataloader):
            print(path_to_frames, seq[i][0])
            data = frame['rd_matrix'].unsqueeze(0).float().to(device)
            rd_mask = frame['rd_mask'].float().to(device)
            rd_outputs = pkc(data)
            rd_metrics.add_batch_dist(torch.argmax(rd_mask, axis=1),
                                      torch.argmax(rd_outputs, axis=1))
            pred_mask = torch.argmax(rd_outputs, axis=1)
            mask = np.array(pred_mask.squeeze().cpu().detach().numpy())
            stats, bbox_dict = mask2bbox(mask, seq[i][0], path_to_frames)
            bbox_save_name = str(path_to_frames).split('/')[-1] + '_' + seq[i][0] + '.json'
            os.makedirs(os.path.join('/home/liwenzhang/PeCoLab/logs/NuaaSim/pkcin4any/pkcin4any_e300_lr0.0001_ce_2/results',
                                     'bbox_results'), exist_ok=True)
            bbox_save_path = os.path.join('/home/liwenzhang/PeCoLab/logs/NuaaSim/pkcin4any/pkcin4any_e300_lr0.0001_ce_2/results',
                                          'bbox_results',
                                          bbox_save_name)
            save_json(bbox_dict, bbox_save_path)

            print(bbox_dict)
            ax = plt.axes()
            plt.imshow(mask,cmap="bone")
            for j in stats:
                rect = patches.Rectangle((j[0]-1,j[1]-1),j[2],j[3],linewidth = 1,edgecolor = 'r',facecolor = 'none')
                ax.add_patch(rect)
            # plt.show()

            rd_losses = [c(rd_outputs, torch.argmax(rd_mask, axis=1))
                         for c in criterion]
            rd_loss = torch.mean(torch.stack(rd_losses))
            loss = rd_loss

            running_losses.append(loss.data.cpu().numpy()[()])
            rd_running_losses.append(rd_loss.data.cpu().numpy()[()])
            rd_running_global_losses[0].append(rd_losses[0].data.cpu().numpy()[()])
            rd_running_global_losses[1].append(rd_losses[1].data.cpu().numpy()[()])
            print(i)

    matrix = rd_metrics.confusion_matrix
    print(matrix)
    pd = matrix[1][1]/sum(matrix[1])
    pf = matrix[0][1]/sum(matrix[0])

    np.save('./net_pf_18dB.npy',pf)
    np.save('./net_pd_18dB.npy', pd)
    print(f'pd:{pd} \n pf:{pf}')
    test_results = dict()
    test_results['range_doppler'] = get_metrics(rd_metrics, np.mean(rd_running_losses),
                                                 [np.mean(sub_loss) for sub_loss
                                                  in rd_running_global_losses])

    test_results['global_acc'] = test_results[signal_type]['acc']
    test_results['global_prec'] = test_results[signal_type]['prec']
    dice = test_results[signal_type]['dice']
    # add global recall and mIoU zlw@20220704
    test_results['global_recall'] = test_results[signal_type]['recall']
    miou = test_results[signal_type]['miou']

    print(f'dice:{dice} \n miou:{miou}')


    rd_metrics.reset()


if __name__ == '__main__':
    # net_eval()
    cfar_eval()