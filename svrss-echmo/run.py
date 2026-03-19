"""Main script to train a model"""
import argparse
import json
import os
import torch
import torch.nn as nn
import copy
from svrss.utils.functions import count_params
from svrss.learners.initializer import Initializer
from svrss.learners.model import Model
from svrss.learners.model_global_dataset import Model_global_dataset
from svrss.models import CWRNet, PKCIn4CWR, PKCIn4CWR2D, PKCIn4CWR_Dns, TARSS4CWR_V1, TARSS4CWR_V2
from svrss.models import KuRADNet, KuRADNet_WSkip, PKCIn4PDR, PKCIn4PDR_Dns, TARSS4PDR_V1, TARSS4PDR_V2
from svrss.models import BaseLine3D, TARSSNet_V1_RD, TARSSNet_V1_RA, TARSSNet_V2_RD, TARSSNet_V2_RA
from svrss.models import UNet, FCN8s, DeepLabV3Plus_Res101, PKCIn4ANY, PKCIn4ANY_cuda, PKCIn4ANY_2D
from svrss.models import TMVANet_SV, ViT
from svrss.models import EchMo, rEchMo
from svrss.utils.distributed_utils import init_distributed_mode, get_rank

from svrss.utils.paths import Paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', required=True, type=str)
    parser.add_argument('--dataset', default='Carrada_RD', 
                        help='Dataset for model training', type=str)
    parser.add_argument('--dist-url', default='env://', help='Distributed training url')
    parser.add_argument('--sync-bn', dest='sync_bn', action='store_true', help='Use synchronized batch normalization')
    parser.add_argument('--finetune', help='Path to pretrained weights for finetuning', type=str, default=None)
    parser.add_argument('--resume', help='Path to full checkpoint for resuming training', type=str, default=None)
    args = parser.parse_args()

    init_distributed_mode(args)
    rank = get_rank()

    with open(args.cfg, 'r') as fp:
        cfg = json.load(fp)

    cfg['distributed'] = args.distributed
    if args.distributed:
        device = torch.device('cuda', args.gpu)
        torch.cuda.set_device(device) 
    else:
        device = torch.device(cfg['device'])

    if cfg['dataset'] in ['CWRdata', 'Carrada_RA', 'Carrada_RD', 'CarradaRAC_RD', 'CarradaRAC_RA']:
        cfg['nb_classes'] = 4
    elif cfg['dataset'] == 'PDRdata':
        cfg['nb_classes'] = 5
    elif cfg['dataset'] in ['XPDdata', 'RDSyn', 'NuaaSim']:
        cfg['nb_classes'] = 2
    elif cfg['dataset'] in ['LSFJamming']:
        cfg['nb_classes'] = 7
    else:
        raise KeyError(f'Dataset {cfg["dataset"]} is not supported')

    torch.manual_seed(cfg['torch_seed'])

    init = Initializer(cfg)
    data = init.get_data()
    add_temp = False  # Temporal sequence flag

    if cfg['model'] in ['echmo', 'rechmo', 'techmo']:
        class EchMo_CFG:
            def __init__(self):
                self.task_name = 'Segmentation'
        paths = Paths().get()
        model_cfg = EchMo_CFG()
        model_cfg.task_name = data['cfg']['task_name']
        model_cfg.coef_path = 'coefs'
        model_cfg.dataset_path = paths[data['cfg']['dataset']]
        model_cfg.n_pulse = data['cfg']['w_size']
        model_cfg.n_range = data['cfg']['h_size']
        model_cfg.n_classes = data['cfg']['nb_classes']
        model_cfg.n_frames = data['cfg']['nb_input_channels']
        model_cfg.scale = data['cfg']['norm_type']
        model_cfg.lmf_chains = data['cfg']['lmf_chains']
        model_cfg.lmf_channels = data['cfg']['lmf_channels']
        model_cfg.lmf_layers = data['cfg']['lmf_layers']
        model_cfg.lmf_coefs = data['cfg']['lmf_coefs']
        model_cfg.ldc_channels = data['cfg']['ldc_channels']
        model_cfg.ldc_orders = data['cfg']['ldc_orders']
        model_cfg.df_channels = data['cfg']['nb_doppler']
        model_cfg.activation = data['cfg']['activation']
        model_cfg.bias = data['cfg']['bias']
        model_cfg.scale_factor = data['cfg']['scale_factor']

        if cfg['model'] == 'echmo':
            net = EchMo(configs=model_cfg)
        elif cfg['model'] == 'rechmo':
            net = rEchMo(configs=model_cfg)
            add_temp = True

    else:
        if cfg['model'] == 'cwrnet':
            net = CWRNet(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'pkcin4cwr':
            net = PKCIn4CWR(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'pkcin4cwr_2d':
            net = PKCIn4CWR2D(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'pkcin4cwr_dns':
            net = PKCIn4CWR_Dns(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarss4cwr_v1':
            net = TARSS4CWR_V1(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarss4cwr_v2':
            net = TARSS4CWR_V2(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'kuradnet':
            net = KuRADNet(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'kuradnet_wskip':
            net = KuRADNet_WSkip(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'pkcin4pdr':
            net = PKCIn4PDR(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'pkcin4pdr_dns':
            net = PKCIn4PDR_Dns(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarss4pdr_v1':
            net = TARSS4PDR_V1(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarss4pdr_v2':
            net = TARSS4PDR_V2(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'pkcin4any':
            net = PKCIn4ANY(n_frames=data['cfg']['nb_input_channels'])      
        elif cfg['model'] == 'pkcin4any_cuda':
            net = PKCIn4ANY_cuda(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'],
                                 w=data['cfg']['w_size'], h=data['cfg']['h_size'])
        elif cfg['model'] == 'pkcin4any_2d':
            net = PKCIn4ANY_2D(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'],
                               w=data['cfg']['w_size'], h=data['cfg']['h_size'])
        elif cfg['model'] == 'baseline3d':
            signal_type = 'range_angle' if cfg['dataset'] in ['CarradaRAC_RA', 'Carrada_RA'] else 'range_doppler'
            net = BaseLine3D(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'], signal_type=signal_type)
        elif cfg['model'] == 'tarssnet_v1_rd':
            net = TARSSNet_V1_RD(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarssnet_v1_ra':
            net = TARSSNet_V1_RA(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarssnet_v2_rd':
            net = TARSSNet_V2_RD(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tarssnet_v2_ra':
            net = TARSSNet_V2_RA(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'deeplabv3plus':
            net = DeepLabV3Plus_Res101(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'unet':
            net = UNet(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'fcn8s':
            net = FCN8s(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'])
        elif cfg['model'] == 'tmva_net_sv':     
            net = TMVANet_SV(n_classes=data['cfg']['nb_classes'], n_frames=data['cfg']['nb_input_channels'], in_ch=1, w=data['cfg']['w_size'], h=data['cfg']['h_size'])
        elif cfg['model'] == 'vit':     
            net = ViT(n_classes=data['cfg']['nb_classes'], in_chans=data['cfg']['nb_input_channels'])
        else:
            raise ValueError(f"Unsupported model type: {cfg['model']}")

    net = net.to(device)

    if rank == 0:
        total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print(f'Total trainable parameters: {count_params(net)}')
        print(f'Precise parameter count: {total_params}')
        print(f'Model parameter data type: {next(net.parameters()).dtype}')
        if 'model_cfg' in locals():
            print(f'EchMo model configuration: {model_cfg}')
            
    add_temp = not cfg['model'] in ['mvnet', 'fcn8s', 'unet', 'deeplabv3plus', 'kuradnet', 'kuradnet_wskip', 'pkcin4any_2d', 'vit', 'echmo', 'tmva_net_sv']
    from thop import profile, clever_format
    if add_temp:
        dummy_input = torch.randn(data['cfg']['batch_size'], 1, data['cfg']['nb_input_channels'], data['cfg']['w_size'], data['cfg']['h_size'])
    else:
        dummy_input = torch.randn(data['cfg']['batch_size'], 1, data['cfg']['w_size'], data['cfg']['h_size'])
    if cfg['model'] in ['echmo', 'rechmo', 'techmo']:
        dummy_input = torch.complex(dummy_input, dummy_input)
    dummy_input = dummy_input.to(device)

    model_to_profile = copy.deepcopy(net.module) if hasattr(net, 'module') else copy.deepcopy(net)
    macs, params = profile(model_to_profile, inputs=(dummy_input,), verbose=False)
    if rank == 0:
        macs_formatted, params_formatted = clever_format([macs, params], "%.3f")
        print("\n===== Model Computational Complexity =====")
        print(f"GMACs: {macs / 1e9:.2f} | GFLOPs: {macs * 2 / 1e9:.2f}")
        print(f"Parameters: {params / 1e6:.2f} M | Formatted: {macs_formatted}, {params_formatted}")

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if cfg['model'] not in ['echmo', 'rechmo']:
        net.apply(_init_weights)

    if rank == 0:
        if args.finetune and args.resume:
            raise ValueError("--finetune and --resume cannot be used simultaneously!")
        if args.finetune:
            print(f"Finetuning mode: Loading weights from {args.finetune}")
        if args.resume:
            print(f"Resume training: Loading checkpoint from {args.resume}")

    if args.finetune:
        saved_model = torch.load(args.finetune, map_location='cpu', weights_only=True)
        net.load_state_dict(saved_model, strict=False)

    if args.distributed:
        try:
            net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu], find_unused_parameters=False)
            if rank == 0:
                print("DDP distributed training initialized successfully")
        except Exception as e:
            print(f"DDP initialization failed: {e}")


    Model_global_dataset(net, data).train(add_temp=add_temp, checkpoint_path=args.resume)
    # Model(net, data).train(add_temp=add_temp)

    if args.distributed:
        torch.distributed.destroy_process_group()


def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif isinstance(m, (nn.Conv2d, nn.Conv3d)):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm3d)):
        nn.init.uniform_(m.weight, 0., 1.)
        nn.init.constant_(m.bias, 0.)

if __name__ == '__main__':
    main()
