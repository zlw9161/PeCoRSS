"""Main script to train a model"""
# End-to-end RSS network runing script
# Echo in Mask out: EchMo zlw@20250317
import argparse
import json
from svrss.utils.functions import count_params
from svrss.learners.initializer import Initializer
from svrss.learners.model import Model
from svrss.models import EchMo, rEchMo
import torch.nn as nn
import torch
from svrss.utils.distributed_utils import init_distributed_mode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json', type=str, required=True)
    parser.add_argument('--dataset', default='Carrada_RD', help='dataset for model training,\
                        options: Carrada_RD, Carrada_RA, CarradaRAC_RD, CarradaRAC_RA, CWRdata, PDRdata, RDSyn, XPDdata', type=str)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--sync-bn', dest='sync_bn', help='Use sync batch norm', action='store_true')
    parser.add_argument('--finetune', help='the finetune path of model', action='store_true')
    args = parser.parse_args()

    init_distributed_mode(args)
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed
    if cfg['dataset'] in ['CWRdata', 'Carrada_RA', 'Carrada_RD', 'CarradaRAC_RD', 'CarradaRAC_RA']:
        cfg['nb_classes'] = 4
    elif cfg['dataset'] == 'PDRdata':
        cfg['nb_classes'] = 5
    elif cfg['dataset'] in ['XPDdata', 'RDSyn', 'NuaaSim']:
        cfg['nb_classes'] = 2
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))
    

    init = Initializer(cfg)
    data = init.get_data()

    class EchMo_CFG:
        def __init__(self):
            self.task_name = 'Segmentation'

    model_cfg = EchMo_CFG()
    model_cfg.task_name = data['cfg']['task_name']
    model_cfg.coef_path = 'coefs'
    model_cfg.dataset_path = '/data/PeCoLab/datasets/NuaaSim'
    # data setting
    model_cfg.n_pulse = data['cfg']['w_size']
    model_cfg.n_range = data['cfg']['h_size']
    model_cfg.n_classes = data['cfg']['nb_classes']
    model_cfg.n_frames = data['cfg']['nb_input_channels']
    model_cfg.scale = data['cfg']['norm_type']
    # lmf config
    model_cfg.lmf_chains = data['cfg']['lmf_chains']
    model_cfg.lmf_channels = data['cfg']['lmf_channels']
    model_cfg.lmf_layers = data['cfg']['lmf_layers']
    model_cfg.lmf_coefs = data['cfg']['lmf_coefs']
    # mldc config
    model_cfg.ldc_channels = data['cfg']['ldc_channels']
    model_cfg.ldc_orders = data['cfg']['ldc_orders']
    # gdf  config
    model_cfg.df_channels = data['cfg']['nb_doppler']
    # global settings
    model_cfg.activation = data['cfg']['activation']
    model_cfg.bias = data['cfg']['bias']

    add_temp = False
    if cfg['model'] == 'echmo':
        net = EchMo(configs=model_cfg)
    elif cfg['model'] == 'rechmo':
        net = rEchMo(configs=model_cfg)
    elif cfg['model'] == 'techmo':
        net = tEchMo(configs=model_cfg)
        add_temp = True
    else:
        print('Please check model NAME !!!')

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))
    print('>>>>>Model Info.<<<<<')
    print('>>>>>Model CFG:', model_cfg)
    for name, module in net.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.Parameter) or isinstance(module, nn.Conv3d):
            param_count = sum(p.numel() for p in module.parameters() if p.requires_grad)
            print(f"Layer: {name}, #Params: {param_count}")
    # Also show trainable tensors that are registered as raw Parameters (not inside Conv/Linear modules). @20260318
    for name, param in net.named_parameters():
        if param.requires_grad and (name.startswith('LMFChain') or name.startswith('GDF')):
            print(f"Param: {name}, #Params: {param.numel()}")
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f">>>>>Total #Params in Model: {total_params}")
    from thop import profile
    if add_temp:
        dummy_input = torch.randn(data['cfg']['batch_size'], 1, model_cfg.n_frames, model_cfg.n_pulse, model_cfg.n_range)
    else:
        dummy_input = torch.randn(data['cfg']['batch_size'], 1, model_cfg.n_pulse, model_cfg.n_range)
    flops, _ = profile(net.cuda(), inputs=(torch.complex(dummy_input, dummy_input).cuda(),))
    print(f">>>>>Model Computing Cosuming (FLOPs): {flops}")
    print('----------------------')

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model)
    
    net.to(device)
    # # lt @20230522 fix random seed
    torch.manual_seed(cfg['torch_seed'])

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net)
    
    if cfg['model'] in ['mvnet', 'fcn8s', 'unet', 'deeplabv3plus', 'kuradnet', 'kuradnet_wskip', 'pkcin4any_2d', 'ecoss', 'echmo', 'rechmo']:
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)
    
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
