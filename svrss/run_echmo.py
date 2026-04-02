"""Main script to train a model"""
# End-to-end RSS network runing script
# Echo in Mask out: EchMo zlw@20250317
import argparse
import importlib
import json
from svrss.utils.functions import count_params
from svrss.learners.initializer import Initializer
from svrss.learners.model import Model as LegacyModel
from svrss.learners.model_improved import Model as ImprovedModel
from svrss.models import EchMo, EchMo4MoE, EchMoDeep, EchMoMemo, EchMoWide, rEchMo
import torch.nn as nn
import torch
from svrss.utils.paths import Paths
from svrss.utils.distributed_utils import init_distributed_mode, get_rank, get_world_size


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json', type=str, required=True)
    parser.add_argument('--dataset', default='Carrada_RD', help='dataset for model training,\
                        options: Carrada_RD, Carrada_RA, CarradaRAC_RD, CarradaRAC_RA, CWRdata, PDRdata, RDSyn, XPDdata, NuaaSim, LSFJamming', type=str)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--sync-bn', dest='sync_bn', help='Use sync batch norm', action='store_true')
    parser.add_argument('--finetune', help='the finetune path of model', action='store_true')
    parser.add_argument('--profile', action='store_true', help='Run THOP FLOPs profiling before training')
    args = parser.parse_args()

    init_distributed_mode(args)
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed
    if args.profile and args.distributed and get_world_size() > 1:
        if get_rank() == 0:
            print('[Profile] Disabled in DDP mode to avoid rank desync/timeout. Use single-process run for profiling.')
        args.profile = False
    if cfg['dataset'] in ['CWRdata', 'Carrada_RA', 'Carrada_RD', 'CarradaRAC_RD', 'CarradaRAC_RA']:
        cfg['nb_classes'] = 4
    elif cfg['dataset'] == 'PDRdata':
        cfg['nb_classes'] = 5
    elif cfg['dataset'] in ['XPDdata', 'RDSyn', 'NuaaSim']:
        cfg['nb_classes'] = 2
    elif cfg['dataset'] == 'LSFJamming':
        cfg['nb_classes'] = 7
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))

    trainer_impl = str(cfg.get('trainer_impl', 'legacy')).lower()
    if trainer_impl in ['improved', 'model_improved']:
        trainer_cls = ImprovedModel
    else:
        trainer_cls = LegacyModel
    

    init = Initializer(cfg)
    data = init.get_data()

    class EchMo_CFG:
        def __init__(self):
            self.task_name = 'Segmentation'

    model_cfg = EchMo_CFG()
    model_cfg.task_name = data['cfg']['task_name']
    model_cfg.coef_path = 'coefs'
    if cfg['dataset'] == 'LSFJamming':
        model_cfg.dataset_path = str(Paths().get()['LSFJamming'])
    else:
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
    model_cfg.lmf_impl = data['cfg'].get('lmf_impl', 'v6')
    model_cfg.lmf_chain_impl = data['cfg'].get('lmf_chain_impl', 'serial')
    # mldc config
    model_cfg.ldc_channels = data['cfg']['ldc_channels']
    model_cfg.ldc_orders = data['cfg']['ldc_orders']
    model_cfg.mldc_impl = data['cfg'].get('mldc_impl', 'serial')
    # gdf  config
    model_cfg.df_channels = data['cfg']['nb_doppler']
    model_cfg.gdf_impl = data['cfg'].get('gdf_impl', 'conv')
    # global settings
    model_cfg.activation = data['cfg']['activation']
    model_cfg.bias = data['cfg']['bias']
    # wide model optional settings
    model_cfg.wide_groups = data['cfg'].get('wide_groups', 4)
    model_cfg.wide_rank = data['cfg'].get('wide_rank', 32)
    model_cfg.wide_share_mldc = data['cfg'].get('wide_share_mldc', True)
    model_cfg.wide_fuse_residual = data['cfg'].get('wide_fuse_residual', True)
    model_cfg.deep_stage2_enable = data['cfg'].get('deep_stage2_enable', True)
    model_cfg.deep_stage2_range_scale = data['cfg'].get('deep_stage2_range_scale', 0.5)
    model_cfg.deep_stage2_pulse_scale = data['cfg'].get('deep_stage2_pulse_scale', 1.0)
    model_cfg.deep_stage2_chain_scale = data['cfg'].get('deep_stage2_chain_scale', 1.0)
    model_cfg.deep_stage2_lmf_channel_scale = data['cfg'].get('deep_stage2_lmf_channel_scale', 1.0)
    model_cfg.deep_stage2_ldc_channel_scale = data['cfg'].get('deep_stage2_ldc_channel_scale', 1.0)
    model_cfg.memo_enable = data['cfg'].get('memo_enable', True)
    model_cfg.memo_beta = data['cfg'].get('memo_beta', 0.8)
    model_cfg.memo_gamma = data['cfg'].get('memo_gamma', 0.4)
    model_cfg.memo_use_state_proj = data['cfg'].get('memo_use_state_proj', True)
    model_cfg.moe_num_experts = data['cfg'].get('moe_num_experts', 4)
    model_cfg.moe_topk = data['cfg'].get('moe_topk', 2)
    model_cfg.moe_gate_hidden = data['cfg'].get('moe_gate_hidden', 32)
    model_cfg.moe_temperature = data['cfg'].get('moe_temperature', 1.0)
    model_cfg.moe_dropout = data['cfg'].get('moe_dropout', 0.0)

    add_temp = False
    if cfg['model'] == 'echmo':
        net = EchMo(configs=model_cfg)
    elif cfg['model'] == 'echmo_deep':
        net = EchMoDeep(configs=model_cfg)
    elif cfg['model'] == 'echmo_memo':
        net = EchMoMemo(configs=model_cfg)
    elif cfg['model'] == 'echmo_4moe':
        net = EchMo4MoE(configs=model_cfg)
    elif cfg['model'] == 'echmo_wide':
        net = EchMoWide(configs=model_cfg)
    elif cfg['model'] == 'rechmo':
        net = rEchMo(configs=model_cfg)
    elif cfg['model'] == 'techmo':
        raise NotImplementedError('techmo is referenced but not implemented in this entrypoint.')
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
        if param.requires_grad and (name.startswith('LMFChain') or name.startswith('MLDC') or name.startswith('GDF')):
            print(f"Param: {name}, #Params: {param.numel()}")
    total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f">>>>>Total #Params in Model: {total_params}")
    if args.profile and get_rank() == 0:
        profile = importlib.import_module('thop').profile

        if add_temp:
            dummy_input = torch.randn(1, 1, model_cfg.n_frames, model_cfg.n_pulse, model_cfg.n_range)
        else:
            dummy_input = torch.randn(1, 1, model_cfg.n_pulse, model_cfg.n_range)
        flops, _ = profile(net.cuda(), inputs=(torch.complex(dummy_input, dummy_input).cuda(),))
        print(f">>>>>Model Computing Cosuming (FLOPs) per Sample: {flops}")
    if args.distributed:
        # Keep all ranks aligned after rank0-only setup such as THOP profiling.
        torch.distributed.barrier()
    print('---------------------------------------------------')

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

    trainer = trainer_cls(net, data)
    if get_rank() == 0:
        print(f">>>>>Trainer Impl: {trainer_cls.__module__}.{trainer_cls.__name__}")

    train_kwargs = {}
    if trainer_cls is ImprovedModel:
        train_kwargs['checkpoint_path'] = cfg.get('checkpoint_path')
        train_kwargs['use_global_dataset'] = bool(cfg.get('use_global_dataset', False))

    if cfg['model'] in ['mvnet', 'fcn8s', 'unet', 'deeplabv3plus', 'kuradnet', 'kuradnet_wskip', 'pkcin4any_2d', 'ecoss', 'echmo', 'echmo_4moe', 'echmo_deep', 'echmo_memo', 'echmo_wide', 'rechmo']:
        trainer.train(add_temp=False, **train_kwargs)
    else:
        trainer.train(add_temp=True, **train_kwargs)
    
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
