"""Main script to train a model"""
import argparse
import json
from svrss.utils.functions import count_params
from svrss.learners.initializer import Initializer
from svrss.learners.model import Model
from svrss.models import ECoSS
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

    class ECoSS_CFG:
        def __init__(self):
            self.task_name = 'Segmentation'

    model_cfg = ECoSS_CFG()
    model_cfg.task_name = data['cfg']['task_name']
    model_cfg.n_pulse = data['cfg']['w_size']
    model_cfg.n_range = data['cfg']['h_size']
    model_cfg.n_classes = data['cfg']['nb_classes']
    model_cfg.n_frames = data['cfg']['nb_input_channels']
    model_cfg.e_chains = data['cfg']['e_chains']
    model_cfg.e_channels = data['cfg']['e_channels']
    model_cfg.e_layers = data['cfg']['e_layers']
    model_cfg.e_doppler = data['cfg']['e_doppler']
    model_cfg.coef_type = data['cfg']['coef_type']
    model_cfg.scale = data['cfg']['norm_type']
    model_cfg.activation = data['cfg']['activation']
    model_cfg.bias = data['cfg']['bias']
    model_cfg.coef_path = 'coefs'
    model_cfg.dataset_path = 'E:\\DataSets\\NuaaSim'

    print(model_cfg.scale)

    if cfg['model'] == 'ecoss':
        net = ECoSS(configs=model_cfg)
    else:
        print('Please check model NAME !!!')

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

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
    
    if cfg['model'] in ['mvnet', 'fcn8s', 'unet', 'deeplabv3plus', 'kuradnet', 'kuradnet_wskip', 'pkcin4any_2d', 'ecoss']:
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)
    
    if args.distributed:
        torch.distributed.destroy_process_group()

if __name__ == '__main__':
    main()
