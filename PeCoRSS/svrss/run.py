"""Main script to train a model"""
import argparse
import json
from svrss.utils.functions import count_params
from svrss.learners.initializer import Initializer
from svrss.learners.model import Model
from svrss.models import CWRNet, PKCIn4CWR, PKCIn4CWR2D, PKCIn4CWR_Dns, TARSS4CWR_V1, TARSS4CWR_V2
from svrss.models import KuRADNet, KuRADNet_WSkip, PKCIn4PDR, PKCIn4PDR_Dns, TARSS4PDR_V1, TARSS4PDR_V2
from svrss.models import BaseLine3D, TARSSNet_V1_RD, TARSSNet_V1_RA, TARSSNet_V2_RD, TARSSNet_V2_RA
from svrss.models import UNet, FCN8s, DeepLabV3Plus_Res101, PKCIn4ANY, PKCIn4ANY_cuda
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
    elif cfg['dataset'] == 'XPDdata':
        cfg['nb_classes'] = 2
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))
    

    init = Initializer(cfg)
    data = init.get_data()
    if cfg['model'] == 'cwrnet':
        net = CWRNet(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4cwr':
        net = PKCIn4CWR(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4cwr_2d':
        net = PKCIn4CWR2D(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4cwr_dns':
        net = PKCIn4CWR_Dns(n_classes=data['cfg']['nb_classes'],
                            n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarss4cwr_v1':
        net = TARSS4CWR_V1(n_classes=data['cfg']['nb_classes'],
                           n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarss4cwr_v2':
        net = TARSS4CWR_V2(n_classes=data['cfg']['nb_classes'],
                           n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'kuradnet':
        net = KuRADNet(n_classes=data['cfg']['nb_classes'],
                       n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'kuradnet_wskip':
        net = KuRADNet_WSkip(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4pdr':
        net = PKCIn4PDR(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4pdr_dns':
        net = PKCIn4PDR_Dns(n_classes=data['cfg']['nb_classes'],
                            n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4pdr':
        net = PKCIn4PDR(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarss4pdr_v1':
        net = TARSS4PDR_V1(n_classes=data['cfg']['nb_classes'],
                           n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarss4pdr_v2':
        net = TARSS4PDR_V2(n_classes=data['cfg']['nb_classes'],
                           n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin4any':
        net = PKCIn4ANY(n_classes=data['cfg']['nb_classes'],
                        n_frames=data['cfg']['nb_input_channels'],
                        w=data['cfg']['w_size'],
                        h=data['cfg']['h_size'])
    elif cfg['model'] == 'pkcin4any_cuda':
        net = PKCIn4ANY_cuda(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'],
                             w=data['cfg']['w_size'],
                             h=data['cfg']['h_size'])
    elif cfg['model'] == 'baseline3d':
        if cfg['dataset'] in ['CarradaRAC_RA', 'Carrada_RA']:
            net = BaseLine3D(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'],
                             signal_type='range_angle')
        else:
            net = BaseLine3D(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'],
                             signal_type='range_doppler')
    elif cfg['model'] == 'tarssnet_v1_rd':
        net = TARSSNet_V1_RD(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarssnet_v1_ra':
        net = TARSSNet_V1_RA(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarssnet_v2_rd':
        net = TARSSNet_V2_RD(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tarssnet_v2_ra':
        net = TARSSNet_V2_RA(n_classes=data['cfg']['nb_classes'],
                             n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'deeplabv3plus':
        net = DeepLabV3Plus_Res101(n_classes=data['cfg']['nb_classes'],
                                   n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'unet':
        net = UNet(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'fcn8s':
        net = FCN8s(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    else:
        net = UNet(n_classes=data['cfg']['nb_classes'],
                   n_frames=data['cfg']['nb_input_channels'])

    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)
    
    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model)
    
    net.to(device)
    # # lt @20230522 fix random seed
    torch.manual_seed(cfg['torch_seed'])
    net.apply(_init_weights)

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net)
    
    if cfg['model'] in ['mvnet', 'fcn8s', 'unet', 'deeplabv3plus', 'kuradnet', 'kuradnet_wskip']:
        Model(net, data).train(add_temp=False)
    else:
        Model(net, data).train(add_temp=True)
    
    if args.distributed:
        torch.distributed.destroy_process_group()

def _init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias, 0.)
    elif isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.uniform_(m.weight, 0., 1.)
            nn.init.constant_(m.bias, 0.)

# def _init_weights(m):
#     if isinstance(m, nn.Linear):
#         torch.nn.init.xavier_uniform_(m.weight)
#         nn.init.constant_(m.bias, 0.)
#     elif isinstance(m, nn.Conv2d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.)
#     elif isinstance(m, nn.Conv3d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.)
#     elif isinstance(m, nn.ConvTranspose2d):
#         torch.nn.init.xavier_uniform_(m.weight)
#         if m.bias is not None:
#             nn.init.constant_(m.bias, 0.)
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.uniform_(m.weight, 0., 1.)
#         nn.init.constant_(m.bias, 0.)
#     elif isinstance(m, nn.BatchNorm3d):
#         nn.init.uniform_(m.weight, 0., 1.)
#         nn.init.constant_(m.bias, 0.)

if __name__ == '__main__':
    main()
