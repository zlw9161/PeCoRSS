"""Main script to train a model"""
import argparse
import json
from mvrss.utils.functions import count_params
from mvrss.utils.distributed_utils import init_distributed_mode
from mvrss.learners.initializer import Initializer
from mvrss.learners.model_F import Model
from mvrss.models import TMVANet, MVNet, MVANet, MVA_DCN, TMVA_DCN, TMVA_TDC, PKCIn, PKCOn, AdaPKC, TFocNet, VitNet
import os
import torch.nn as nn
import torch
os.environ['CUDA_VISIBLE_DEVICES'] = '0'



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json')
    parser.add_argument('--dataset', default='Carrada', help='dataset for model training,\
                            options: Carrada, CarradaRAC', type=str)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    # parser.add_argument('--dist-url', default='tcp://192.168.8.119:8080', help='url used to set up distributed training')
    parser.add_argument("--sync-bn", dest="sync_bn", help="Use sync batch norm", action='store_true')
    args = parser.parse_args()

    init_distributed_mode(args)
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)


    init = Initializer(cfg)
    data = init.get_data()


    if cfg['model'] == 'mvnet':
        net = MVNet(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'mvanet':
        net = MVANet(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'mva_dcn':
        net = MVA_DCN(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tmva_dcn':
        net = TMVA_DCN(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tmva_tdc':
        net = TMVA_TDC(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcin':
        net = PKCIn(n_classes=data['cfg']['nb_classes'],
                          n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'adapkc':
        net = AdaPKC(n_classes=data['cfg']['nb_classes'],
                     n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'pkcon':
        net = PKCOn(n_classes=data['cfg']['nb_classes'],
                    n_frames=data['cfg']['nb_input_channels'])
    elif cfg['model'] == 'tfoc':
        net = TFocNet(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'],
                      signal_type=data['cfg']['signal_type'],
                      zero_doppler_filter=False)
    elif cfg['model'] == 'vit':
        net = VitNet(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'],
                      signal_type=data['cfg']['signal_type'])
    else:
        net = TMVANet(n_classes=data['cfg']['nb_classes'],
                      n_frames=data['cfg']['nb_input_channels'])


    print('Number of trainable parameters in the model: %s' % str(count_params(net)))

    if args.distributed and args.sync_bn:
        net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(net)

    if args.finetune:
        saved_model = torch.load(args.finetune, map_location=torch.device('cpu'))
        net.load_state_dict(saved_model)
    net = net.cuda()
    torch.manual_seed(cfg['torch_seed'])
    net.apply(_init_weights)

    if args.distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.gpu])

        # dp mode zlw0240705
    if len(cfg['gpu_id']) > 1:
        print('Running in DataParrallel Mode on GPU:', cfg['gpu_id'])
        net = nn.DataParallel(net, cfg['gpu_id'])

    if cfg['model'] == 'mvnet':
        Model(net, data, trackloss=False).train(add_temp=False)
    elif cfg['model'] == 'tfoc':
        Model(net, data, cfg['trackloss']).train(add_temp=True)
    else:
        Model(net, data, trackloss=False).train(add_temp=True)



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

if __name__ == '__main__':
    main()
