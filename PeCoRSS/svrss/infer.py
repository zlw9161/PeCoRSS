"""Main script to infer a pretrained model"""
import argparse
import json
import torch
from torch.utils.data import DataLoader

from svrss.utils.paths import Paths
from svrss.utils.functions import count_params
from svrss.learners.tester import Tester
from svrss.loaders.dataset import Carrada, PDRdata, CWRdata, RDSyn, XPDdata
from svrss.models import CWRNet, PKCIn4CWR, PKCIn4CWR_Dns, TARSS4CWR_V1, TARSS4CWR_V2
from svrss.models import KuRADNet, KuRADNet_WSkip, PKCIn4PDR, PKCIn4PDR_Dns, TARSS4PDR_V1, TARSS4PDR_V2
from svrss.models import BaseLine3D, TARSSNet_V1_RD, TARSSNet_V1_RA, TARSSNet_V2_RD, TARSSNet_V2_RA
from svrss.models import UNet, FCN8s, DeepLabV3Plus_Res101, PKCIn4ANY
from svrss.loaders.dataloaders import SequenceDataset
from svrss.utils.distributed_utils import init_distributed_mode


def test_model():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', help='Path to config file.', default='config.json', type=str, required=True)
    parser.add_argument('--dataset', default='Carrada_RD', help='dataset for model training,\
                        options: Carrada_RD, Carrada_RA, CarradaRAC_RD, CarradaRAC_RA, CWRdata, PDRdata, RDSyn', type=str)
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--sync-bn', dest='sync_bn', help='Use sync batch norm', action='store_true')
    args = parser.parse_args()

    init_distributed_mode(args)
    cfg_path = args.cfg
    with open(cfg_path, 'r') as fp:
        cfg = json.load(fp)
    device = torch.device(cfg['device'])
    cfg['distributed'] = args.distributed

    paths = Paths().get()
    # print(cfg)
    exp_name = cfg['name_exp'] + '_' + str(cfg['version'])
    path = paths['logs'] / cfg['dataset'] / cfg['model'] / exp_name
    model_path = path / 'results' / (cfg['model']+'.pt')
    test_results_path = path / 'results' / 'test_results.json'
    
    if cfg['dataset'] in ['CWRdata', 'Carrada_RA', 'Carrada_RD', 'CarradaRAC_RD', 'CarradaRAC_RA']:
        cfg['nb_classes'] = 4
    elif cfg['dataset'] == 'PDRdata':
        cfg['nb_classes'] = 5
    elif cfg['dataset'] == 'RDSyn':
        cfg['nb_classes'] = 2
    else:
        raise KeyError('Dataset {} has not been supported yet.'.format(cfg['dataset']))

    if cfg['model'] == 'cwrnet':
        # loading parallel saved model in local single gpu mode
        model = CWRNet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcin4cwr':
        model = PKCIn4CWR(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcin4cwr_dns':
        model = PKCIn4CWR_Dns(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcin4pdr':
        model = PKCIn4PDR(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcin4pdr_dns':
        model = PKCIn4PDR_Dns(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarss4cwr_v1':
        model = TARSS4CWR_V1(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarss4cwr_v2':
        model = TARSS4CWR_V2(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarss4pdr_v1':
        model = TARSS4PDR_V1(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarss4pdr_v2':
        model = TARSS4PDR_V2(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'kuradnet':
        model = KuRADNet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'kuradnet_wskip':
        model = KuRADNet_WSkip(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'baseline3d':
        if cfg['dataset'] in ['CarradaRAC_RA', 'Carrada_RA']:
            model = BaseLine3D(n_classes=cfg['nb_classes'],
                               n_frames=cfg['nb_input_channels'],
                               signal_type='range_angle')
        else:
            model = BaseLine3D(n_classes=cfg['nb_classes'],
                               n_frames=cfg['nb_input_channels'],
                               signal_type='range_doppler')
    elif cfg['model'] == 'tarssnet_v1_ra':
        model = TARSSNet_V1_RA(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarssnet_v1_rd':
        model = TARSSNet_V1_RD(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarssnet_v2_ra':
        model = TARSSNet_V2_RA(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'tarssnet_v2_rd':
        model = TARSSNet_V2_RD(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'unet':
        model = UNet(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'fcn8s':
        model = FCN8s(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'deeplabv3plus':
        model = DeepLabV3Plus_Res101(n_classes=cfg['nb_classes'], n_frames=cfg['nb_input_channels'])
    elif cfg['model'] == 'pkcin4any':
        model = PKCIn4ANY(n_classes=cfg['nb_classes'],
                          n_frames=cfg['nb_input_channels'],
                          w=cfg['w_size'],
                          h=cfg['h_size'])
    else:
        raise ValueError('model {} is not supported in infer.py yet.'.format(cfg['model']))
    
    print('Number of trainable parameters in the model: %s' % str(count_params(model)))
    print('Model Name: %s' % cfg['model'])
    print('DataSet Name: %s' % cfg['dataset'])
    saved_model = torch.load(model_path, map_location=torch.device('cpu'))
    if args.distributed and args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    first_key = next(iter(saved_model))
    from collections import OrderedDict
    if first_key.startswith('module.'):
        new_state_dict = OrderedDict()
        for k, v in saved_model.items():
            name = k[7:] if k.startswith('module.') else k  # 去除 'module.' 前缀
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(saved_model)
    
    model.to(device)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

    tester = Tester(cfg)
    print(tester)

    if cfg['dataset'] in ['Carrada_RA', 'Carrada_RD', 'CarradaRAC_RA', 'CarradaRAC_RD']:
        data = Carrada()
    elif cfg['dataset'] == 'CWRdata':
        data = CWRdata()
    elif cfg['dataset'] == 'PDRdata':
        data = PDRdata()
    elif cfg['dataset'] == 'RDSyn':
        data = RDSyn()
    elif cfg['dataset'] == 'XPDdata':
        data = XPDdata()
    test = data.get('Test')
    testset = SequenceDataset(test)
    seq_testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=0)
    tester.set_annot_type(cfg['annot_type'])
    if cfg['model'] in ['fcn8s', 'unet', 'deeplabv3plus', 'kuradnet', 'kuradnet_wskip']:
        test_metrics = tester.predict(model, seq_testloader, get_quali=True, add_temp=False)
    else:
        # print(cfg['model'])
        test_metrics = tester.predict(model, seq_testloader, get_quali=True, add_temp=True)
    # if get_rank() == 0:

    if cfg['dataset'] in ['Carrada_RA', 'CarradaRAC_RA']:
        signal_type = 'range_angle'
    else:
        signal_type = 'range_doppler'

    print('Test Prec: '
          '({}) = {}'.format(signal_type, test_metrics[signal_type]['prec']))
    # zlw@20220227
    print('Test mIoU: '
        '({}) = {}'.format(signal_type, test_metrics[signal_type]['miou']))
    print('Test mIoU by class: '
        '({}) = {}'.format(signal_type, test_metrics[signal_type]['miou_by_class'][0:2]))
    print(test_metrics[signal_type]['miou_by_class'][2:])
    print('Test Dice: '
        '({}) = {}'.format(signal_type, test_metrics[signal_type]['dice']))
    print('Test Dice by class: '
        '({}) = {}'.format(signal_type, test_metrics[signal_type]['dice_by_class'][0:2]))
    print(test_metrics[signal_type]['miou_by_class'][2:])
    print('Confusion matrix: ({}) = '.format(signal_type))
    for i in range(cfg['nb_classes']):
        print('{}'.format(test_metrics[signal_type]['confusion_matrix'][i]))

if __name__ == '__main__':
    test_model()
