"""Script to set the path in the config.ini file"""
import argparse
from svrss.utils.configurable import Configurable
from svrss.utils import SVRSS_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--cwr', default='/data/PeCoLab/DataSets/CWRdata',
                        help='Path to the continuous wave radar dataset.')
    parser.add_argument('--pdr', default='/data/PeCoLab/DataSets/KuRALS/KuPD',
                        help='Path to the pulse doppler radar dataset.')
    parser.add_argument('--carrada', default='/data/PeCoLab/DataSets/Carrada',
                        help='Path to the Carrada dataset.')
    parser.add_argument('--rdsyn', default='/data/PeCoLab/DataSets/RDSyn',
                        help='Path to the synthesis radar dataset.')
    parser.add_argument('--xpd', default='/home/liwenzhang/PeCoLab/DataSets/XPDData/S70-20240717-npy',
                        help='Path to the synthesis radar dataset.')
    parser.add_argument('--logs', default='/data/PeCoLab/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = SVRSS_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'cwr', args.cwr)
    configurable.set('data', 'pdr', args.pdr)
    configurable.set('data', 'carrada', args.carrada)
    configurable.set('data', 'rdsyn', args.rdsyn)
    configurable.set('data', 'xpd', args.xpd)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
