"""Script to set the path to CARRADA in the config.ini file"""
import argparse
from mvrss.utils.configurable import Configurable
from mvrss.utils import MVRSS_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--carrada', default='/media/liwenzhang/Data/DataSets/CarradaRAC/Carrada',
                        help='Path to the Carrada dataset.')
    parser.add_argument('--logs', default='/home/liwenzhang/PeCoLab/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = MVRSS_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'carrada', args.carrada)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
