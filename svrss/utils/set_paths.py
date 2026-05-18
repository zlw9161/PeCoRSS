"""Script to set the path to CARRADA in the config.ini file"""
import argparse
from svrss.utils.configurable import Configurable
from svrss.utils import SVRSS_HOME

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Settings paths for training and testing.')
    parser.add_argument('--carrada', default='/media/ycc/4TB/DataSets',
                        help='Path to the CARRADA dataset.')
    parser.add_argument('--cwr', default='/media/ycc/4TB/DataSets/KuRALS/KuCW',
                        help='Path to the KuRALS_CW dataset.')
    parser.add_argument('--pdr', default='/media/ycc/4TB/DataSets/KuRALS/KuPD',
                        help='Path to the KuRALS_PD dataset.')
    parser.add_argument('--logs', default='/root/workspace/logs',
                        help='Path to the save the logs and models.')
    args = parser.parse_args()
    config_path = SVRSS_HOME / 'config_files' / 'config.ini'
    configurable = Configurable(config_path)
    configurable.set('data', 'warehouse', args.carrada)
    configurable.set('data', 'logs', args.logs)
    with open(config_path, 'w') as fp:
        configurable.config.write(fp)
