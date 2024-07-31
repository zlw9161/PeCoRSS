"""Class to get global paths"""
from pathlib import Path
from svrss.utils import SVRSS_HOME
from svrss.utils.configurable import Configurable


class Paths(Configurable):

    def __init__(self):
        self.config_path = SVRSS_HOME / 'config_files' / 'config.ini'
        super().__init__(self.config_path)
        self.paths = dict()
        self._build()

    def _build(self):
        # warehouse = Path(self.config['data']['warehouse'])
        # self.paths['warehouse'] = warehouse
        self.paths['logs'] = Path(self.config['data']['logs'])
        self.paths['CWRdata'] = Path(self.config['data']['cwr'])
        self.paths['PDRdata'] = Path(self.config['data']['pdr'])
        self.paths['Carrada'] = Path(self.config['data']['carrada'])
        self.paths['RDSyn'] = Path(self.config['data']['rdsyn'])
        self.paths['XPDdata'] = Path(self.config['data']['xpd'])

    def get(self):
        return self.paths
