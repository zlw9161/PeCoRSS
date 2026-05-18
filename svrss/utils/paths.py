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
        data = self.config['data']
        warehouse = Path(data['warehouse'])
        self.paths['warehouse'] = warehouse
        self.paths['logs'] = Path(data['logs'])
        self.paths['carrada'] = warehouse / 'Carrada'
        if data.get('cwr'):
            self.paths['CWRdata'] = Path(data['cwr'])
        if data.get('pdr'):
            self.paths['PDRdata'] = Path(data['pdr'])
        if data.get('xpddata'):
            self.paths['XPDdata'] = Path(data['xpddata'])
        if data.get('rdsyn'):
            self.paths['RDSyn'] = Path(data['rdsyn'])
        if data.get('nuaasim'):
            self.paths['NuaaSim'] = Path(data['nuaasim'])

    def get(self):
        return self.paths
