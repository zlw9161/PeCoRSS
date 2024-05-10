from setuptools import setup
from setuptools import find_packages


setup(name='pecorss',
      version='1.0',
      author=['Liwen Zhang',
              'Youcheng Zhang',
              'Xinyan Zhang',
              'Teng Li'],
      author_email='lwzhang9161@126.com',
      maintainer=['Liwen Zhang',
                  'Youcheng Zhang',
                  'Xinyan Zhang',
                  'Teng Li'],
      maintainer_email='lwzhang9161@126.com',
      url='https://github.com/zlw9161',
      license='Apache 2.0',
      description='pecorss a.k.a. PeCoLab Radar Semantic Segmentation,\
            which is A unified awsome RSS libarary developed by PeCoLab of ISTA@CASIC.',
      long_description='This version is developed from 2024.05.07,\
            and the detailed description is coming soon.',
      packages=find_packages())
