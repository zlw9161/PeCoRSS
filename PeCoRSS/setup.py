from setuptools import setup
from setuptools import find_packages

import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.core import Distribution
from distutils.command.install import install

class PeCoRSSDistribution(Distribution):
      def __init__(self, attrs=None):
            self.with_cuda = None
            super().__init__(attrs=attrs)
            
class PeCoRSSInstallCommand(install):
      user_options = BuildExtension.user_options + [
        ('with-cuda', None, 'Include CUDA extension')
      ]
      
      def initialize_options(self):
            install.initialize_options(self)
            self.with_cuda = None
      
      def finalize_options(self):
            install.finalize_options(self)
            if self.with_cuda:
                  self.distribution.with_cuda = True
                  
def get_extensions():
      extenstions = []
      if 'with_cuda' in Distribution.get_option_dict('install').get('with_cuda', (None, None))[1]:
            extenstions.append(
                  CUDAExtension(
                        name="prf_cuda",
                        source=["extension/prf_cuda_ops.cpp", "extension/prf_cuda.cu"],
                  )
            )
            return extenstions
      
def has_cuda_option():
      return '--with-cuda' in sys.argv

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
      packages=find_packages(),
      ext_modules=get_extensions(has_cuda_option()),
      cmdclass={
            'build_ext': BuildExtension,
            'install': PeCoRSSInstallCommand
      },
      distclass=PeCoRSSDistribution)
