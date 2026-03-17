from setuptools import setup
from setuptools import find_packages

import sys
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from distutils.core import Distribution
from distutils.command.install import install
from setuptools.command.develop import develop

class PeCoRSSBaseCommand:
    user_options = [
        ('with-cuda', None, 'Include CUDA extension')
    ]

    def initialize_options(self):
        super().initialize_options()
        self.with_cuda = None

    def finalize_options(self):
        super().finalize_options()
        if self.with_cuda:
            self.distribution.with_cuda = True

class PeCoRSSInstallCommand(PeCoRSSBaseCommand, install):
    user_options = install.user_options + PeCoRSSBaseCommand.user_options

class PeCoRSSDevelopCommand(PeCoRSSBaseCommand, develop):
    user_options = develop.user_options + PeCoRSSBaseCommand.user_options
            
class PeCoRSSBuildExtCommand(BuildExtension):
    def run(self):
        if not hasattr(self.distribution, 'with_cuda') or not self.distribution.with_cuda:
            print("Skipping CUDA extension installation.")
            return
        super().run()
                  
def get_extensions(with_cuda):
    extensions = []
    if with_cuda:
        extensions.append(
            CUDAExtension(
                name="prf_cuda",
                sources=["svrss/extension/prf_cuda_ops.cpp", "svrss/extension/prf_cuda.cu"],
            )
        )
    return extensions
      
def has_cuda_option():
    return '--with-cuda' in sys.argv

setup(name='pecorss',
      version='1.3',
      author=['Liwen Zhang',
              'Youcheng Zhang',
              'Xinyan Zhang',
              'Teng Li',
              'Jinlong Li'],
      author_email='lwzhang9161@126.com',
      maintainer=['Liwen Zhang',
                  'Youcheng Zhang',
                  'Xinyan Zhang',
                  'Teng Li',
                  'Jinlong Li'],
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
            'install': PeCoRSSInstallCommand,
            'build_ext': PeCoRSSBuildExtCommand,
            'develop': PeCoRSSDevelopCommand
      })