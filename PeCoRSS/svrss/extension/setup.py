from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="prf_cuda",
    ext_modules=[
        CUDAExtension(
            name="prf_cuda",
            sources=["prf_cuda_ops.cpp", "prf_cuda.cu"],
        )
    ],
    cmdclass={
        "build_ext": BuildExtension
    }
)