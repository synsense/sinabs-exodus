from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os

if cpp_extension.check_compiler_abi_compatibility("g++"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
else:
    os.environ["CC"] = "c++"
    os.environ["CXX"] = "c++"

setup(
    name='sinabs-slayer',
    packages=['sinabs.slayer', 'sinabs.slayer.layers'],
    ext_modules=[
        CUDAExtension(
            name='sinabsslayerCuda',
            sources=[
                'csrc/cuda/slayerKernels.cu'
            ],
            depends=[
                'csrc/cuda/spikeKernels.h',
                'csrc/cuda/convKernels.h',
                'csrc/cuda/shiftKernels.h'
            ],
            extra_compile_args={
                'cxx': ['-g'],
                'nvcc': ['-arch=sm_60', '-O3', '-use_fast_math']
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=["sinabs"]
)

