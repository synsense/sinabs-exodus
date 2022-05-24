from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import os

with open("sinabs/exodus/version.py") as version_info:
    exec(version_info.read())

if cpp_extension.check_compiler_abi_compatibility("g++"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
else:
    os.environ["CC"] = "c++"
    os.environ["CXX"] = "c++"

setup(
    name='exodus',
    packages=['sinabs.exodus', 'sinabs.exodus.layers'],
    ext_modules=[
        CUDAExtension(
            name='exodus_cuda',
            sources=[
                'cuda/bindings.cu',
                # 'cuda/leaky_bindings.cu',
                # 'cuda/experimental_bindings.cu',
            ],
            depends=[
                'cuda/lif_kernels.h'
                'cuda/leaky_kernels.h'
                # 'cuda/experimental_kernels.h'
            ],
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    install_requires=["sinabs"],
    version=__version__,
)

