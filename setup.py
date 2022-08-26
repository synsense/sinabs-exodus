from setuptools import setup
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import versioneer

import os

if cpp_extension.check_compiler_abi_compatibility("g++"):
    os.environ["CC"] = "g++"
    os.environ["CXX"] = "g++"
else:
    os.environ["CC"] = "c++"
    os.environ["CXX"] = "c++"

cmdclass = versioneer.get_cmdclass()
cmdclass.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})

setup(
    name='exodus',
    version=versioneer.get_version(),
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
    cmdclass=cmdclass,
    install_requires=["sinabs"],
)

