from setuptools import setup, Command
from torch.utils import cpp_extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import versioneer

import os

if hasattr(cpp_extension, "get_compiler_abi_compatibility_and_version"):
    # - New function since torch 1.12
    if cpp_extension.get_compiler_abi_compatibility_and_version("g++")[0]:
        os.environ["CC"] = "g++"
        os.environ["CXX"] = "g++"
    else:
        os.environ["CC"] = "c++"
        os.environ["CXX"] = "c++"
else:
    # - This works up to torch 1.11
    if cpp_extension.check_compiler_abi_compatibility("g++"):
        os.environ["CC"] = "g++"
        os.environ["CXX"] = "g++"
    else:
        os.environ["CC"] = "c++"
        os.environ["CXX"] = "c++"

# Class for clean command
class Cleaner(Command):
    """Clean command to tidy up after building"""
    user_options = []
    def initialize_options(self):
        pass
    def finalize_options(self):
        pass
    def run(self):
        os.system("rm -vrf ./build ./dist ./*pyc ./*egg-info")

# Update cmdclass
cmdclass = versioneer.get_cmdclass()
cmdclass.update({"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)})
cmdclass.update({"clean": Cleaner})

# Handle versions
version = versioneer.get_version()
version_major = version.split(".")[0]

# Install
setup(
    name='sinabs-exodus',
    version=version,
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
    install_requires=["torch", f"sinabs == {version_major}.*, >= 1.1.1"],
)

