# sinabs-exodus

Sinabs-exodus is a plugin to the [sinabs](https://sinabs.ai) spiking neural network library. It can provide massive speedups in training and inference on GPU.

The tool is based on [EXODUS](https://arxiv.org/abs/2205.10242), a formulation of backpropagation-through-time with surrogate gradients, that allows for efficient parallelization. EXODUS stands for _**EX**act calculation **O**f **D**erivatives as **U**pdate to **S**LAYER_. It builds upon the SLAYER[^1] algorithm, but uses mathematically accurate gradients and tends to be more robust to surrogate gradient scaling, making training less prone to suffer from exploding or vanishin gradients.

Some of the code in this library is loosely based upon [slayerPytorch](https://github.com/bamsumit/slayerPytorch), the python implementation of SLAYER.

If you use any of this code please cite the following publication:
```
@article{bauer2022exodus,
  title={EXODUS: Stable and Efficient Training of Spiking Neural Networks},
  author={Bauer, Felix Christian and Lenz, Gregor and Haghighatshoar, Saeid and Sheik, Sadique},
  journal={arXiv preprint arXiv:2205.10242},
  year={2022}
}
```

## Getting started

### Prerequisites
<a name="prerequisites"></a>
EXODUS uses CUDA for efficient computation, so you will need a CUDA-capable GPU, and a working installation of [CUDA](https://docs.nvidia.com/cuda/index.html).

If you have CUDA installed, you can use the command
```
$ nvcc -V
```
to see the installed version. The last line should say something like `Build cuda_xx.x.....`, where x.xx is the version.
Note that
```
$ nvidia-smi
```
does **not** show you the installed CUDA version, but only the newest version your Nvidia driver is compatible with.

You should also make sure that you have a [PyTorch](https://pytorch.org/get-started/locally/) installation that is compatible with your CUDA version.
To verify this, open a python console and run
```
import torch
print(torch.__version__)
```
The part after the `+` in the output is the CUDA version that PyTorch has been installed for and should match that of your system.

### Installation

After cloning this repository, the package can simply be installed via pip.
This is a `namespace package` meaning that once installed this will be sharing its namespace with `sinabs` package.

```
$ pip install . 
```

Do not install in editable (`-e`) mode.


## Usage

If you have used sinabs before, using EXODUS is straightforward, as the APIs are the same.
You just need to import the spiking or leaky layer classes that you want to speed up from `sinabs.exodus.layers` instead of `sinabs.layers`.

Supported classes are:
- `IAF`
- `LIF`
- `ExpLeak`

For example, instead of
```
from sinabs.layers import IAF

iaf = IAF()
```

do 
```
from sinabs.exodus.layers import IAF

iaf = IAF()
```

## Frequent Issues

### CUDA is not installed or version does not match that of torch

If during installation you get an error, such as
```
RuntimeError:
The detected CUDA version (...) mismatches the version that was used to compile
PyTorch (...). Please make sure to use the same CUDA versions.
```
or
```
OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
```
CUDA is either not installed properly on your system or the version does not match that of torch (see [above](#prerequisites)).
If you do have the correct version installed and the error still comes up, try to make sure that the environment variables such as `PATH` and `LD_LIBRARY_PATH` contain references to the correct directories. Please refer to NVIDIA's installation instructions for more details on how to do this for your system.


## License

Sinabs-exodus is published under AGPL v3.0. See the LICENSE file for details.

## Footnotes
[^1]: Sumit Bam Shrestha and Garrick Orchard. "SLAYER: Spike Layer Error Reassignment in Time." 
In _Advances in Neural Information Processing Systems_, pp. 1417-1426. 2018.

