# sinabs-exodus

This project is loosely based upon the SLAYER-pytorch repository.
As a plugin to the [sinabs](https://sinabs.ai) spiking neural network library it can provide massive speedups in trianing and inference on GPU.


## Getting started

### Prerequisites

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

You should also make sure that you have a [pyTorch](https://pytorch.org/get-started/locally/) installation that is compatible with your CUDA version.
To verify this, open a python console and run
```
import torch
print(torch.__version__)
```
The part after the `+` in the output is the CUDA version that pyTorch has been installed for and should match that of your system.

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

Note that the `tau_syn` parameter for adding synaptic dynamics to `IAF` and `LIF` layers is currently not supported. However, you can include an additional `ExpLeak` layer in your model for the same functionality.
