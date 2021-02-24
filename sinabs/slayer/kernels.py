import torch
from typing import Optional, List

FACTOR = 5


def exp_kernel(tau: float, dt: float = 1.0, size: Optional[int] = None):
    """
    Returns an exponential kernel

    Args:
        tau: Time constant in ms
        dt: Time constant in ms
        size: Time constant in ms

    Returns:
        kernel: torch.Tensor

    """
    if size is None:
        size = int(FACTOR * tau / dt)

    t = torch.arange(0, size * dt, dt)
    kernel = torch.exp(-t / tau)
    return kernel


def psp_kernel(
    tau_syn: float, tau_mem: float, dt: float = 1.0, size: Optional[int] = None
):
    if size is None:
        size = int(FACTOR * max(tau_syn, tau_mem) / dt)

    kernel_syn = exp_kernel(tau_syn, dt, size)
    kernel_mem = exp_kernel(tau_mem, dt, size)

    padding = max(len(kernel_mem), len(kernel_syn))
    from torch.nn.functional import conv1d

    psp = conv1d(
        kernel_syn.reshape(1, 1, -1),
        kernel_mem.flip(0).reshape(1, 1, -1),
        padding=padding,
    )
    return psp.squeeze()[:size]


def psp_kernels(
    tau_syn: List, tau_mem: float, dt: float = 1.0, size: Optional[int] = None
):
    size = int(FACTOR * max(*tau_syn, tau_mem) / dt)
    kernel_syn = []
    for tau_s in tau_syn:
        kernel_s = psp_kernel(tau_s, tau_mem=tau_mem, dt=dt, size=size)
        kernel_syn.append(kernel_s)

    kernel_syn = torch.stack(kernel_syn)
    return kernel_syn


def heaviside_kernel(size: int, scale: float = 1.0):
    return torch.ones(size) * scale
