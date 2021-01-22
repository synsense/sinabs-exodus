import torch
from slayerSNN.slayer import _pspFunction as pspFunction


def generateEpsp(input_spikes, epsp_kernel, t_sim):
    out = []
    if len(epsp_kernel.shape) == 1:
        out.append(pspFunction.apply(input_spikes, epsp_kernel, t_sim))
    if len(epsp_kernel.shape) == 2:
        for k in epsp_kernel:
            out.append(pspFunction.apply(input_spikes, k, t_sim))

    return torch.stack(out)
