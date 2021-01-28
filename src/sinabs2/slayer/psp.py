import torch
from slayerSNN.slayer import _pspFunction as pspFunction


def generateEpsp(input_spikes, epsp_kernel, t_sim):
    out = []
    if len(epsp_kernel.shape) == 1:
        out.append(pspFunction.apply(input_spikes, epsp_kernel, 1))
    if len(epsp_kernel.shape) == 2:
        for i, k in enumerate(epsp_kernel):
            out.append(pspFunction.apply(input_spikes[i], k, 1))

    return torch.stack(out)
