import torch
import sinabsslayerCuda


def generateEpsp(input_spikes, epsp_kernel):
    out = []
    if len(epsp_kernel.shape) == 1:
        out.append(pspFunction(input_spikes, epsp_kernel))

    if len(epsp_kernel.shape) == 2:
        for i, k in enumerate(epsp_kernel):
            out.append(pspFunction(input_spikes[i], k))

    return torch.stack(out)


class PspFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike, filter):
        device = spike.device
        dtype = spike.dtype
        psp = sinabsslayerCuda.conv(spike.contiguous(), filter, 1)
        ctx.save_for_backward(filter)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        (filter,) = ctx.saved_tensors
        gradInput = sinabsslayerCuda.corr(gradOutput.contiguous(), filter, 1)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass

        return gradInput, gradFilter


pspFunction = PspFunction().apply
