import torch
import sinabsslayerCuda


def generateEpsp(input_spikes, epsp_kernel, t_sim):
    out = []
    if len(epsp_kernel.shape) == 1:
        out.append(pspFunction.apply(input_spikes, epsp_kernel, 1))
    if len(epsp_kernel.shape) == 2:
        for i, k in enumerate(epsp_kernel):
            out.append(pspFunction.apply(input_spikes[i], k, 1))

    return torch.stack(out)


class pspFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spike, filter, Ts):
        device = spike.device
        dtype = spike.dtype
        psp = sinabsslayerCuda.conv(spike.contiguous(), filter, Ts)
        Ts = torch.autograd.Variable(torch.tensor(Ts, device=device, dtype=dtype), requires_grad=False)
        ctx.save_for_backward(filter, Ts)
        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        (filter, Ts) = ctx.saved_tensors
        gradInput = sinabsslayerCuda.corr(gradOutput.contiguous(), filter, Ts)
        if filter.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = None
            pass

        return gradInput, gradFilter, None

