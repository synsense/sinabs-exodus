import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, threshold, tauRho):
        threshold = threshold

        spikes, vmem_before_spikes = sinabsslayerCuda.getSpikes(membranePotential.contiguous(), refractoryResponse, threshold, 1.0)
        pdfTimeConstant = tauRho * threshold
        ctx.threshold = threshold
        ctx.pdfTimeConstant = pdfTimeConstant
        ctx.save_for_backward(vmem_before_spikes)

        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        membranePotential, = ctx.saved_tensors
        vmem = (membranePotential - ctx.threshold / 2) % ctx.threshold
        vmem_below = membranePotential * (membranePotential < ctx.threshold)
        vmem_above = vmem * (membranePotential >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = 1 / ctx.pdfTimeConstant * torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.pdfTimeConstant)
        #spikePdf = (membranePotential >= (ctx.threshold - 0.5)).float()

        return gradOutput * spikePdf, None, None, None, None


spikeFunction = SpikeFunction().apply
