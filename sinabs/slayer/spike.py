import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, threshold, tauRho):
        threshold = threshold

        spikes = sinabsslayerCuda.getSpikes(
            membranePotential.contiguous(), refractoryResponse, threshold, 1.0
        )
        pdfTimeConstant = tauRho * threshold
        ctx.threshold = threshold
        ctx.pdfTimeConstant = pdfTimeConstant
        ctx.save_for_backward(membranePotential)

        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        membranePotential, = ctx.saved_tensors
        vmem_shifted = membranePotential - ctx.threshold / 2
        vmem_periodic = vmem_shifted % ctx.threshold
        vmem_below = vmem_shifted * (membranePotential < ctx.threshold)
        vmem_above = vmem_periodic * (membranePotential >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.pdfTimeConstant) / ctx.threshold
        # spikePdf = (membranePotential >= (ctx.threshold - 0.5)).float()

        return gradOutput * spikePdf, None, None, None, None


spikeFunction = SpikeFunction().apply
