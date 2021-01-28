import torch
import slayerCuda


class SpikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, threshold, tauRho, scaleRho):
        print(membranePotential.shape)
        threshold = threshold

        spikes = slayerCuda.getSpikes(membranePotential.contiguous(), refractoryResponse, threshold, 1.0)
        pdfScale = scaleRho
        pdfTimeConstant = tauRho * threshold
        ctx.threshold = threshold
        ctx.pdfTimeConstant = pdfTimeConstant
        ctx.pdfScale = pdfScale
        ctx.save_for_backward(membranePotential)

        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        membranePotential, = ctx.saved_tensors
        spikePdf = ctx.pdfScale / ctx.pdfTimeConstant * torch.exp(-torch.abs(membranePotential - ctx.threshold) / ctx.pdfTimeConstant)

        # return gradOutput, None, None, None # This seems to work better!
        return gradOutput * spikePdf, None, None, None, None


spikeFunction = SpikeFunction().apply
