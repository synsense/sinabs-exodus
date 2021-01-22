import torch
import slayerCuda

class spikeFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, membranePotential, refractoryResponse, threshold, t_sim, tauRho, scaleRho):
        device = membranePotential.device
        dtype = membranePotential.dtype
        threshold = threshold

        #spikes = slayerCuda.getSpikes(membranePotential.contiguous(), refractoryResponse, threshold, t_sim)
        spikes = slayerCuda.getSpikes(membranePotential.contiguous(), refractoryResponse, threshold, t_sim)

        pdfScale = torch.autograd.Variable(torch.tensor(scaleRho, device=device, dtype=dtype),
                                           requires_grad=False)
        pdfTimeConstant = torch.autograd.Variable(
            torch.tensor(tauRho * threshold, device=device, dtype=dtype),
            requires_grad=False)  # needs to be scaled by theta
        threshold = torch.autograd.Variable(torch.tensor(threshold, device=device, dtype=dtype),
                                            requires_grad=False)
        ctx.save_for_backward(membranePotential, threshold, pdfTimeConstant, pdfScale)

        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        (membranePotential, threshold, pdfTimeConstant, pdfScale) = ctx.saved_tensors
        spikePdf = pdfScale / pdfTimeConstant * torch.exp(-torch.abs(membranePotential - threshold) / pdfTimeConstant)

        # return gradOutput, None, None, None # This seems to work better!
        return gradOutput * spikePdf, None, None, None, None, None
