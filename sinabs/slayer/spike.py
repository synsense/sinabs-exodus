import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, membr_pot, refr_response, threshold, tau_rho, scale_rho=1.0):
        """
        Generate spikes and apply refractory response to membrane potential.
        Will modifie membrane potential in-place.

        Parameters
        ----------
        membr_pot: torch.Tensor
            The membrane potential. Expected shape: (N, T_sim), where N is
            *anything* that can be computed in parallel, i.e. batches, neurons...
            Has to be contiguous.
        refr_response: torch.Tensor
            Refractory response. Has to be 1-dimensional
        threshold: float
            Firing threshold
        tau_rho: float
            Width of the surrogate gradient exponential.
        scale_rho: float
            Scales the surrogate gradients.

        Returns
        -------
        torch.tensor
            Integer spike raster. Same shape as ``membr_pot``
        """

        if not membr_pot.is_contiguous():
            raise ValueError("'membr_pot' has to be contiguous.")
        if not membr_pot.ndim == 2:
            raise ValueError("'membr_pot' must be 2D, (N, Time)")
        if not refr_response.ndim == 1:
            raise ValueError("'refr_response' has to be 1D.")

        spikes = sinabsslayerCuda.getSpikes(membr_pot, refr_response, threshold, 1.0)

        # Prepare backward
        pdf_time_const = tau_rho * threshold
        ctx.threshold = threshold
        ctx.pdf_time_const = pdf_time_const
        ctx.scale_rho = scale_rho
        ctx.save_for_backward(membr_pot)

        return spikes

    @staticmethod
    def backward(ctx, gradOutput):
        membr_pot, = ctx.saved_tensors
        vmem_shifted = membr_pot - ctx.threshold / 2
        vmem_periodic = vmem_shifted % ctx.threshold
        vmem_below = vmem_shifted * (membr_pot < ctx.threshold)
        vmem_above = vmem_periodic * (membr_pot >= ctx.threshold)
        vmem_new = vmem_above + vmem_below
        spikePdf = (
            torch.exp(-torch.abs(vmem_new - ctx.threshold / 2) / ctx.pdf_time_const)
            / ctx.threshold
        )

        return ctx.scale_rho * gradOutput * spikePdf, None, None, None, None


class SpikeFunctionLB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, membranePotential, refractoryResponse, threshold, threshold_low, tauRho
    ):
        threshold = threshold

        spikes = sinabsslayerCuda.getSpikesLB(
            membranePotential.contiguous(),
            refractoryResponse,
            threshold,
            threshold_low,
            1.0,
        )
        pdfTimeConstant = tauRho * threshold
        ctx.threshold = threshold
        ctx.pdfTimeConstant = pdfTimeConstant
        ctx.save_for_backward(membranePotential)

        return spikes


spikeFunction = SpikeFunction().apply
