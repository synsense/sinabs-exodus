from typing import Optional

import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        membr_pot: torch.tensor,
        refr_response: torch.tensor,
        threshold: float,
        window: Optional[float] = None,
        scale_rho: float = 1.0,
    ):
        """
        Generate spikes and apply refractory response to membrane potential.
        Will modifie membrane potential in-place.

        Parameters
        ----------
        membr_pot : torch.Tensor
            The membrane potential. Expected shape: (N, T_sim), where N is
            the product of all dimensions that can be computed in parallel,
            i.e. batches, neurons...
            Has to be contiguous.
        refr_response : torch.Tensor
            Refractory response. Has to be 1-dimensional
        threshold : float
            Firing threshold
        window : Optional[float]
            Surrogate gradient will be Heaviside(membr_pot - (threshold - window))
            If None, will be set to `threshold`.
        scale_rho : float
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
        ctx.threshold = threshold
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        ctx.membrane_subtract = refr_response[0].item()
        ctx.save_for_backward(membr_pot)

        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        membr_pot, = ctx.saved_tensors

        # Heaviside surrogate gradients
        surrogates = (membr_pot >= (ctx.threshold - ctx.window)).float() / ctx.threshold

        # Gradient wrt. input
        grad_input = sinabsslayerCuda.spikeGrads(
            surrogates.contiguous(), grad_output.contiguous(), ctx.membrane_subtract
        )

        return ctx.scale_rho * grad_input, None, None, None, None


class SpikeFunctionLB(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        membr_pot: torch.tensor,
        refr_response: torch.tensor,
        threshold: float,
        threshold_low: float,
        window: Optional[float] = None,
        scale_rho=1.0,
    ):
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
        threshold_low: float
            Lower limit for membr_pot
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
        if threshold <= threshold_low:
            raise ValueError("`threshold` must be greater than `threshold_low`.")

        spikes = sinabsslayerCuda.getSpikesLB(
            membr_pot, refr_response, threshold, threshold_low, 1.0
        )
        ctx.threshold = threshold
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        ctx.membrane_subtract = refr_response[0].item()
        ctx.save_for_backward(membr_pot)

        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        membr_pot, = ctx.saved_tensors

        # Heaviside surrogate gradients
        surrogates = (membr_pot >= (ctx.threshold - ctx.window)).float() / ctx.threshold

        # Gradient wrt. input
        grad_input = sinabsslayerCuda.spikeGrads(
            surrogates, grad_output, ctx.membrane_subtract
        )

        return ctx.scale_rho * grad_input, None, None, None, None, None


spikeFunction = SpikeFunction().apply
spikeFunctionLB = SpikeFunctionLB().apply
