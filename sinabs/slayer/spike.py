from typing import Optional

import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        membr_pot: torch.tensor,
        membrane_subtract: float,
        threshold: float,
        window: Optional[float] = None,
        scale_rho: float = 1.0,
    ):
        """
        Generate spikes and apply refractory response to membrane potential.
        Will modifie membrane potential in-place. For non-leaky IAF neuron models,
        SpikeFunctionIterForward is the faster option.

        Parameters
        ----------
        membr_pot : torch.Tensor
            The membrane potential. Expected shape: (N, T_sim), where N is
            the product of all dimensions that can be computed in parallel,
            i.e. batches, neurons...
            Has to be contiguous.
        membrane_subtract : torch.Tensor
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

        spikes = sinabsslayerCuda.getSpikes(membr_pot, membrane_subtract, threshold)

        # Prepare backward
        ctx.threshold = threshold
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        ctx.membrane_subtract = membrane_subtract
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
        membrane_subtract: float,
        threshold: float,
        threshold_low: float,
        window: Optional[float] = None,
        scale_rho=1.0,
    ):
        """
        Generate spikes and apply refractory response to membrane potential, considering
        a non-optional lower limit for the membrane potential. Will modifie membrane
        potential in-place. For non-leaky IAF neuron models, SpikeFunctionIterForward is
        the faster option.

        Parameters
        ----------
        membr_pot: torch.Tensor
            The membrane potential. Expected shape: (N, T_sim), where N is
            *anything* that can be computed in parallel, i.e. batches, neurons...
            Has to be contiguous.
        membrane_subtract: float
            Value that is subracted from membrane potential after spike
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
        if threshold <= threshold_low:
            raise ValueError("`threshold` must be greater than `threshold_low`.")

        spikes = sinabsslayerCuda.getSpikesLB(
            membr_pot, membrane_subtract, threshold, threshold_low
        )
        ctx.threshold = threshold
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        ctx.membrane_subtract = membrane_subtract
        ctx.save_for_backward(membr_pot)

        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        membr_pot, = ctx.saved_tensors

        # Heaviside surrogate gradients
        surrogates = (membr_pot >= (ctx.threshold - ctx.window)).float() / ctx.threshold

        # Gradient wrt. input
        grad_input = sinabsslayerCuda.spikeGradsLB(
            surrogates.contiguous(), grad_output.contiguous(), ctx.membrane_subtract
        )

        return ctx.scale_rho * grad_input, None, None, None, None, None


class SpikeFunctionIterForward(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.tensor,
        membrane_subtract: float,
        state: torch.tensor,
        activations: torch.tensor,
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
        membrane_subtract: float
            Value that is subracted from membrane potential after spike
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
        torch.tensor (T x T_sim)
            Membrane potential for each neuron and time step
        torch.tensor (N x T_sim)
            Integer spike raster. Same shape as ``membr_pot``
        """

        if not inp.ndim == 2:
            raise ValueError("'inp' must be 2D, (N, Time)")
        if threshold_low is not None and threshold <= threshold_low:
            raise ValueError("`threshold` must be greater than `threshold_low`.")

        time_steps = inp.shape[1]

        states = torch.zeros_like(inp)
        spikes = []

        for t in range(time_steps):

            # subtract a number of membrane_subtract's as there are spikes
            state = inp[:, t] + state - activations * membrane_subtract
            if threshold_low is not None:
                # ReLU for efficient implementation of lower limit
                state = torch.nn.functional.relu(state - threshold_low) + threshold_low
            states[:, t] = state

            # generate spikes
            activations = (state > 0) * torch.div(
                state, threshold, rounding_mode="floor"
            )

            spikes.append(activations)

        output_spikes = torch.stack(spikes).transpose(0, 1)

        ctx.threshold = threshold
        ctx.threshold_low = threshold_low
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        ctx.membrane_subtract = membrane_subtract
        ctx.save_for_backward(states)

        return output_spikes, states

    @staticmethod
    def backward(ctx, grad_output, grad_state):
        states, = ctx.saved_tensors

        # Heaviside surrogate gradients
        surrogates = (states >= (ctx.threshold - ctx.window)).float() / ctx.threshold

        # Gradient becomes 0 where states is clipped to lower threshold
        if ctx.threshold_low is None:
            not_clipped = torch.ones_like(surrogates)
        else:
            not_clipped = (states > ctx.threshold_low).float()

        # Gradient wrt. intermediate states
        grad_input = sinabsslayerCuda.spikeGradsFull(
            surrogates.contiguous(),
            grad_output.contiguous(),
            not_clipped.contiguous(),
            ctx.membrane_subtract,
        )

        return ctx.scale_rho * grad_input, None, None, None, None, None, None, None


spikeFunction = SpikeFunction().apply
spikeFunctionLB = SpikeFunctionLB().apply
spikeFunctionIterForward = SpikeFunctionIterForward().apply
