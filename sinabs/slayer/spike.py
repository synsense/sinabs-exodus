import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        membr_pot: torch.tensor,
        membrane_subtract: float,
        threshold: float,
        window: float = 1.0,
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
        (membr_pot,) = ctx.saved_tensors

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
        window: float = 1.0,
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
        (membr_pot,) = ctx.saved_tensors

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
        alpha: float,
        state: torch.tensor,
        activations: torch.tensor,
        threshold: float,
        threshold_low: float,
        window: float = 1.0,
        scale_rho=1.0,
    ):
        """
        Generate spikes and apply refractory response to membrane potential.
        Will modifie membrane potential in-place.

        Parameters
        ----------
        inp: torch.Tensor
            Input to the layer. Expected shape: (N, T_sim), where N is
            *anything* that can be computed in parallel, i.e. batches, neurons...
            Has to be contiguous.
        membrane_subtract: float
            Value that is subracted from membrane potential after spike
        alpha : float
            State decay factor (exp(-dt/tau)). Set 1 for IAF neurons.
        state : torch.Tensor
            1D shape (N,).  Initial states. Has to be contiguous.
        activations : torch.tensor
            1D, shape (N,). Activations from previous time step.
            Has to be contiguous.
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
        if not inp.is_contiguous():
            raise ValueError("'inp' has to be contiguous.")
        if threshold_low is not None and threshold <= threshold_low:
            raise ValueError("`threshold` must be greater than `threshold_low`.")
        if not 0 <= alpha <= 1:
            raise ValueError("'alpha' must be between 0 and 1.")

        vmem = torch.empty_like(inp).contiguous()
        output_spikes = torch.empty_like(inp).contiguous()

        sinabsslayerCuda.lifForward(
            output_spikes,
            vmem,
            inp,
            state,
            activations,
            membrane_subtract,
            alpha,
            threshold,
            threshold_low if threshold_low is not None else 0,
            threshold_low is not None,
        )

        ctx.threshold = threshold
        ctx.threshold_low = threshold_low
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        ctx.membrane_subtract = membrane_subtract
        ctx.alpha = alpha
        ctx.save_for_backward(vmem)

        return output_spikes, vmem

    @staticmethod
    def backward(ctx, grad_output, grad_state):
        (states,) = ctx.saved_tensors

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
            ctx.membrane_subtract * ctx.alpha,
            ctx.alpha,
        )

        # TODO:
        # Currently gradient for `grad_state` is ignored. Would have to add gradient
        # of states to returned grads. Could do this by not multiplying grad_output
        # with surrogates in spikeGradsFull and passing surrogates * grad_output + grad_states
        # instead of grad_output.

        return (
            ctx.scale_rho * grad_input,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
