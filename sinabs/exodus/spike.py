from typing import Callable, Optional

import torch
import exodusCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_mem: torch.tensor,
        membrane_subtract: float,
        surrogate_grad_fn: Callable,
        threshold: float,
        threshold_low: float = None,
    ):
        """
        Generate spikes and apply refractory response to membrane potential, considering
        a non-optional lower limit for the membrane potential. Will modifie membrane
        potential in-place.
        IntegrateAndFires is the faster option.

        Parameters
        ----------
        v_mem: torch.Tensor
            The membrane potential. Expected shape: (N, T_sim), where N is
            *anything* that can be computed in parallel, i.e. batches, neurons...
            Has to be contiguous.
        membrane_subtract: float
            Value that is subracted from membrane potential after spike
        surrogate_grad_fn: Callable
            Calculates surrogate gradients as function of v_mem
        threshold: float
            Firing threshold
        threshold_low: float
            Lower limit for v_mem

        Returns
        -------
        torch.tensor
            Integer spike raster. Same shape as ``v_mem``
        """

        if not v_mem.is_contiguous():
            raise ValueError("'v_mem' has to be contiguous.")
        if not v_mem.ndim == 2:
            raise ValueError("'v_mem' must be 2D, (N, Time)")
        if threshold <= threshold_low:
            raise ValueError("`threshold` must be greater than `threshold_low`.")

        if threshold_low is None:
            spikes = exodusCuda.getSpikes(v_mem, membrane_subtract, threshold)
        else:
            spikes = exodusCuda.getSpikesLB(
                v_mem, membrane_subtract, threshold, threshold_low
            )
        ctx.threshold = threshold
        ctx.threshold_low = threshold_low
        ctx.membrane_subtract = membrane_subtract
        ctx.surrogate_grad_fn = surrogate_grad_fn
        ctx.save_for_backward(v_mem)

        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        (v_mem,) = ctx.saved_tensors

        # Surrogate gradients
        surrogates = ctx.surrogate_grad_fn(v_mem, ctx.threshold)

        if ctx.threshold_low is None:
            # Gradient wrt. input
            grad_input = exodusCuda.spikeGrads(
                surrogates.contiguous(), grad_output.contiguous(), ctx.membrane_subtract
            )
        else:
            # Indicate whether membrane potential (probably) has been clipped
            not_clipped = v_mem > ctx.threshold_low
            # Gradient wrt. input
            grad_input = exodusCuda.spikeGradsLB(
                surrogates.contiguous(),
                grad_output.contiguous(),
                not_clipped.float().contiguous(),
                ctx.membrane_subtract
            )

        return ctx.scale_rho * grad_input, None, None, None, None, None


class IntegrateAndFire(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.tensor,
        membrane_subtract: float,
        alpha: float,
        v_mem_init: torch.tensor,
        activations: torch.tensor,
        threshold: float,
        threshold_low: float,
        surrogate_grad_fn: Callable,
        max_num_spikes_per_bin: Optional[int] = None,
    ):
        """
        Integrate membrane potential with or without leak. Then generate spikes and apply
        reset to membrane potential. Will modifie membrane potential in-place.

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
        v_mem_init : torch.Tensor
            1D shape (N,).  Initial v_mem. Has to be contiguous.
        activations : torch.tensor
            1D, shape (N,). Activations from previous time step.
            Has to be contiguous.
        threshold: float
            Firing threshold
        threshold_low: float
            Lower limit for v_mem
        surrogate_grad_fn: Callable
            Calculates surrogate gradients as function of v_mem
        max_num_spikes_per_bin: int
            Maximum number of neurons that a neuron can emit per time step. Set None to
            remove limit (default).

        Returns
        -------
        torch.tensor (T x T_sim)
            Membrane potential for each neuron and time step
        torch.tensor (N x T_sim)
            Integer spike raster. Same shape as membrane potential
        """

        if not inp.ndim == 2:
            raise ValueError("'inp' must be 2D, (N, Time)")
        if not inp.is_contiguous():
            raise ValueError("'inp' has to be contiguous.")
        if threshold_low is not None and threshold <= threshold_low:
            raise ValueError("`threshold` must be greater than `threshold_low`.")
        if not 0 <= alpha <= 1:
            raise ValueError("'alpha' must be between 0 and 1.")

        v_mem = torch.empty_like(inp).contiguous()
        output_spikes = torch.empty_like(inp).contiguous()

        exodusCuda.lifForward(
            output_spikes,
            v_mem,
            inp,
            v_mem_init,
            activations,
            membrane_subtract,
            alpha,
            threshold,
            threshold_low if threshold_low is not None else 0,
            threshold_low is not None,
            -1 if max_num_spikes_per_bin is None else max_num_spikes_per_bin,
        )

        ctx.threshold = threshold
        ctx.threshold_low = threshold_low
        ctx.surrogate_grad_fn = surrogate_grad_fn
        ctx.membrane_subtract = membrane_subtract
        ctx.alpha = alpha
        ctx.save_for_backward(v_mem)

        return output_spikes, v_mem

    @staticmethod
    def backward(ctx, grad_output, grad_v_mem):

        if torch.nonzero(grad_v_mem).any():
            raise NotImplementedError(
                "Direct Backpropagation through membrane potential is currently not supported."
            )

        (v_mem,) = ctx.saved_tensors

        # Surrogate gradients
        surrogates = ctx.surrogate_grad_fn(v_mem, ctx.threshold)

        # Gradient becomes 0 where v_mem is clipped to lower threshold
        if ctx.threshold_low is None:
            not_clipped = torch.ones_like(surrogates)
        else:
            not_clipped = (v_mem > ctx.threshold_low).float()

        # Gradient wrt. intermediate v_mem
        grad_input = exodusCuda.spikeGradsFull(
            surrogates.contiguous(),
            grad_output.contiguous(),
            not_clipped.contiguous(),
            ctx.membrane_subtract * ctx.alpha,
            ctx.alpha,
        )

        # TODO:
        # Currently gradient for `grad_v_mem` is ignored. Would have to add gradient
        # of v_mem to returned grads. Could do this by not multiplying grad_output
        # with surrogates in spikeGradsFull and passing surrogates * grad_output + grad_v_mem
        # instead of grad_output.

        return (grad_input, None, None, None, None, None, None, None, None)