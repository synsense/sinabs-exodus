from typing import Callable, Optional

import torch
import exodus_cuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        v_mem: torch.tensor,
        membrane_subtract: float,
        alpha: float,
        surrogate_grad_fn: Callable,
        threshold: float,
        min_v_mem: Optional[float] = None,
        max_num_spikes_per_bin: Optional[int] = None,
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
        alpha : float
            State decay factor (exp(-dt/tau)). Set 1 for IAF neurons.
        surrogate_grad_fn: Callable
            Calculates surrogate gradients as function of v_mem
        threshold: float
            Firing threshold
        min_v_mem: float
            Lower limit for v_mem
        max_num_spikes_per_bin: int
            Maximum number of neurons that a neuron can emit per time step. Set None to
            remove limit (default).

        Returns
        -------
        torch.tensor
            Integer spike raster. Same shape as ``v_mem``
        """

        if not v_mem.is_contiguous():
            raise ValueError("'v_mem' has to be contiguous.")
        if not v_mem.ndim == 2:
            raise ValueError("'v_mem' must be 2D, (N, Time)")
        if min_v_mem is not None and (threshold <= min_v_mem):
            raise ValueError("`threshold` must be greater than `min_v_mem`.")

        spikes = exodus_cuda.spikeForward(
            v_mem,
            alpha,
            membrane_subtract,
            threshold,
            0 if min_v_mem is None else threshold,  # min_v_mem
            min_v_mem is not None,  # Apply min_v_mem
            -1 if max_num_spikes_per_bin is None else max_num_spikes_per_bin
        )

        ctx.alpha = alpha
        ctx.threshold = threshold
        ctx.min_v_mem = min_v_mem
        ctx.membrane_subtract = membrane_subtract
        ctx.surrogate_grad_fn = surrogate_grad_fn
        ctx.save_for_backward(v_mem)

        return spikes, v_mem

    @staticmethod
    def backward(ctx, grad_output, grad_v_mem):

        if torch.nonzero(grad_v_mem).any():
            raise NotImplementedError(
                "Direct Backpropagation through membrane potential is currently not supported."
            )

        (v_mem,) = ctx.saved_tensors

        # Surrogate gradients
        surrogates = ctx.surrogate_grad_fn(v_mem, ctx.threshold)

        if ctx.min_v_mem is None:
            not_clipped = torch.ones_like(surrogates)
        else:
            # Indicate whether membrane potential (probably) has been clipped
            not_clipped = v_mem > ctx.min_v_mem
        # Gradient wrt. input
        grad_input = exodus_cuda.spikeBackward(
            surrogates.contiguous(),
            grad_output.contiguous(),
            not_clipped.float().contiguous(),
            ctx.alpha,
            ctx.membrane_subtract,
        )

        return grad_input, None, None, None, None, None, None


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
        min_v_mem: float,
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
        min_v_mem: float
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
        if min_v_mem is not None and threshold <= min_v_mem:
            raise ValueError("`threshold` must be greater than `min_v_mem`.")
        if not 0 <= alpha <= 1:
            raise ValueError("'alpha' must be between 0 and 1.")

        v_mem = torch.empty_like(inp).contiguous()
        output_spikes = torch.empty_like(inp).contiguous()

        exodus_cuda.lifForward(
            output_spikes,
            v_mem,
            inp,
            v_mem_init,
            activations,
            membrane_subtract,
            alpha,
            threshold,
            min_v_mem if min_v_mem is not None else 0,
            min_v_mem is not None,
            -1 if max_num_spikes_per_bin is None else max_num_spikes_per_bin,
        )

        ctx.threshold = threshold
        ctx.min_v_mem = min_v_mem
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
        if ctx.min_v_mem is None:
            not_clipped = torch.ones_like(surrogates)
        else:
            not_clipped = (v_mem > ctx.min_v_mem).float()

        # Gradient wrt. intermediate v_mem
        grad_input = exodus_cuda.lifBackward(
            surrogates.contiguous(),
            grad_output.contiguous(),
            not_clipped.contiguous(),
            ctx.membrane_subtract * ctx.alpha,
            ctx.alpha,
        )

        return (grad_input, None, None, None, None, None, None, None, None)
