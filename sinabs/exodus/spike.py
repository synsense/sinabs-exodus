from typing import Callable, Optional, Union

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
        if min_v_mem is not None and (threshold <= min_v_mem).any():
            raise ValueError("`threshold` must be greater than `min_v_mem`.")

        spikes = exodus_cuda.spikeForward(
            v_mem,
            alpha,
            membrane_subtract,
            threshold,
            0 if min_v_mem is None else threshold,  # min_v_mem
            min_v_mem is not None,  # Apply min_v_mem
            -1 if max_num_spikes_per_bin is None else max_num_spikes_per_bin,
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
        alpha: torch.tensor,
        v_mem_init: torch.tensor,
        threshold: torch.tensor,
        membrane_subtract: torch.tensor,
        min_v_mem: Union[torch.tensor, None],
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
        alpha : torch.Tensor
            1D shape (N,). State decay factor (exp(-dt/tau)). Set 1 for IAF neurons.
        v_mem_init : torch.Tensor
            1D shape (N,).  Initial v_mem (after reset). Has to be contiguous.
        activations : torch.tensor
            1D, shape (N,). Activations from previous time step.
            Has to be contiguous.
        threshold: torch.tensor
            1D, shape (N,). Firing thresholds
        membrane_subtract: torch.Tensor
            1D, shape (N,). Value that is subracted from membrane potential after spike
        min_v_mem: torch.Tensor or None
            1D, shape (N,). Lower limits for v_mem. If 'None', don't apply limits
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

        if membrane_subtract is None:
            membrane_subtract = torch.ones_like(alpha) * threshold
        if not (apply_min_v_mem := (min_v_mem is not None)):
            # Pass some empty tensor to match CUDA function signature
            min_v_mem = torch.empty_like(threshold)

        if not inp.ndim == 2:
            raise ValueError("'inp' must be 2D, (N, Time)")
        if not inp.is_contiguous():
            raise ValueError("'inp' has to be contiguous.")
        if not alpha.ndim == 1:
            raise ValueError("'alpha' must be 1D, (N,)")
        if not alpha.is_contiguous():
            raise ValueError("'alpha' has to be contiguous.")
        if not membrane_subtract.ndim == 1:
            raise ValueError("'membrane_subtract' must be 1D, (N,)")
        if not membrane_subtract.is_contiguous():
            raise ValueError("'membrane_subtract' has to be contiguous.")
        if not v_mem_init.ndim == 1:
            raise ValueError("'v_mem_init' must be 1D, (N,)")
        if not v_mem_init.is_contiguous():
            raise ValueError("'v_mem_init' has to be contiguous.")
        if not threshold.ndim == 1:
            raise ValueError("'threshold' must be 1D, (N,)")
        if not threshold.is_contiguous():
            raise ValueError("'threshold' has to be contiguous.")
        if (alpha < 0).any() or (alpha > 1).any():
            raise ValueError("'alpha' must be between 0 and 1.")
        if apply_min_v_mem:
            if not min_v_mem.ndim == 1:
                raise ValueError("'min_v_mem' must be 1D, (N,)")
            if not min_v_mem.is_contiguous():
                raise ValueError("'min_v_mem' has to be contiguous.")
            if (threshold <= min_v_mem).any():
                raise ValueError("`threshold` must be greater than `min_v_mem`.")

        v_mem = torch.empty_like(inp).contiguous()
        output_spikes = torch.empty_like(inp).contiguous()

        exodus_cuda.lifForward(
            output_spikes,
            v_mem,
            inp,
            v_mem_init,
            alpha,
            membrane_subtract,
            threshold,
            min_v_mem,
            apply_min_v_mem,
            -1 if max_num_spikes_per_bin is None else max_num_spikes_per_bin,
        )

        ctx.surrogate_grad_fn = surrogate_grad_fn
        ctx.apply_min_v_mem = apply_min_v_mem
        # vmem is stored before reset (to calculate surrogate gradients in backward)
        # however, vmem_initial should already have reset applied
        if alpha.requires_grad:
            ctx.save_for_backward(
                output_spikes,
                v_mem,
                v_mem_init,
                alpha,
                membrane_subtract,
                threshold,
                min_v_mem,
            )
        else:
            ctx.save_for_backward(v_mem, alpha, membrane_subtract, threshold, min_v_mem)
        ctx.get_alpha_grads = alpha.requires_grad

        return output_spikes, v_mem

    @staticmethod
    def backward(ctx, grad_output, grad_v_mem):

        # if (grad_v_mem != 0).any():
        #     raise NotImplementedError(
        #         "Direct Backpropagation through membrane potential is currently not supported."
        #     )

        if ctx.get_alpha_grads:
            (
                output_spikes,
                v_mem,
                v_mem_init,
                alpha,
                membrane_subtract,
                threshold,
                min_v_mem,
            ) = ctx.saved_tensors
        else:
            (v_mem, alpha, membrane_subtract, threshold, min_v_mem) = ctx.saved_tensors

        # Surrogate gradients
        surrogates = ctx.surrogate_grad_fn(v_mem, threshold.unsqueeze(1))

        # Gradient becomes 0 where v_mem is clipped to lower threshold
        if ctx.apply_min_v_mem:
            not_clipped = (v_mem > min_v_mem.unsqueeze(1)).float()
        else:
            not_clipped = torch.ones_like(surrogates)

        # Gradient wrt. input
        # Scaling membrane_subtract with alpha compensates for different execution order
        # in forward pass (i.e. reset happens after spiking and before decay, whereas
        # backward pass assumes reset to happen after decay)
        surrogates = surrogates.contiguous()
        grad_input = exodus_cuda.lifBackward(
            surrogates,
            surrogates * grad_output.contiguous() + grad_v_mem.contiguous(),
            not_clipped.contiguous(),
            alpha.contiguous(),
            alpha * membrane_subtract.contiguous(),
        )

        # Gradient wrt alpha
        if ctx.get_alpha_grads:
            v_mem_post = v_mem - membrane_subtract.unsqueeze(1) * output_spikes
            grad_alpha = exodus_cuda.lifBackwardAlpha(
                surrogates,
                surrogates * grad_output.contiguous() + grad_v_mem.contiguous(),
                v_mem_post.contiguous(),
                v_mem_init.contiguous(),
                not_clipped.contiguous(),
                alpha.contiguous(),
                membrane_subtract.contiguous(),
            )
        else:
            grad_alpha = None

        # Backpropagate one more decay step from first time point.
        # Works because d v_1 / d inp_1 = 1 and reset on v_mem_ini is done externally.
        grad_init = alpha * grad_input[:, 0]

        return (grad_input, grad_alpha, grad_init, None, None, None, None, None, None)
