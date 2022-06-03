import exodus_cuda
import torch


class LeakyIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.tensor,
        alpha: torh.tensor,
        v_mem_init: torch.tensor,
        decay_early: bool = False,
        get_alpha_grads: bool = False,
    ):
        """
        Evolve a leaky integrator as
        v_t = alpha * v_{t-1} + input_{t}

        Parameters
        ----------
        inp: torch.tensor
            2D input tensor, shape: (N, T_sim), where N is
            *anything* that can be computed in parallel, i.e. batches, neurons...
            Has to be contiguous.
        alpha : torch.Tensor
            1D, shape: (N,). State decay factor (exp(-dt/tau)). Set 1 for IAF neurons.
        v_mem_init: torch.Tensor
            1D Tensor with initial states for each neuron, shape (N,). Has to
            be contiguous.
        decay_early: bool
            If True, will scale inputs by exp(-1/tau). This corresponds to the Xylo-behavior of
            decaying the input within the same time step.
        get_alpha_grads: bool
            If True, gradients for alpha will be calculated during backward call.
        """

        if not inp.ndim == 2:
            raise ValueError("'inp' must be 2D, (N, Time)")
        if not inp.is_contiguous():
            raise ValueError("'inp' has to be contiguous.")
        if not alpha.ndim == 1:
            raise ValueError("'alpha' must be 1D, (N,)")
        if not alpha.is_contiguous():
            raise ValueError("'alpha' has to be contiguous.")
        if not v_mem_init.ndim == 1:
            raise ValueError("'v_mem_init' must be 1D, (N,)")
        if not v_mem_init.is_contiguous():
            raise ValueError("'v_mem_init' has to be contiguous.")

        if decay_early:
            inp = alpha * inp

        states = exodus_cuda.leakyForward(inp, v_mem_init, alpha)

        ctx.alpha = alpha
        ctx.decay_early = decay_early
        ctx.get_alpha_grads = get_alpha_grads

        return states

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = exodus_cuda.leakyBackward(grad_output, ctx.alpha)

        if ctx.decay_early:
            grad_input = ctx.alpha * grad_input

        return grad_input, None, None, None, None, None
