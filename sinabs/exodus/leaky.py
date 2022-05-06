import exodus_cuda
import torch


class LeakyIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.tensor,
        state_initial: torch.tensor,
        alpha: float,
        decay_early: bool = False,
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
        state_initial: torch.Tensor
            1D Tensor with initial states for each neuron, shape (N,). Has to
            be contiguous.
        alpha : float
            State decay factor (exp(-dt/tau)). Set 1 for IAF neurons.
        decay_early: bool
            If True, will scale inputs by exp(-1/tau). This corresponds to the Xylo-behavior of
            decaying the input within the same time step.
        """

        if not inp.ndim == 2:
            raise ValueError("'inp' must be 2D (N, Time)")
        if not state_initial.ndim == 1:
            raise ValueError("'state_initial' must be 1D (N,)")

        if decay_early:
            inp = alpha * inp

        states = exodus_cuda.leakyForward(inp, state_initial, alpha)

        ctx.alpha = alpha
        ctx.decay_early = decay_early

        return states

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = exodus_cuda.leakyBackward(grad_output, ctx.alpha)

        if ctx.decay_early:
            grad_input = ctx.alpha * grad_input

        return grad_input, None, None, None, None
