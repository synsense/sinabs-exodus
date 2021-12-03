import sinabsslayerCuda
import torch


class LeakyIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp: torch.tensor, state_initial: torch.tensor, alpha: float):
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
        """

        if not inp.ndim == 2:
            raise ValueError("'inp' must be 2D (N, Time)")
        if not state_initial.ndim == 1:
            raise ValueError("'state_initial' must be 1D (N,)")

        states = sinabsslayerCuda.leakyForward(inp, state_initial, alpha)

        ctx.alpha = alpha

        return states

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = sinabsslayerCuda.leakyBackward(grad_output, ctx.alpha)

        return grad_input, None, None
