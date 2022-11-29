import exodus_cuda
import torch


class LeakyIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.tensor,
        alpha: torch.tensor,
        v_mem_init: torch.tensor,
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

        out = exodus_cuda.leakyForward(inp, v_mem_init, alpha)

        ctx.get_alpha_grads = alpha.requires_grad
        if alpha.requires_grad:
            ctx.save_for_backward(out, v_mem_init, alpha)
        else:
            ctx.save_for_backward(alpha)

        return out

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.get_alpha_grads:
            out, v_mem_init, alpha = ctx.saved_tensors
            grad_alpha = exodus_cuda.leakyBackwardAlpha(
                grad_output.contiguous(),
                out.contiguous(),
                v_mem_init.contiguous(),
                alpha.contiguous(),
            )
        else:
            (alpha,) = ctx.saved_tensors
            grad_alpha = None

        grad_input = exodus_cuda.leakyBackward(
            grad_output.contiguous(), alpha.contiguous()
        )

        grad_init = alpha * grad_input[:, 0]

        # Backpropagate one more decay step from first time point.
        # Works because v_1 = alpha * v_init
        return grad_input, grad_alpha, grad_init
