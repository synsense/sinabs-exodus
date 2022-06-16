import exodus_cuda
import torch


class LeakyIntegrator(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inp: torch.tensor,
        alpha: torch.tensor,
        v_mem_init: torch.tensor,
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
        alpha : torch.Tensor
            1D, shape: (N,). State decay factor (exp(-dt/tau)). Set 1 for IAF neurons.
        v_mem_init: torch.Tensor
            1D Tensor with initial states for each neuron, shape (N,). Has to
            be contiguous.
        decay_early: bool
            If True, will scale inputs by exp(-1/tau). This corresponds to the Xylo-behavior of
            decaying the input within the same time step.
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

        ctx.decay_early = decay_early
        ctx.get_alpha_grads = alpha.requires_grad
        if alpha.requires_grad:
            # For alpha grads, we need the unscaled output states, even with early decay
            ctx.save_for_backward(out, alpha)
        else:
            ctx.save_for_backward(alpha)

        if decay_early:
            # Early decay is like rescaling inp with alpha. Due to linearity, it is
            # equivalent to scaling the output states. For the alpha gradients.
            out = alpha.view(-1, 1) * out

        return out

    @staticmethod
    def backward(ctx, grad_output):

        if ctx.get_alpha_grads:
            out, alpha = ctx.saved_tensors
            grad_alpha = exodus_cuda.leakyBackwardAlpha(grad_output, out, alpha)
            print("out:", out)
            print("do:", grad_output)
            print("alpha:", alpha)
            print("da:", grad_alpha)
        else:
            (alpha, ) = ctx.saved_tensors
            grad_alpha = None

        grad_input = exodus_cuda.leakyBackward(grad_output, alpha)

        if ctx.decay_early:
            grad_input = alpha.view(-1, 1) * grad_input

            if ctx.get_alpha_grads:
                # Because inp was replaced by alpha * inp, the actual output o' is
                # the original output o times alpha. Using the product rule, the
                # new gradient is \frac{do'}{d\alpha} = \frac{do}{d\alpha} + o ,
                # which needs to be multiplied with the output gradients
                grad_alpha += torch.matmul(out, grad_output.t())
                print("da new:", grad_alpha)

        return grad_input, grad_alpha, None, None
