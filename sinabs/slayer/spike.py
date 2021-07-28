from typing import Optional

import torch
import sinabsslayerCuda


class SpikeFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        membr_pot: torch.tensor,
        refr_response: torch.tensor,
        threshold: float,
        window: Optional[float] = None,
        scale_rho: float = 1.0,
    ):
        """
        Generate spikes and apply refractory response to membrane potential.
        Will modifie membrane potential in-place.

        Parameters
        ----------
        membr_pot : torch.Tensor
            The membrane potential. Expected shape: (N, T_sim), where N is
            the product of all dimensions that can be computed in parallel,
            i.e. batches, neurons...
            Has to be contiguous.
        refr_response : torch.Tensor
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
        if not refr_response.ndim == 1:
            raise ValueError("'refr_response' has to be 1D.")

        spikes = sinabsslayerCuda.getSpikes(membr_pot, refr_response, threshold, 1.0)

        # Prepare backward
        ctx.threshold = threshold
        ctx.scale_rho = scale_rho
        ctx.window = window or threshold
        # ctx.subtract = refr_response[1]
        ctx.refr_response = refr_response
        ctx.save_for_backward(membr_pot)

        return spikes

    @staticmethod
    def backward(ctx, grad_output):
        """"""
        # spike_pdf = (membr_pot >= (ctx.threshold - ctx.window)).float() / ctx.threshold

        # print("spike - grad_out", grad_output)
        # print("spike - membr_pot", membr_pot)
        # print("spike - pdf", spike_pdf)
        # print("spike - grad_in", spike_pdf * grad_output * ctx.scale_rho)

        # return spike_pdf * grad_output * ctx.scale_rho, None, None, None, None

        # n = grad_output.shape[1]
        membr_pot, = ctx.saved_tensors

        # Derivatives within individual timesteps: d(spikes)_t / d(membr_pot)_t
        # Heaviside surrogate gradients
        surrogates = (membr_pot >= (ctx.threshold - ctx.window)).float() / ctx.threshold
        # print("surrogates", surrogates)

        # Calculate transposed jacobian matrix
        # jaco_t = sinabsslayerCuda.spikeGrads(
        #     surrogates.clone(), ctx.refr_response.clone()
        # )
        # print("jaco", jaco_t)
        # jaco_t = torch.zeros(
        #     (*grad_output.shape, grad_output.shape[1]), device="cuda"
        # ).float()
        # grad_input = torch.stack([j @ o for j, o in zip(jaco_t, grad_output)])
        grad_input = sinabsslayerCuda.spikeGrads(
            surrogates, ctx.refr_response, grad_output
        )
        # print("Grad out", grad_output)
        # print("Grad in", grad_input)
        # grad_input = torch.zeros_like(grad_output)

        # #
        # chi = ctx.scale_rho / ctx.threshold
        # gamma = ctx.subtract * chi ** 2
        # alphas = torch.tensor(
        #     [chi] + [gamma * (1 + ctx.subtract * chi) ** k for k in range(n - 1)],
        #     device="cuda",
        # ).contiguous()

        # # Actually want corr(alphas, grad_output) but the kernel function expects the
        # # higher dimensional argument first, so we need a little trick here.
        # grad_input = sinabsslayerCuda.conv(
        #     grad_output.flip(1).contiguous(), alphas, 1
        # ).flip(1)
        # jacobian = torch.diag(torch.tensor(n * [chi]))
        # for i, a in enumerate(alphas):
        #     jacobian += torch.diag(torch.tensor((n - i - 1) * [a]), i + 1)
        # jacobian = jacobian.cuda()

        # grad_input = torch.stack([jacobian @ out_batch for out_batch in grad_output])

        #  print("spike - alphas", alphas)
        # print("spike - grad_out", grad_output)
        # print("spike - transposed jacobian \n", jaco_t)
        # print("spike - grad_in", grad_input)

        return grad_input, None, None, None, None


spikeFunction = SpikeFunction().apply
