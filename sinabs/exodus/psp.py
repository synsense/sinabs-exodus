import torch
import exodusCuda


def generateEpsp(
    input_spikes: "torch.tensor", epsp_kernel: "torch.tensor"
) -> "torch.tensor":
    if epsp_kernel.ndim == 1:
        return PspFunction.apply(input_spikes, epsp_kernel)

    if epsp_kernel.ndim == 2:
        out = [PspFunction.apply(s, k) for s, k in zip(input_spikes, epsp_kernel)]
        return torch.stack(out)


class PspFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, spikes: "torch.tensor", kernel: "torch.tensor") -> torch.tensor:
        """
        Generate post-synaptic potential from input spikes.

        Parameters
        ----------
        spikes: torch.Tensor
            Input spikes. Expected shape: (N, T_sim), where N is *anything*
            that can be computed in parallel, i.e. batches, neurons,...
            Has to be contiguous.
        kernel: torch.Tensor
            Refractory response. Has to be 1-dimensional

        Returns
        -------
        torch.tensor
            Integer spikes raster. Same shape as ``spikes``
        """

        if not spikes.is_contiguous():
            raise ValueError("'spikes' has to be contiguous.")
        if not spikes.ndim == 2:
            raise ValueError("'spikes' must be 2D, (N, Time)")
        if not kernel.ndim == 1:
            raise ValueError("'kernel' has to be 1D.")

        psp = exodusCuda.conv(spikes, kernel, 1)

        ctx.save_for_backward(kernel)

        return psp

    @staticmethod
    def backward(ctx, gradOutput):
        (kernel,) = ctx.saved_tensors
        # Note that cross-correlation is defined as (a * b)[T] = sum a[t+T]*b[t],
        # whereas in literature this would often be (b * a)[T].
        gradInput = exodusCuda.corr(gradOutput.contiguous(), kernel, 1)

        # print("psp - grad_out", gradOutput)
        # print("psp - kernel", kernel)
        # print("psp - grad_in", gradInput)
        if kernel.requires_grad is False:
            gradFilter = None
        else:
            gradFilter = kernel

        return gradInput, gradFilter
