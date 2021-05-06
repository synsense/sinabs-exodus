from typing import Optional

from torch import nn

from sinabs.slayer.kernels import heaviside_kernel
from sinabs.slayer.psp import generateEpsp
from sinabs.slayer.spike import spikeFunction


class SpikingLayer(nn.Module):
    def __init__(
        self,
        t_sim: int,
        threshold: float = 1.0,
        membrane_subtract: Optional[float] = None,
        tau_learning: float = 0.5,
        scale_grads: float = 1.0,
        threshold_low=None,
    ):
        """
        Pytorch implementation of a spiking, non-leaky, IAF neuron with learning enabled.

        Parameters:
        -----------
        t_sim: int
            Number of timesteps per sample.
        threshold: float
            Spiking threshold of the neuron.
        membrane_subtract: Optional[float]
            Constant to be subtracted from membrane potential when neuron spikes.
            If ``None`` (default): Same as ``threshold``.
        tau_learning: float
            How fast do surrogate gradients decay around thresholds.
        scale_grads: float
            Scale surrogate gradients in backpropagation.
        threshold_low: None
            Currently not supported.
        """
        super().__init__()

        # - Store hyperparameters
        self.threshold = threshold
        self.scale_grads = scale_grads
        self.tau_learning = tau_learning
        self.t_sim = t_sim

        if threshold_low is not None:
            raise NotImplementedError("Lower threshold not implemented for this layer.")

        epsp_kernel = heaviside_kernel(size=t_sim, scale=1.0)

        if membrane_subtract is None:
            membrane_subtract = threshold
        ref_kernel = heaviside_kernel(size=t_sim, scale=membrane_subtract)
        assert ref_kernel.ndim == 1

        # Blank parameter place holders
        self.register_buffer("epsp_kernel", epsp_kernel)
        self.register_buffer("ref_kernel", ref_kernel)
        self.spikes_number = None

    def forward(self, spike_input: "torch.tensor") -> "torch.tensor":
        """
        Generate membrane potential and resulting output spike train based on
        spiking input. Membrane potential will be stored as `self.vmem`.

        Parameters
        ----------
        spike_input: torch.tensor
            Spike input raster. May take non-binary integer values.
            Expected shapes:
                5D: (n_batches, t_sim, channels, height, width)
                or 4D: (n_batches x t_sim, channels, height, width)

        Returns
        -------
        torch.tensor
            Output spikes. Same shape as `spike_input`.
        """

        if spike_input.ndim == 5:
            # Expected input dimension: (n_batches, t_sim, *neuron_shape )
            seperate_batch_time = True

        else:
            # Expected input dimension: (n_batches x t_sim, *neuron_shape )
            seperate_batch_time = False
            neuron_shape = spike_input.shape[1:]
            try:
                # Separate batch and time dimensions
                spike_input = spike_input.reshape(-1, self.t_sim, *neuron_shape)
            except RuntimeError:
                raise ValueError(
                    f"First input dimension (time) must be multiple of {self.t_sim}"
                    + f" but is {spike_input.shape[0]}."
                )
        # Shape is now (n_batches, t_sim, *neuron_shape)

        # Move time to last dimension
        spike_input = spike_input.movedim(1, -1)  # -> (n_batches, *neuron_shape, t_sim)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        shape_before_flat = spike_input.shape
        spike_input = spike_input.reshape(-1, spike_input.shape[-1]).contiguous()
        # -> (n_parallel, t_sim)
        assert spike_input.ndim == 2
        assert spike_input.is_contiguous()

        vmem = generateEpsp(spike_input, self.epsp_kernel)

        assert self.ref_kernel.ndim == 1
        assert vmem.ndim == 2
        assert vmem.is_contiguous()

        output_spikes = spikeFunction(
            vmem, -self.ref_kernel, self.threshold, self.tau_learning, self.scale_grads
        )

        assert vmem.ndim == 2

        # Restore original 5d-shape: (n_batches, t_sim, *neuron_shape)
        vmem = vmem.reshape(*shape_before_flat).movedim(-1, 1)
        output_spikes = output_spikes.reshape(*shape_before_flat).movedim(-1, 1)

        if not seperate_batch_time:
            # Combine batch and time dimensions -> (n_batches x t_sim, *neuron_shape)
            vmem = vmem.reshape(-1, *vmem.shape[2:])
            output_spikes = output_spikes.reshape(-1, *output_spikes.shape[2:])

        self.vmem = vmem.clone()
        self.tw = self.t_sim

        self.spikes_number = output_spikes.sum()

        return output_spikes
