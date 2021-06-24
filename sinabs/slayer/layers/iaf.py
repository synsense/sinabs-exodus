from typing import Optional

from sinabs.slayer.kernels import heaviside_kernel
from sinabs.slayer.psp import generateEpsp
from sinabs.slayer.spike import spikeFunction
from sinabs.layers.pack_dims import squeeze_class
from sinabs.layers import IAF


__all__ = ["IAFSlayer", "IAFSlayerSqueeze"]


class IAFSlayer(IAF):
    def __init__(
        self,
        num_timesteps: int,
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
        num_timesteps: int
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
        self.num_timesteps = num_timesteps

        if threshold_low is not None:
            raise NotImplementedError("Lower threshold not implemented for this layer.")

        epsp_kernel = heaviside_kernel(size=num_timesteps, scale=1.0)

        if membrane_subtract is None:
            membrane_subtract = threshold
        ref_kernel = heaviside_kernel(size=num_timesteps, scale=membrane_subtract)
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
            Expected shape: (n_batches, num_timesteps, ...)

        Returns
        -------
        torch.tensor
            Output spikes. Same shape as `spike_input`.
        """

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        spike_input = spike_input.movedim(1, -1)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        shape_before_flat = spike_input.shape
        spike_input = spike_input.reshape(-1, spike_input.shape[-1]).contiguous()
        # -> (n_parallel, num_timesteps)
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

        # Restore original shape: (n_batches, num_timesteps, *neuron_shape)
        vmem = vmem.reshape(*shape_before_flat).movedim(-1, 1)
        output_spikes = output_spikes.reshape(*shape_before_flat).movedim(-1, 1)

        self.vmem = vmem.clone()
        self.tw = self.num_timesteps

        self.spikes_number = output_spikes.sum()

        return output_spikes


# Class to accept data with batch and time dimensions combined
IAFSlayerSqueeze = squeeze_class(IAFSlayer)
