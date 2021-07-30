from typing import Optional

from sinabs.slayer.kernels import heaviside_kernel
from sinabs.slayer.psp import generateEpsp
from sinabs.layers.pack_dims import squeeze_class
from sinabs.slayer.layers import SpikingLayer


__all__ = ["IAF", "IAFSqueeze"]


class IAF(SpikingLayer):
    def __init__(
        self,
        num_timesteps: int,
        threshold: float = 1.0,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        window: float = 1.0,
        scale_grads: float = 1.0,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a spiking, non-leaky, IAF neuron with learning enabled.

        Parameters:
        -----------
        num_timesteps: int
            Number of timesteps per sample.
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: Optional[float]
            Lower bound for membrane potential.
        membrane_subtract: Optional[float]
            Constant to be subtracted from membrane potential when neuron spikes.
            If ``None`` (default): Same as ``threshold``.
        window: float
            Distance between step of Heaviside surrogate gradient and threshold.
        scale_grads: float
            Scale surrogate gradients in backpropagation.
        membrane_reset: bool
            Currently not supported.
        """

        if membrane_reset:
            raise NotImplementedError("Membrane reset not implemented for this layer.")

        super().__init__(
            num_timesteps=num_timesteps,
            threshold=threshold,
            threshold_low=threshold_low,
            window=window,
            scale_grads=scale_grads,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )

        # - Initialize kernels
        epsp_kernel = heaviside_kernel(size=num_timesteps, scale=1.0)
        ref_kernel = heaviside_kernel(size=num_timesteps, scale=self.membrane_subtract)
        assert ref_kernel.ndim == 1

        self.register_buffer("epsp_kernel", epsp_kernel)
        self.register_buffer("ref_kernel", ref_kernel)

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

        n_batches, num_timesteps, *n_neurons = spike_input.shape

        # Make sure time dimension matches
        if num_timesteps != self.num_timesteps:
            raise ValueError(
                f"Time (2nd) dimension of `spike_input` must be {self.num_timesteps}"
            )

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        spike_input = spike_input.movedim(1, -1)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.reshape(-1, num_timesteps).contiguous()
        # -> (n_parallel, num_timesteps)

        vmem = generateEpsp(spike_input, self.epsp_kernel).contiguous()

        output_spikes = self.spike_function(vmem)

        return self._post_spike_processing(vmem, output_spikes, n_batches, n_neurons)


# Class to accept data with batch and time dimensions combined
IAFSqueeze = squeeze_class(IAF)
