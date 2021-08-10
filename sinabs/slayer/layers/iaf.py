from typing import Optional

from sinabs.slayer.kernels import heaviside_kernel
from sinabs.slayer.spike import spikeFunctionIterForward
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

    def spike_function(self, spike_input):
        """
        Generate membrane potential and output spikes from membrane potential.

        Parameters
        ----------
        spike_input : torch.tensor
            Input spikes. Expected shape: (batches x neurons, time)
            Has to be contiguous.

        Returns:
        --------
        torch.tensor
            Membrane potential. Same shape as `spike_input`
        torch.tensor
            Output spikes. Same shape as `spike_input`
        """
        return spikeFunctionIterForward(
            spike_input,
            self.membrane_subtract,
            self.state.flatten(),
            self.activations.flatten(),
            self.threshold,
            self.threshold_low,
            self.window_abs,
            self.scale_grads,
        )

    def _update_neuron_states(
        self, vmem: "torch.tensor", output_spikes: "torch.tensor"
    ):
        """
        Update neuron states based on membrane potential and output spikes of
        last evolution.

        Parameters
        ----------
        vmem : torch.tensor
            Membrane potential of last evolution. Shape: (batches, num_timesteps, *neurons)
        output_spikes : torch.tensor
            Output spikes of last evolution. Shape: (batches, num_timesteps, *neurons)
        """

        self.state = vmem[:, -1].clone()
        self.activations = output_spikes[:, -1].clone()

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

        if not hasattr(self, "state") or self.state.shape != (n_batches, *n_neurons):
            self.reset_states(shape=(n_batches, *n_neurons), randomize=False)

        # Make sure time dimension matches
        if num_timesteps != self.num_timesteps:
            raise ValueError(
                f"Time (2nd) dimension of `spike_input` must be {self.num_timesteps}"
            )

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        spike_input = spike_input.movedim(1, -1)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.reshape(-1, num_timesteps)
        # -> (n_parallel, num_timesteps)

        output_spikes, states = self.spike_function(spike_input)

        # Reshape output spikes and vmem, store states in vmem
        output_spikes = self._post_spike_processing(
            states, output_spikes, n_batches, n_neurons
        )

        self._update_neuron_states(self.vmem, output_spikes)

        return output_spikes


# Class to accept data with batch and time dimensions combined
IAFSqueeze = squeeze_class(IAF)
