from typing import Optional, Tuple

import torch

from sinabs.slayer.spike import spikeFunctionIterForward
from sinabs.layers import SpikingLayer


__all__ = ["SpikingLayer"]


class IntegrateFireBase(SpikingLayer):
    """
    Slayer implementation of a leaky or non-leaky integrate and fire neuron with
    learning enabled. Does not simulate synaptic dynamics.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        window: float = 1.0,
        scale_grads: float = 1.0,
        alpha: float = 1.0,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Slayer implementation of a leaky or non-leaky integrate and fire neuron with
        learning enabled. Does not simulate synaptic dynamics.

        Parameters:
        -----------
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: Optional[float]
            Lower bound for membrane potential.
        membrane_subtract: Optional[float]
            Constant to be subtracted from membrane potential when neuron spikes.
            If ``None`` (default): Same as ``threshold``.
        window: float
            Distance between step of Heaviside surrogate gradient and threshold.
            (Relative to size of threshold)
        scale_grads: float
            Scale surrogate gradients in backpropagation.
        decay_factor: float
            Neuron state is multiplied by this factor at each timestep. For IAF dynamics
            set to 1, for LIF to exp(-dt/tau).
        membrane_reset: bool
            Currently not supported.
        """

        if membrane_reset:
            raise NotImplementedError("Membrane reset not implemented for this layer.")
        if not (0 < alpha <= 1.0):
            raise ValueError("`alpha` must be between 0 and 1.")

        super().__init__(
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )

        # - Store hyperparameters
        self.scale_grads = scale_grads
        self.window_abs = window * threshold
        self.alpha = alpha

    def spike_function(self, inp: "torch.tensor") -> "torch.tensor":
        """
        Generate spikes from membrane potential.

        Parameters
        ----------
        inp : torch.tensor
            Input to the neuron. Expected shape: (batches x neurons, time)
            Has to be contiguous.

        Returns:
        --------
        torch.tensor
            Membrane potential. Same shape as `spike_input`
        torch.tensor
            Output spikes. Same shape as `spike_input`
        """

        # Note: Do not make `vmem` contiguous only here because a new object will
        # be created, so that any modifications (membrane reset etc.) would not
        # have effect on the original `vmem`.

        # Generate output_spikes
        return spikeFunctionIterForward(
            inp,
            self.membrane_subtract,
            self.alpha,
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

    def _post_spike_processing(
        self,
        vmem: "torch.tensor",
        output_spikes: "torch.tensor",
        n_batches: int,
        n_neurons: Tuple[int],
    ) -> "torch.tensor":
        """
        Handle post-processing after spikes have been generated: Reshape output spikes,
        reshape and record vmem, record output spike statistics. Update neuron states.

        Parameters
        ----------
        vmem : torch.tensor
            Membrane potential. Expected shape: (batches x neurons, time)
        output_spikes : torch.tensor
            Output spikes. Expected shape: (batches x neurons, time)
        n_batches : int
            Number of batches. Needed for correct reshaping of output and vmem
        n_neurons : Tuple of ints
            Remaining neuron dimensions (e.g. channels, height, width). Determines output
            shape after batch and time dimensions.

        Returns
        -------
        torch.tensor
            Output spikes. Shape: (n_batches, time, *n_neurons)
        """

        # Separate batch and neuron dimensions -> (batches, *neurons, time)
        vmem = vmem.reshape(n_batches, *n_neurons, -1)
        output_spikes = output_spikes.reshape(n_batches, *n_neurons, -1)

        # Move time dimension to second -> (batches, time, *neurons)
        vmem = vmem.movedim(-1, 1)
        output_spikes = output_spikes.movedim(-1, 1)

        if self.record:
            # Record states and output
            self.vmem = vmem
            self.spikes_out = output_spikes

        self.spikes_number = output_spikes.sum()

        self._update_neuron_states(vmem, output_spikes)

        return output_spikes

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

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        spike_input = spike_input.movedim(1, -1)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.reshape(-1, num_timesteps)
        # -> (n_parallel, num_timesteps)

        output_spikes, states = self.spike_function(spike_input)

        # Reshape output spikes and vmem, store neuron states
        output_spikes = self._post_spike_processing(
            states, output_spikes, n_batches, n_neurons
        )

        return output_spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict()
        param_dict.update(
            scale_grads=self.scale_grads,
            window=self.window_abs / self.threshold,
            alpha=self.alpha,
        )
