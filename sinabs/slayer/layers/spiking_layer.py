from typing import Optional, Tuple

import torch

from sinabs.slayer.spike import SpikeFunctionIterForward
from sinabs.layers import SpikingLayer


__all__ = ["IntegrateFireBase"]


class IntegrateFireBase(SpikingLayer):
    """
    Slayer implementation of a leaky or non-leaky integrate and fire neuron with
    learning enabled. Does not simulate synaptic dynamics.
    """

    backend = "slayer"

    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        window: float = 1.0,
        scale_grads: float = 1.0,
        alpha_mem: float = 1.0,
        record: bool = True,
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
        alpha_mem: float
            Neuron state is multiplied by this factor at each timestep. For IAF dynamics
            set to 1, for LIF to exp(-dt/tau).
        record: bool
            Record membrane potential and spike output during forward call.
        membrane_reset: bool
            Currently not supported.
        """

        if membrane_reset:
            raise NotImplementedError("Membrane reset not implemented for this layer.")
        if not (0 < alpha_mem <= 1.0):
            raise ValueError("`alpha_mem` must be between 0 and 1.")

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
        self.learning_window = window * threshold
        self.alpha_mem = alpha_mem
        self.record = record

        # - Make sure buffers and parameters are on cuda
        self.cuda()

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

        # Note: Do not make `inp` contiguous only here because a new object will
        # be created, so that any modifications (membrane reset etc.) would not
        # have effect on the original `inp`.

        # Generate output_spikes
        return SpikeFunctionIterForward.apply(
            inp,
            self.membrane_subtract,
            self.alpha_mem,
            self.v_mem.flatten(),
            self.activations.flatten(),
            self.threshold,
            self.threshold_low,
            self.learning_window,
            self.scale_grads,
        )

    def _update_neuron_states(
        self, v_mem_full: "torch.tensor", output_spikes: "torch.tensor"
    ):
        """
        Update neuron states based on membrane potential and output spikes of
        last evolution.

        Parameters
        ----------
        v_mem_full : torch.tensor
            Membrane potential of last evolution. Shape: (batches, num_timesteps, *neurons)
        output_spikes : torch.tensor
            Output spikes of last evolution. Shape: (batches, num_timesteps, *neurons)
        """

        self.v_mem = v_mem_full[:, -1].clone()
        self.activations = output_spikes[:, -1].clone()
        self.tw = v_mem_full.shape[1]
        self.spikes_number = output_spikes.sum()

    def _post_spike_processing(
        self,
        v_mem_full: "torch.tensor",
        output_spikes: "torch.tensor",
        n_batches: int,
        n_neurons: Tuple[int],
    ) -> "torch.tensor":
        """
        Handle post-processing after spikes have been generated: Reshape output spikes,
        reshape and record v_mem_full, record output spike statistics. Update neuron states.

        Parameters
        ----------
        v_mem_full : torch.tensor
            Membrane potential. Expected shape: (batches x neurons, time)
        output_spikes : torch.tensor
            Output spikes. Expected shape: (batches x neurons, time)
        n_batches : int
            Number of batches. Needed for correct reshaping of output and v_mem_full
        n_neurons : Tuple of ints
            Remaining neuron dimensions (e.g. channels, height, width). Determines output
            shape after batch and time dimensions.

        Returns
        -------
        torch.tensor
            Output spikes. Shape: (n_batches, time, *n_neurons)
        """

        # Separate batch and neuron dimensions -> (batches, *neurons, time)
        v_mem_full = v_mem_full.reshape(n_batches, *n_neurons, -1)
        output_spikes = output_spikes.reshape(n_batches, *n_neurons, -1)

        # Move time dimension to second -> (batches, time, *neurons)
        v_mem_full = v_mem_full.movedim(-1, 1)
        output_spikes = output_spikes.movedim(-1, 1)

        if self.record:
            # Record states and output
            self.v_mem_recorded = v_mem_full
            self.spikes_out = output_spikes

        self._update_neuron_states(v_mem_full, output_spikes)

        return output_spikes

    def forward(self, spike_input: "torch.tensor") -> "torch.tensor":
        """
        Generate membrane potential and resulting output spike train based on
        spiking input. Membrane potential will be stored as `self.v_mem_recorded`.

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

        if not hasattr(self, "v_mem") or self.v_mem.shape != (n_batches, *n_neurons):
            self.reset_states(shape=(n_batches, *n_neurons), randomize=False)

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        spike_input = spike_input.movedim(1, -1)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.reshape(-1, num_timesteps)
        # -> (n_parallel, num_timesteps)

        output_spikes, v_mem_full = self.spike_function(spike_input.contiguous())

        # Reshape output spikes and v_mem_full, store neuron states
        output_spikes = self._post_spike_processing(
            v_mem_full, output_spikes, n_batches, n_neurons
        )

        return output_spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            scale_grads=self.scale_grads,
            window=self.learning_window / self.threshold,
            alpha_mem=self.alpha_mem,
            record=self.record,
        )

        return param_dict
