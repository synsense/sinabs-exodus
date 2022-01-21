from typing import Callable, Optional
from copy import deepcopy
import torch
from sinabs.slayer.spike import SpikeFunctionIterForward
from sinabs.layers import StatefulLayer
from sinabs.activation import (
    ActivationFunction,
    MultiSpike,
    SingleSpike,
    MembraneSubtract,
)


__all__ = ["IntegrateFireBase"]


class IntegrateFireBase(StatefulLayer):
    """
    Slayer implementation of a leaky or non-leaky integrate and fire neuron with
    learning enabled. Does not simulate synaptic dynamics.
    """

    backend = "slayer"

    def __init__(
        self,
        alpha_mem: float = 1.0,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        record_v_mem: bool = False,
    ):
        """
        Slayer implementation of a leaky or non-leaky integrate and fire neuron with
        learning enabled. Does not simulate synaptic dynamics.

        Parameters
        ----------
        alpha_mem: float
            Neuron state is multiplied by this factor at each timestep. For IAF dynamics
            set to 1, for LIF to exp(-dt/tau).
        activation_fn: Callable
            a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        record_v_mem: bool
            Record membrane potential and spike output during forward call. Default is False.
        """

        if activation_fn.spike_fn not in (MultiSpike, SingleSpike) or not isinstance(
            activation_fn.reset_fn, MembraneSubtract
        ):
            raise NotImplementedError(
                "Spike mechanism config not supported. Use MultiSpike/SingleSpike and MembraneSubtract functions."
            )

        if not (0 < alpha_mem <= 1.0):
            raise ValueError("`alpha_mem` must be between 0 and 1.")

        super().__init__(state_names=["v_mem", "activations"])

        self.threshold = activation_fn.spike_threshold
        self.threshold_low = threshold_low
        self.membrane_subtract = activation_fn.spike_threshold
        self.surrogate_grad_fn = activation_fn.surrogate_grad_fn
        self.alpha_mem = alpha_mem
        self.record_v_mem = record_v_mem
        if shape:
            self.init_state_with_shape(shape)

        self.multiple_spikes = activation_fn.spike_fn == MultiSpike

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

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (n_batches, *n_neurons)
        ):
            self.init_state_with_shape((n_batches, *n_neurons))

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.movedim(1, -1).reshape(-1, num_timesteps)

        output_spikes, v_mem_full = SpikeFunctionIterForward.apply(
            spike_input.contiguous(),
            self.membrane_subtract,
            self.alpha_mem,
            self.v_mem.flatten(),
            self.activations.flatten(),
            self.threshold,
            self.threshold_low,
            self.surrogate_grad_fn,
            self.multiple_spikes,
        )

        # Reshape output spikes and v_mem_full, store neuron states
        v_mem_full = v_mem_full.reshape(n_batches, *n_neurons, -1).movedim(-1, 1)
        output_spikes = output_spikes.reshape(n_batches, *n_neurons, -1).movedim(-1, 1)

        if self.record_v_mem:
            self.v_mem_recorded = v_mem_full

        # update neuron states
        self.v_mem = v_mem_full[:, -1].clone()

        return output_spikes

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.update(
            alpha_mem=self.alpha_mem,
            record=self.record,
            activation_fn=deepcopy(self.activation_fn),
        )

        return param_dict
