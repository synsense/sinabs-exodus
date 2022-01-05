from typing import Callable, Optional, Tuple

import torch
from sinabs import activation

from sinabs.slayer.spike import SpikeFunctionIterForward
from sinabs.layers import StatefulLayer
from sinabs.activation import ActivationFunction, MultiSpike, MembraneSubtract, Heaviside


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
        record: bool = False,
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
        record: bool
            Record membrane potential and spike output during forward call. Default is False.
        """

        if activation_fn.spike_fn != MultiSpike \
             and not isinstance(activation_fn.reset_fn, MembraneSubtract) \
                 and not isinstance(activation_fn.surrogate_grad_fn, Heaviside):
            raise NotImplementedError("Spike mechanism config not supported. Use MultiSpike, MembraneSubtract and Heaviside surrogate grad functions.")
        
        if not (0 < alpha_mem <= 1.0):
            raise ValueError("`alpha_mem` must be between 0 and 1.")

        super().__init__(
            state_names=['v_mem']
        )

        self.threshold = activation_fn.spike_threshold
        self.threshold_low = threshold_low
        self.membrane_subtract = activation_fn.spike_threshold
        self.scale_grads = 1.
        self.learning_window = activation_fn.surrogate_grad_fn.window * activation_fn.spike_threshold
        self.alpha_mem = alpha_mem
        self.record = record

        # - Make sure buffers and parameters are on cuda
        self.cuda()

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
        if not self.is_state_initialised() or not self.state_has_shape((n_batches, *n_neurons)):
            self.init_state_with_shape((n_batches, *n_neurons))
            self.activations = torch.zeros((n_batches, *n_neurons), device=spike_input.device)

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        spike_input = spike_input.movedim(1, -1)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.reshape(-1, num_timesteps)
        # -> (n_parallel, num_timesteps)

        output_spikes, v_mem_full = SpikeFunctionIterForward.apply(
            spike_input.contiguous(),
            self.membrane_subtract,
            self.alpha_mem,
            self.v_mem.flatten(),
            self.activations.flatten(),
            self.threshold,
            self.threshold_low,
            self.learning_window,
            self.scale_grads,
        )

        # Reshape output spikes and v_mem_full, store neuron states
        v_mem_full = v_mem_full.reshape(n_batches, *n_neurons, -1)
        output_spikes = output_spikes.reshape(n_batches, *n_neurons, -1)

        v_mem_full = v_mem_full.movedim(-1, 1)
        output_spikes = output_spikes.movedim(-1, 1)

        if self.record:
            self.v_mem_recorded = v_mem_full
            self.spikes_out = output_spikes

        # update neuron states
        self.v_mem = v_mem_full[:, -1].clone()
        self.activations = output_spikes[:, -1].clone()
        self.tw = v_mem_full.shape[1]
        self.spikes_number = output_spikes.sum()

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
