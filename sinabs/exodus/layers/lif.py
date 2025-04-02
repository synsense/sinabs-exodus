from typing import Callable, Optional, Tuple, Union

import torch
from sinabs.layers import SqueezeMixin
from sinabs.layers import LIF as LIFSinabs
from sinabs.activation import (
    MultiSpike,
    SingleSpike,
    MaxSpike,
    MembraneSubtract,
    SingleExponential,
)

from sinabs.exodus.leaky import LeakyIntegrator
from sinabs.exodus.spike import IntegrateAndFire

__all__ = ["LIF", "LIFSqueeze"]


def expand_to_1d_contiguous(
    value: Union[float, torch.Tensor], shape: Tuple[int]
) -> torch.Tensor:
    """
    Expand tensor to tensor of given shape.
    Then flatten and make contiguous.

    Not flattening immediately ensures that non-scalar tensors are
    broadcast correctly to the required shape.

    Parameters
    ----------
    value: float or torch
        Tensor to be expanded
    shape: tuple of ints
        shape to expand to

    Returns
    -------
    torch.Tensor
        Contiguous 1D-tensor from expanded `value`
    """

    expanded_tensor = torch.as_tensor(value).expand(shape)
    return expanded_tensor.float().flatten().contiguous()


class LIF(LIFSinabs):
    """
    Exodus implementation of a Leaky Integrate and Fire neuron layer.

    Neuron dynamics in discrete time:

    .. math ::
        V_{mem}(t+1) = \\alpha V_{mem}(t) + (1-\\alpha)\\sum z(t)

        \\text{if } V_{mem}(t) >= V_{th} \\text{, then } V_{mem} \\rightarrow V_{reset}

    where :math:`\\alpha =  e^{-1/tau_{mem}}` and :math:`\\sum z(t)` represents the sum of all input currents at time :math:`t`.

    Parameters
    ----------
    tau_mem: float
        Membrane potential time constant.
    tau_syn: float
        Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
    spike_threshold: float
        Spikes are emitted if v_mem is above that threshold. By default set to 1.0.
    spike_fn: torch.autograd.Function
        Choose a Sinabs or custom torch.autograd.Function that takes a dict of states,
        a spike threshold and a surrogate gradient function and returns spikes. Be aware
        that the class itself is passed here (because torch.autograd methods are static)
        rather than an object instance.
    reset_fn: Callable
        A function that defines how the membrane potential is reset after a spike.
    surrogate_grad_fn: Callable
        Choose how to define gradients for the spiking non-linearity during the
        backward pass. This is a function of membrane potential.
    min_v_mem: float or None
        Lower bound for membrane potential v_mem, clipped at every time step.
    train_alphas: bool
        When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself.
    shape: torch.Size
        Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
    norm_input: bool
        When True, normalise input current by tau. This helps when training time constants.
    record_states: bool
        When True, will record all internal states such as v_mem or i_syn in a dictionary attribute `recordings`. Default is False.
    decay_early: bool
        When True, exponential decay is applied to synaptic input already in the same
        time step, i.e. an input pulse of 1 will result in a synaptic current of
        alpha, rather than one. This only holds for synaptic currents. Membrane
        potential will be decayed only in next time step, irrespective of how
        `decay_early` is set. This is the same behavior as in sinabs. Default: True.
    """

    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        spike_threshold: Optional[Union[float, torch.Tensor]] = 1.0,
        spike_fn: Callable = MultiSpike,
        reset_fn: Callable = MembraneSubtract(),
        surrogate_grad_fn: Callable = SingleExponential(),
        min_v_mem: Optional[Union[float, torch.Tensor]] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        norm_input: bool = True,
        record_states: bool = False,
        decay_early: bool = True,
    ):
        # Make sure activation functions match exodus specifications
        self._parse_activation_fn(spike_fn, reset_fn)

        super().__init__(
            tau_mem=tau_mem,
            tau_syn=tau_syn,
            spike_threshold=spike_threshold,
            spike_fn=spike_fn,
            reset_fn=reset_fn,
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            train_alphas=train_alphas,
            shape=shape,
            norm_input=norm_input,
            record_states=record_states,
        )

        self.decay_early = decay_early

    def _parse_activation_fn(self, spike_fn, reset_fn):

        if spike_fn is None:
            # Non-spiking neurons
            return

        if (
            spike_fn not in (MultiSpike, SingleSpike)
            and not isinstance(spike_fn, MaxSpike)
        ) or not isinstance(reset_fn, MembraneSubtract):
            raise NotImplementedError(
                "Spike mechanism config not supported. "
                "Use MultiSpike/SingleSpike/MaxSpike and MembraneSubtract functions."
            )

        if isinstance(spike_fn, MaxSpike):
            self.max_num_spikes_per_bin = spike_fn.max_num_spikes_per_bin
        elif spike_fn == MultiSpike:
            self.max_num_spikes_per_bin = None
        else:
            self.max_num_spikes_per_bin = 1

    def _prepare_input(self, input_data: torch.Tensor):
        """
        Make sure neuron states are initialized in correct shape.
        Reshape input to 2D

        Parameters
        ----------
        input_data: torch.tensor
            Input of shape (batch_size, time_steps, *trailing_dim)

        Returns
        -------
        torch.tensor
            Input, reshaped to (N, time_steps), where N is the product of
            batch_size and all trailing dimensions.
        tuple of int
            Original shape of input
        """

        batch_size, time_steps, *trailing_dim = input_data.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        # Move time to last dimension -> (n_batches, *trailing_dim, time_steps)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        input_2d = input_data.movedim(1, -1).reshape(-1, time_steps)

        return input_2d, (batch_size, time_steps, *trailing_dim)

    def _forward_synaptic(self, input_2d: torch.Tensor):
        """Evolve synaptic dynamics"""

        alpha_syn = expand_to_1d_contiguous(self.alpha_syn_calculated, self.v_mem.shape)

        if self.decay_early:
            input_2d = input_2d * alpha_syn.unsqueeze(1)

        # Apply exponential filter to input
        return LeakyIntegrator.apply(
            input_2d.contiguous(),  # Input data
            alpha_syn,  # Synaptic alpha
            self.i_syn.flatten().contiguous(),  # Initial synaptic states
        )

    def _forward_membrane(self, i_syn_2d: torch.Tensor):
        """Evolve membrane dynamics"""

        # Broadcast alpha to number of neurons (x batches)
        alpha_mem = expand_to_1d_contiguous(self.alpha_mem_calculated, self.v_mem.shape)

        if self.norm_input:
            # Rescale input with 1 - alpha (based on approximation that
            # alpha = exp(-1/tau) ~ 1 / (1 - tau) for tau >> 1)
            i_syn_2d = (1.0 - alpha_mem.unsqueeze(1)) * i_syn_2d

        if self.spike_fn is None:

            if self.decay_early:
                i_syn_2d = i_syn_2d * alpha_mem.unsqueeze(1)

            # - Non-spiking case (leaky integrator)
            v_mem = LeakyIntegrator.apply(
                i_syn_2d,  # Input data
                alpha_mem,  # Membrane alpha
                self.v_mem.flatten().contiguous(),  # Initial vmem
            )

            return v_mem, v_mem

        # Expand spike threshold
        spike_threshold = expand_to_1d_contiguous(
            self.spike_threshold, self.v_mem.shape
        )

        # Expand min_v_mem
        if self.min_v_mem is None:
            min_v_mem = None
        else:
            min_v_mem = expand_to_1d_contiguous(self.min_v_mem, self.v_mem.shape)

        # Expand membrane subtract
        membrane_subtract = self.reset_fn.subtract_value
        if membrane_subtract is None:
            membrane_subtract = spike_threshold
        else:
            membrane_subtract = expand_to_1d_contiguous(
                membrane_subtract, self.v_mem.shape
            )

        output_2d, v_mem_2d = IntegrateAndFire.apply(
            i_syn_2d.contiguous(),  # Input data
            alpha_mem,  # Alphas
            self.v_mem.flatten().contiguous(),  # Initial vmem
            spike_threshold,  # Spike threshold
            membrane_subtract,  # Membrane subtract
            min_v_mem,  # Lower bound on vmem
            self.surrogate_grad_fn,  # Surrogate gradient
            self.max_num_spikes_per_bin,  # Max. number of spikes per bin
        )
        # Apply reset to membrne potential
        v_mem_2d = v_mem_2d - membrane_subtract.unsqueeze(1) * output_2d

        return output_2d, v_mem_2d

    def forward(self, input_data: torch.Tensor):
        """
        Forward pass with given data.

        Parameters:
            input_current : torch.Tensor
                Data to be processed. Expected shape: (batch, time, ...)

        Returns:
            torch.Tensor
                Output data. Same shape as `input_data`.
        """

        input_2d, original_shape = self._prepare_input(input_data)
        batch_size, time_steps, *trailing_dim = original_shape

        self.recordings = dict()

        # - Synaptic dynamics
        if self.tau_syn_calculated is None:
            i_syn_2d = input_2d
        else:
            i_syn_2d = self._forward_synaptic(input_2d)

            # Bring i_syn to shape that matches input
            i_syn_full = i_syn_2d.reshape(batch_size, *trailing_dim, -1).movedim(-1, 1)

            # Update internal i_syn
            self.i_syn = i_syn_full[:, -1].clone()
            if self.record_states:
                self.recordings["i_syn"] = i_syn_full

        # - Membrane dynamics
        output_2d, v_mem_2d = self._forward_membrane(i_syn_2d)

        # Reshape output spikes and v_mem_full, store neuron states
        v_mem_full = v_mem_2d.reshape(batch_size, *trailing_dim, -1).movedim(-1, 1)
        output_full = output_2d.reshape(batch_size, *trailing_dim, -1).movedim(-1, 1)

        if self.record_states:
            self.recordings["v_mem"] = v_mem_full

        # update neuron states
        self.v_mem = v_mem_full[:, -1].clone()

        self.firing_rate = output_full.sum() / output_full.numel()

        return output_full

    def __repr__(self):
        return "EXODUS " + super().__repr__()


class LIFSqueeze(LIF, SqueezeMixin):
    """
    Same as parent class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width)
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers.
    """

    def __init__(self, batch_size=None, num_timesteps=None, **kwargs):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)

    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
