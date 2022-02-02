import torch
from sinabs.exodus.leaky import LeakyIntegrator
from sinabs.layers import StatefulLayer, SqueezeMixin
from typing import Union, Optional


__all__ = ["ExpLeak", "ExpLeakSqueeze"]


class ExpLeak(StatefulLayer):
    def __init__(
        self,
        tau_leak: Union[float, torch.Tensor],
        shape: Optional[torch.Size] = None,
        threshold_low: Optional[float] = None,
        norm_input: bool = True,
        decay_early: bool = False,
    ):
        """
        Exodus implementation of an integrator with exponential leak.

        Parameters
        ----------
        tau_leak: float
            Rate of leak of the state
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        norm_input: bool
            If True, will normalise the inputs by tau_mem. This helps when training time constants.
        decay_early: bool
            If True, will scale inputs by exp(-1/tau). This corresponds to the Xylo-behavior of
            decaying the input within the same time step.
        """
        super().__init__(state_names=["v_mem"])
        self.tau_leak = torch.as_tensor(tau_leak, dtype=float)
        self.alpha_leak = torch.exp(-1 / self.tau_leak)
        self.threshold_low = threshold_low
        self.norm_input = norm_input
        self.decay_early = decay_early
        if shape:
            self.init_state_with_shape(shape)

    def forward(self, input_current: "torch.tensor") -> "torch.tensor":
        """
        Forward pass.

        Parameters
        ----------
        input_current : torch.tensor
            Expected shape: (Batch, Time, *...)

        Returns
        -------
        torch.tensor
            ExpLeak layer states. Same shape as `input_current`
        """

        batch_size, time_steps, *trailing_dim = input_current.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (batch_size, *trailing_dim)
        ):
            self.init_state_with_shape((batch_size, *trailing_dim))

        # Reshape input to 2D -> (N, Time)
        input_2d = input_current.movedim(1, -1).flatten(end_dim=-2).contiguous()

        if self.norm_input:
            # Rescale input with 1 - alpha
            input_2d = (1.0 - self.alpha_leak) * input_2d

        if self.decay_early:
            # Rescale input with alpha
            input_2d = self.alpha_leak * input_2d

        # Actual evolution of states
        states = LeakyIntegrator.apply(
            input_2d, self.v_mem.flatten().contiguous(), self.alpha_leak
        )

        # Reshape states to original shape -> (Batch, Time, ...)
        states = states.reshape(*(batch_size, *trailing_dim), time_steps).movedim(-1, 1)

        # Store current state based on last evolution time step
        self.v_mem = states[:, -1, ...].clone()

        self.tw = time_steps

        return states


class ExpLeakSqueeze(ExpLeak, SqueezeMixin):
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
