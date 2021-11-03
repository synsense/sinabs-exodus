from typing import Optional

import torch

from sinabs.layers.pack_dims import squeeze_class
from sinabs.slayer.layers import IntegrateFireBase

__all__ = ["LIF", "LIFSqueeze"]


class LIF(IntegrateFireBase):
    def __init__(
        self,
        threshold: float = 1.0,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        window: float = 1.0,
        scale_grads: float = 1.0,
        tau_mem: float = 10.0,
        dt: float = 1.0,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Slayer implementation of a spiking, LIF neuron with learning enabled.
        Does not simulate synaptic dynamics.

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
        tau_mem: float
            Membrane time constant.
        dt: float
            Simulation time step. Only used for calculating decay factor.
        scale_grads: float
            Scale surrogate gradients in backpropagation.
        membrane_reset: bool
            Currently not supported.
        """

        if dt <= 0:
            raise ValueError("`dt` must be greater than 0.")
        if tau_mem < dt:
            raise ValueError("`tau_mem` must be greater than `dt`.")

        self.dt = dt
        self.tau_mem = tau_mem

        super().__init__(
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            window=window,
            scale_grads=scale_grads,
            alpha=torch.exp(-dt / tau_mem).item(),
            membrane_reset=membrane_reset,
        )

    @property
    def tau_mem(self):
        return -self.dt / torch.log(self.alpha).item()

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict()
        param_dict.update(tau_mem=self.tau_mem, dt=self.dt)


# Class to accept data with batch and time dimensions combined
LIFSqueeze = squeeze_class(LIF)
