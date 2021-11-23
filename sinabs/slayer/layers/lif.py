from typing import Optional

import torch

from sinabs.layers.pack_dims import squeeze_class
from sinabs.slayer.layers import IntegrateFireBase

__all__ = ["LIF", "LIFSqueeze"]


class LIF(IntegrateFireBase):
    def __init__(
        self,
        tau_mem: float,
        threshold: float = 1.0,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        window: float = 1.0,
        scale_grads: float = 1.0,
        record: bool = True,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Slayer implementation of a spiking, LIF neuron with learning enabled.
        Does not simulate synaptic dynamics.

        Parameters:
        -----------
        tau_mem: float
            Membrane time constant.
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
        record: bool
            Record membrane potential and spike output during forward call.
        membrane_reset: bool
            Currently not supported.
        """

        # if dt <= 0:
        #     raise ValueError("`dt` must be greater than 0.")
        # if tau_mem < dt:
        #     raise ValueError("`tau_mem` must be greater than `dt`.")

        # self.dt = dt
        # tau_mem = tau_mem

        super().__init__(
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            window=window,
            scale_grads=scale_grads,
            alpha_mem=torch.exp(-torch.tensor(1.0 / tau_mem)).item(),
            membrane_reset=membrane_reset,
        )

    @property
    def tau_mem(self):
        return -1.0 / torch.log(torch.tensor(self.alpha_mem)).item()

    @tau_mem.setter
    def tau_mem(self, new_tau):
        if new_tau <= 0:
            raise ValueError("'tau_mem' must be greater than 0.")
        self.alpha_mem = torch.exp(-torch.tensor(1.0 / new_tau)).item()

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("alpha_mem")
        param_dict.update(tau_mem=self.tau_mem)

        return param_dict

    def forward(self, inp):
        inp_rescaled = (1.0 - self.alpha_mem) * inp
        return super().forward(inp_rescaled)


# Class to accept data with batch and time dimensions combined
LIFSqueeze = squeeze_class(LIF)
