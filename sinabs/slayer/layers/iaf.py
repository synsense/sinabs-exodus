from typing import Optional

from sinabs.layers.pack_dims import squeeze_class
from sinabs.slayer.layers import IntegrateFireBase


__all__ = ["IAF", "IAFSqueeze"]


class IAF(IntegrateFireBase):
    def __init__(
        self,
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
        Slayer implementation of a spiking, non-leaky, IAF neuron with learning enabled.

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
        scale_grads: float
            Scale surrogate gradients in backpropagation.
        record: bool
            Record membrane potential and spike output during forward call.
        membrane_reset: bool
            Currently not supported.
        """

        super().__init__(
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            window=window,
            scale_grads=scale_grads,
            alpha=1.0,
            membrane_reset=membrane_reset,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict()
        param_dict.pop("alpha")

        return param_dict


# Class to accept data with batch and time dimensions combined
IAFSqueeze = squeeze_class(IAF)
