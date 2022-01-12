import torch 
from typing import Callable, Optional
from sinabs.slayer.layers import IntegrateFireBase
from sinabs.layers import SqueezeMixin
from sinabs.activation import ActivationFunction


__all__ = ["IAF", "IAFSqueeze"]


class IAF(IntegrateFireBase):
    def __init__(
        self,
        activation_fn: Callable = ActivationFunction(),
        threshold_low: Optional[float] = None,
        shape: Optional[torch.Size] = None,
        record_v_mem: bool = False,
    ):
        """
        Slayer implementation of a spiking, non-leaky, IAF neuron with learning enabled.

        Parameters
        ----------
        activation_fn: Callable
            a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        record_v_mem: bool
            Record membrane potential and spike output during forward call. Default is False.
        """

        super().__init__(
            alpha_mem=1.0,
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            shape=shape,
            record_v_mem=record_v_mem,
        )

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict
        param_dict.pop("alpha_mem")

        return param_dict


class IAFSqueeze(IAF, SqueezeMixin):
    """
    Same as parent class, only takes in squeezed 4D input (Batch*Time, Channel, Height, Width) 
    instead of 5D input (Batch, Time, Channel, Height, Width) in order to be compatible with
    layers that can only take a 4D input, such as convolutional and pooling layers. 
    """
    def __init__(self,
                 batch_size = None,
                 num_timesteps = None,
                 **kwargs,
                ):
        super().__init__(**kwargs)
        self.squeeze_init(batch_size, num_timesteps)
    
    def forward(self, input_data: torch.Tensor) -> torch.Tensor:
        return self.squeeze_forward(input_data, super().forward)

    @property
    def _param_dict(self) -> dict:
        return self.squeeze_param_dict(super()._param_dict)
