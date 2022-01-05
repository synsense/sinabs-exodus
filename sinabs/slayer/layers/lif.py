import torch
import sinabs.activation as sina
from sinabs.slayer.layers import IntegrateFireBase
from sinabs.layers import SqueezeMixin
from typing import Callable, Optional, Union


__all__ = ["LIF", "LIFSqueeze"]


class LIF(IntegrateFireBase):
    def __init__(
        self,
        tau_mem: Union[float, torch.Tensor],
        tau_syn: Optional[Union[float, torch.Tensor]] = None,
        activation_fn: Callable = sina.ActivationFunction(),
        threshold_low: Optional[float] = None,
        train_alphas: bool = False,
        shape: Optional[torch.Size] = None,
        record: bool = True,
    ):
        """
        Slayer implementation of a spiking, LIF neuron with learning enabled.
        Does not simulate synaptic dynamics.

        Parameters
        ----------
        tau_mem: float
            Membrane potential time constant.
        tau_syn: float
            Synaptic decay time constants. If None, no synaptic dynamics are used, which is the default.
        activation_fn: Callable
            a sinabs.activation.ActivationFunction to provide spiking and reset mechanism. Also defines a surrogate gradient.
        threshold_low: float or None
            Lower bound for membrane potential v_mem, clipped at every time step.
        train_alphas: bool
            When True, the discrete decay factor exp(-1/tau) is used for training rather than tau itself. 
        shape: torch.Size
            Optionally initialise the layer state with given shape. If None, will be inferred from input_size.
        record: bool
            Record membrane potential and spike output during forward call.
        """

        super().__init__(
            alpha_mem=torch.exp(-torch.tensor(1.0 / tau_mem)).item(),
            activation_fn=activation_fn,
            threshold_low=threshold_low,
            shape=shape,
            record=record,
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


class LIFSqueeze(LIF, SqueezeMixin):
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
