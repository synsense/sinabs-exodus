import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from src.sinabs2.slayer.kernels import psp_kernels, exp_kernel
from src.sinabs2.slayer.spike import spikeFunction

# - Type alias for array-like objects
from src.sinabs2.slayer.psp import generateEpsp

ArrayLike = Union[np.ndarray, List, Tuple]

window = 1.0


class SpikingLayer(nn.Module):
    def __init__(
            self,
            tau_mem: float = 10.0,
            tau_syn: List[float] = [5.0, ],
            threshold: float = 1.0,
            batch_size: Optional[int] = None,
    ):
        """
        Pytorch implementation of a spiking neuron with learning enabled.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        Parameters:
        -----------
        tau_mem:
            Membrane time constant
        tau_syn:
            Synaptic time constant
        n_syn:
            Number of synapses per neuron
        threshold:
            Spiking threshold of the neuron.

        batch_size:
            The batch size. Needed to distinguish between timesteps and batch dimension.
        """
        super().__init__()
        # Initialize neuron states
        self.threshold = threshold
        epsp_kernel = psp_kernels(tau_mem=tau_mem, tau_syn=tau_syn, dt=1.0)
        ref_kernel = exp_kernel(tau_mem, dt=1.0) * threshold

        # Blank parameter place holders
        self.register_buffer("epsp_kernel", epsp_kernel)
        self.register_buffer("ref_kernel", ref_kernel)
        self.spikes_number = None
        if batch_size is None:
            batch_size = 1
        self.batch_size = batch_size
        self.n_syn = len(tau_syn)

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method needs to be overridden/defined by the child class
        Default implementation is pass through

        :param input_spikes: torch.Tensor input to the layer.
        :return:  torch.Tensor - synaptic output current
        """
        return input_spikes

    def forward(self, binary_input: torch.Tensor):

        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)
        t_sim = syn_out.shape[-1]  # Last dimension is time
        vsyn = generateEpsp(syn_out, self.epsp_kernel, t_sim)
        vmem = vsyn.sum(0).clone()

        tauRho = 1.0
        scaleRho = 1.0
        all_spikes = spikeFunction(vmem, -self.ref_kernel, self.threshold, tauRho, scaleRho)

        self.vmem = vmem
        self.tw = t_sim
        self.activations = all_spikes
        self.spikes_number = all_spikes.sum()

        return all_spikes

    def __deepcopy__(self, memo=None):
        raise NotImplementedError()
