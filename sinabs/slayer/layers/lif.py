import torch
import numpy as np
import torch.nn as nn
from typing import Optional, Union, List, Tuple
from sinabs.slayer.kernels import psp_kernels, exp_kernel
from sinabs.slayer.spike import spikeFunction

# - Type alias for array-like objects
from sinabs.slayer.psp import generateEpsp

ArrayLike = Union[np.ndarray, List, Tuple]

window = 1.0


class SpikingLayer(nn.Module):
    def __init__(
        self,
        tau_mem: float = 10.0,
        tau_syn: List[float] = [5.0],
        threshold: float = 1.0,
        tau_learning: float = 0.5,
        scale_grads: float = 1.0,
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
        """
        super().__init__()
        # Initialize neuron states
        self.threshold = threshold
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        epsp_kernel = psp_kernels(tau_mem=tau_mem, tau_syn=tau_syn, dt=1.0)
        ref_kernel = exp_kernel(tau_mem, dt=1.0) * threshold
        self.tau_learning = tau_learning
        self.scale_grads = scale_grads

        # Blank parameter place holders
        self.register_buffer("epsp_kernel", epsp_kernel)
        self.register_buffer("ref_kernel", ref_kernel)
        self.spikes_number = None
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

        # expected input dimension: (t_sim, n_batches, n_syn, n_channels)
        binary_input = binary_input.movedim(
            0, -1
        )  # move t_sim to last dimension (n_batches, n_syn, n_channels, t_sim)
        binary_input = binary_input.movedim(
            1, 0
        )  # move n_syn to first dimension (n_syn, n_batches, n_channels, t_sim)
        binary_input = binary_input.unsqueeze(1).unsqueeze(
            1
        )  # unsqueeze twice at dim 1 (n_syn, 1, 1, n_batches, n_channels, t_sim)

        # Compute the synaptic current
        syn_out: torch.Tensor = self.synaptic_output(binary_input)
        t_sim = syn_out.shape[-1]  # Last dimension is time
        vsyn = generateEpsp(syn_out, self.epsp_kernel)
        vmem = vsyn.sum(0)

        assert len(vmem.shape) == 5
        all_spikes = spikeFunction(
            vmem, -self.ref_kernel, self.threshold, self.tau_learning, self.scale_grads
        )

        self.vmem = vmem
        self.tw = t_sim

        all_spikes = (
            all_spikes.squeeze(0).squeeze(0).movedim(-1, 0)
        )  # move time back to front (t_sim, n_batches, n_channels)

        self.spikes_number = all_spikes.sum()
        self.n_spikes_out = all_spikes
        return all_spikes

    def __deepcopy__(self, memo=None):
        raise NotImplementedError()
