import torch
import numpy as np
import torch.nn as nn
from typing import Union, List, Tuple
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

        Parameters
        ----------
        input_spikes: torch.Tensor
            Input to the layer. Shape: (synapses, batches x neurons, time)

        Returns
        -------
            torch.Tensor
            Synaptic output current. Same shape as 'input_spikes'
        """
        return input_spikes

    def forward(self, spike_input: torch.Tensor) -> torch.Tensor:
        """
        Generate membrane potential and resulting output spike train based on
        spiking inputs. Membrane potential will be stored as `self.vmem`.

        Parameters
        ----------
        spike_input: torch.tensor
            Spike input raster. May take non-binary integer values.
            Expected shape: (time, batches, synapses, neurons)
            Third dimension corresponds to inputs for different synaptic time
            constants.

        Returns
        -------
        torch.tensor
            Output spikes. Same shape as `spike_input`.
        """

        if spike_input.ndim != 4:
            raise ValueError(
                "'spike_input` must be of shape (time, batches, synapses, neurons"
            )
        # Change dimension order to (synapses, batches, neurons, time)
        spike_input = spike_input.permute(2, 1, 3, 0)
        # Combine batches and synapses (can be computed in parallel)
        # -> (synapses, batches x neurons, time)
        n_syn, n_batches, n_neurons, t_sim = spike_input.shape
        spike_input = spike_input.reshape(n_syn, -1, t_sim)

        # Apply synapse function
        syn_out = self.synaptic_output(spike_input).contiguous()

        assert syn_out.ndim == 3
        assert self.epsp_kernel.ndim == 2
        assert syn_out.is_contiguous()

        # Membrane potential from individual synaptic time constants
        vmem_syn = generateEpsp(syn_out, self.epsp_kernel)
        # Joint membrane potential -> (batches x neurons, time)
        vmem = vmem_syn.sum(0)

        assert vmem.ndim == 2
        assert self.ref_kernel.ndim == 1
        assert vmem.is_contiguous()

        output_spikes = spikeFunction(
            vmem, -self.ref_kernel, self.threshold, self.tau_learning, self.scale_grads
        )

        # Separate batch and neuron dimensions -> (batches, neurons, time)
        vmem = vmem.reshape(n_batches, n_neurons, t_sim)
        output_spikes = output_spikes.reshape(n_batches, n_neurons, t_sim)

        # # Join batch and time dimensions -> (batches x time, neurons)
        # vmem = vmem.movedim(-1, 1).reshape(-1, n_neurons)
        # output_spikes = output_spikes.movedim(-1, 1).reshape(-1, n_neurons)

        vmem = vmem.movedim(-1, 0)
        output_spikes = output_spikes.movedim(-1, 0)

        self.vmem = vmem
        self.tw = t_sim

        self.spikes_number = output_spikes.sum()
        self.n_spikes_out = output_spikes

        return output_spikes

    def __deepcopy__(self, memo=None):
        raise NotImplementedError()
