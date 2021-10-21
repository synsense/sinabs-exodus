from warnings import warn
from typing import Optional, List

import torch

from sinabs.slayer.kernels import psp_kernels, exp_kernel
from sinabs.slayer.psp import generateEpsp
from sinabs.layers.pack_dims import squeeze_class
from sinabs.slayer.layers import SpikingLayer


class LIF(SpikingLayer):
    def __init__(
        self,
        tau_mem: float = 10.0,
        tau_syn: List[float] = [5.0],
        threshold: float = 1.0,
        threshold_low: Optional[float] = None,
        membrane_subtract: Optional[float] = None,
        window: float = 1.0,
        scale_grads: float = 1.0,
        membrane_reset=False,
        *args,
        **kwargs,
    ):
        """
        Pytorch implementation of a spiking neuron with learning enabled.
        This class is the base class for any layer that need to implement integrate-and-fire operations.

        Parameters:
        -----------
        tau_mem : float
            Membrane time constant
        tau_syn : float
            Synaptic time constant
        threshold : float
            Spiking threshold of the neuron.
        threshold_low: Optional[float]
            Lower bound for membrane potential.
        membrane_subtract : Optional[float]
            Constant to be subtracted from membrane potential when neuron spikes.
            If ``None`` (default): Same as ``threshold``.
        window: float
            Distance between step of Heaviside surrogate gradient and threshold.
        scale_grads : float
            Scale surrogate gradients in backpropagation.
        membrane_reset : bool
            Currently not supported.
        """

        if membrane_reset:
            raise NotImplementedError("Membrane reset not implemented for this layer.")

        if threshold_low is not None:
            warn(
                "Using a lower threshold with this layer results in slower backward computation."
            )

        warn("This layer currently does not store states between forward calls.")

        super().__init__(
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            window=window,
            scale_grads=scale_grads,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )

        # - Store hyperparameters
        self.tau_mem = tau_mem
        self.tau_syn = tau_syn
        self.n_syn = len(tau_syn)

        # - Initialize kernel
        epsp_kernel = psp_kernels(tau_mem=tau_mem, tau_syn=tau_syn, dt=1.0)

        self.register_buffer("epsp_kernel", epsp_kernel)

    def synaptic_output(self, input_spikes: torch.Tensor) -> torch.Tensor:
        """
        This method needs to be overridden/defined by the child class
        Default implementation is pass through

        Parameters
        ----------
        input_spikes: torch.Tensor
            Input to the layer. Shape: (batches, time, synapses, ...)

        Returns
        -------
            torch.Tensor
            Synaptic output current. Same shape as 'input_spikes'
        """
        return input_spikes

    def forward(self, spike_input: "torch.Tensor") -> "torch.Tensor":
        """
        Generate membrane potential and resulting output spike train based on
        spiking inputs. Membrane potential will be stored as `self.vmem`.

        Parameters
        ----------
        spike_input: torch.tensor
            Spike input raster. May take non-binary integer values.
            Expected shape: (batches, time, synapses, *neurons)
            Third dimension corresponds to inputs for different synaptic time
            constants. neurons can be any tuple of integers, corresponding to
            different neuron dimensions, such as channels, height, width, ...

        Returns
        -------
        torch.tensor
            Output spikes. Shape: (batches, time, *neurons).
        """

        n_batches, num_timesteps, n_syn, *n_neurons = spike_input.shape

        # Make sure synapse dimensions match
        if n_syn != self.n_syn:
            raise ValueError(
                f"Synapse (3nd) dimension of `spike_input` must be {self.n_syn}"
            )

        # Apply synapse function
        syn_out = self.synaptic_output(spike_input)

        # Move synapse dimension to front -> (synapses, batch, time, *neurons)
        syn_out = syn_out.movedim(2, 0)
        # Move time dimension to back -> (synapses, batch, *neurons, time)
        syn_out = syn_out.movedim(2, -1)

        # Combine batch and all neuron dimensions (can be computed in parallel)
        # -> (synapses, batches x neurons, time)
        syn_out = syn_out.reshape(n_syn, num_timesteps, -1).contiguous()

        assert syn_out.ndim == 3
        assert self.epsp_kernel.ndim == 2

        # Membrane potential from individual synaptic time constants
        vmem_syn = generateEpsp(syn_out, self.epsp_kernel)
        # Joint membrane potential -> (time, batches x neurons)
        vmem = vmem_syn.sum(0).contiguous()

        # Generate spikes
        output_spikes = self.spike_function(vmem)

        # Post-process and return
        return self._post_spike_processing(vmem, output_spikes, n_batches, n_neurons)

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict()
        param_dict.update(tau_mem=self.tau_mem, tau_syn=self.tau_syn)


# Class to accept data with batch and time dimensions combined
LIFSqueeze = squeeze_class(LIF)
