from typing import Optional

from torch import nn

from sinabs.slayer.kernels import heaviside_kernel
from sinabs.slayer.psp import generateEpsp
from sinabs.slayer.spike import spikeFunction


class SpikingLayer(nn.Module):
    def __init__(
        self,
        t_sim: int,
        threshold: float = 1.0,
        membrane_subtract: Optional[float] = None,
        tau_learning: float = 0.5,
        threshold_low=None,
    ):
        """
        Pytorch implementation of a spiking, non-leaky, IAF neuron with learning enabled.

        Parameters:
        -----------
        t_sim:
            Number of timesteps per sample.
        threshold:
            Spiking threshold of the neuron.
        membrane_subtract:
            Constant to be subtracted from membrane potential when neuron spikes.
            If ``None`` (default): Same as ``threshold``.
        """
        super().__init__()
        # Initialize neuron states
        self.threshold = threshold

        if threshold_low is not None:
            raise NotImplementedError("Lower threshold not implemented for this layer.")

        epsp_kernel = heaviside_kernel(size=t_sim, scale=1.0)

        if membrane_subtract is None:
            membrane_subtract = threshold
        ref_kernel = heaviside_kernel(size=t_sim, scale=membrane_subtract)

        self.tau_learning = tau_learning
        self.t_sim = t_sim

        # Blank parameter place holders
        self.register_buffer("epsp_kernel", epsp_kernel)
        self.register_buffer("ref_kernel", ref_kernel)
        self.spikes_number = None

    def forward(self, spike_input):

        if spike_input.ndim == 5:
            seperate_batch_time = True
            # expected input dimension: (n_batches, t_sim, *channel_shape )
            channel_shape = spike_input.shape[2:]

        else:
            seperate_batch_time = False
            # expected input dimension: (n_batches x t_sim, *channel_shape )
            channel_shape = spike_input.shape[1:]
            try:
                spike_input = spike_input.reshape(-1, self.t_sim, *channel_shape)
            except RuntimeError:
                raise ValueError(
                    f"First input dimension (time) must be multiple of {self.t_sim}"
                    + f" but is {spike_input.shape[0]}."
                )

        # Flatten out remaining dimensions -> (n_batches, t_sim, channels)
        spike_input = spike_input.reshape(*spike_input.shape[:2], -1)
        # move t_sim to last dimension -> (n_batches, n_channels, t_sim)
        spike_input = spike_input.movedim(1, -1)
        # unsqueeze at dim 0 -> (1, n_batches, n_channels, t_sim)
        spike_input = spike_input.unsqueeze(0)

        vmem = generateEpsp(spike_input, self.epsp_kernel)

        assert vmem.ndim == 5  # (1, 1, n_batches, n_channels, t_sim)
        all_spikes = spikeFunction(
            vmem, -self.ref_kernel, self.threshold, self.tau_learning
        )

        # move time back to front -> (n_batches, t_sim, n_channels)
        all_spikes = all_spikes.squeeze(0).squeeze(0).movedim(-1, 1)
        vmem = vmem.squeeze(0).squeeze(0).movedim(-1, 1)
        if seperate_batch_time:
            # expand channel dimensions -> (n_batches, t_sim, ...)
            all_spikes = all_spikes.reshape(*all_spikes.shape[:2], *channel_shape)
            vmem = vmem.reshape(*vmem.shape[:2], *channel_shape)
        else:
            # flatten time and batch dimensions -> (n_batches x t_sim, n_channels)
            all_spikes = all_spikes.reshape(-1, all_spikes.shape[-1])
            vmem = vmem.reshape(-1, vmem.shape[-1])
            # expand channel dimensions -> (n_batches x t_sim, ...)
            all_spikes = all_spikes.reshape(-1, *channel_shape)
            vmem = vmem.reshape(-1, *channel_shape)

        self.vmem = vmem
        self.tw = self.t_sim

        self.spikes_number = all_spikes.sum()

        return all_spikes
