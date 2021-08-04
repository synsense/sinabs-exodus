from typing import Optional, Tuple

import torch

from sinabs.slayer.spike import spikeFunction, spikeFunctionLB, spikeFunctionOldForward
from sinabs.layers import SpikingLayer as SpikingLayerBase


__all__ = ["SpikingLayer"]


class SpikingLayer(SpikingLayerBase):
    """
    Slayer implementation of a spiking neuron with learning enabled.
    This class is the base class for any layer that need to implement leaky or
    non-leaky integrate-and-fire operations with Slayer as backend.
    This is an abstract base class.
    """

    def __init__(
        self,
        num_timesteps: Optional[int] = None,
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
        Slayer implementation of a spiking neuron with learning enabled.
        This class is the base class for any layer that need to implement leaky or
        non-leaky integrate-and-fire operations with Slayer as backend.
        This is an abstract base class.

        Parameters:
        -----------
        num_timesteps: int
            Number of timesteps per sample.
        threshold: float
            Spiking threshold of the neuron.
        threshold_low: Optional[float]
            Lower bound for membrane potential.
        membrane_subtract: Optional[float]
            Constant to be subtracted from membrane potential when neuron spikes.
            If ``None`` (default): Same as ``threshold``.
        window: float
            Distance between step of Heaviside surrogate gradient and threshold.
            (Relative to size of threshold)
        scale_grads: float
            Scale surrogate gradients in backpropagation.
        membrane_reset: bool
            Currently not supported.
        """

        if membrane_reset:
            raise NotImplementedError("Membrane reset not implemented for this layer.")

        super().__init__(
            *args,
            **kwargs,
            threshold=threshold,
            threshold_low=threshold_low,
            membrane_subtract=membrane_subtract,
            membrane_reset=membrane_reset,
        )

        # - Store hyperparameters
        self.scale_grads = scale_grads
        self.window_abs = window * threshold
        self._num_timesteps = num_timesteps

    def spike_function(self, vmem: "torch.tensor") -> "torch.tensor":
        """
        Generate spikes from membrane potential.

        Parameters
        ----------
        vmem : torch.tensor
            Membrane potential. Expected shape: (batches x neurons, time)
            Has to be contiguous.

        Returns:
        --------
        torch.tensor
            Output spikes. Same shape as `vmem`
        """

        # Note: Do not make `vmem` contiguous only here because a new object will
        # be created, so that any modifications (membrane reset etc.) would not
        # have effect on the original `vmem`.

        # Generate output_spikes
        if True:  # self.threshold_low is not None:
            # return spikeFunctionLB(
            #     vmem,
            #     -self.ref_kernel,
            #     self.threshold,
            #     self.threshold_low,
            #     self.window_abs,
            #     self.scale_grads,
            # )
            return spikeFunctionOldForward(
                vmem,
                self.membrane_subtract,
                torch.zeros_like(vmem[:, 0]),
                torch.zeros_like(vmem[:, 0]),
                self.threshold,
                self.threshold_low,
                self.window_abs,
                self.scale_grads,
            )

        else:
            return spikeFunction(
                vmem,
                -self.ref_kernel,
                self.threshold,
                self.window_abs,
                self.scale_grads,
            )

    def _post_spike_processing(
        self,
        vmem: "torch.tensor",
        output_spikes: "torch.tensor",
        n_batches: int,
        n_neurons: Tuple[int],
    ) -> "torch.tensor":
        """
        Handle post-processing after spikes have been generated: Reshape output spikes,
        reshape and record vmem, record output spike statistics.

        Parameters
        ----------
        vmem : torch.tensor
            Membrane potential. Expected shape: (batches x neurons, time)
        output_spikes : torch.tensor
            Output spikes. Expected shape: (batches x neurons, time)
        n_batches : int
            Number of batches. Needed for correct reshaping of output and vmem
        n_neurons : Tuple of ints
            Remaining neuron dimensions (e.g. channels, height, width). Determines output
            shape after batch and time dimensions.

        Returns
        -------
        torch.tensor
            Output spikes. Shape: (n_batches, num_timesteps, *n_neurons)
        """

        # Separate batch and neuron dimensions -> (batches, *neurons, num_timesteps)
        vmem = vmem.reshape(n_batches, *n_neurons, self.num_timesteps)
        output_spikes = output_spikes.reshape(n_batches, *n_neurons, self.num_timesteps)

        # Move time dimension to second -> (batches, num_timesteps, *neurons)
        vmem = vmem.movedim(-1, 1)
        output_spikes = output_spikes.movedim(-1, 1)

        # Record states and output statistics
        self.vmem = vmem

        self.spikes_number = output_spikes.sum()
        self.n_spikes_out = output_spikes

        return output_spikes

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def _param_dict(self) -> dict:
        param_dict = super()._param_dict()
        param_dict.update(
            scale_grads=self.scale_grads,
            window=self.window_abs / self.threshold,
            num_timesteps=self.num_timesteps,
        )
