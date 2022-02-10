import torch
from torch import nn
from time import time
import numpy as np
from sinabs.exodus.layers import LIF
from sinabs.exodus.leaky import LeakyIntegrator
from sinabs.exodus.spike import SpikeFunction
from sinabs.activation import ActivationFunction


class LIFSplit(LIF):
    """
    Same as LIF but leaky integration and spike generation are implemented internally
    by two separate functions.
    """
    def forward(self, spike_input: "torch.tensor") -> "torch.tensor":
        """
        Generate membrane potential and resulting output spike train based on
        spiking input. Membrane potential will be stored as `self.v_mem_recorded`.

        Parameters
        ----------
        spike_input: torch.tensor
            Spike input raster. May take non-binary integer values.
            Expected shape: (n_batches, num_timesteps, ...)

        Returns
        -------
        torch.tensor
            Output spikes. Same shape as `spike_input`.
        """
        if self.norm_input:
            spike_input = (1.0 - self.alpha_mem) * spike_input

        n_batches, num_timesteps, *n_neurons = spike_input.shape

        # Ensure the neuron state are initialized
        if not self.is_state_initialised() or not self.state_has_shape(
            (n_batches, *n_neurons)
        ):
            self.init_state_with_shape((n_batches, *n_neurons))

        # Move time to last dimension -> (n_batches, *neuron_shape, num_timesteps)
        # Flatten out all dimensions that can be processed in parallel and ensure contiguity
        spike_input = spike_input.movedim(1, -1).reshape(-1, num_timesteps)

        filtered_input = LeakyIntegrator.apply(
            spike_input.contiguous(),
            self.v_mem.flatten(),
            self.alpha_mem,
            False
        )

        output_spikes, v_mem_full = SpikeFunction.apply(
            filtered_input,
            self.membrane_subtract,
            self.alpha_mem,
            self.surrogate_grad_fn,
            self.threshold,
            self.threshold_low,
            self.max_num_spikes_per_bin,
        )

        # Reshape output spikes and v_mem_full, store neuron states
        v_mem_full = v_mem_full.reshape(n_batches, *n_neurons, -1).movedim(-1, 1)
        output_spikes = output_spikes.reshape(n_batches, *n_neurons, -1).movedim(-1, 1)

        if self.record_v_mem:
            self.v_mem_recorded = v_mem_full

        # update neuron states
        self.v_mem = v_mem_full[:, -1].clone()

        return output_spikes

def build_exodus_model(
    n_channels=16,
    n_classes=10,
    tau_mem=10.,
    threshold=1.0,
    threshold_low=-1,
    norm_input=False,
):
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(5, 5, 1),
                bias=False,
            )
            activation_fn = ActivationFunction(spike_threshold=threshold)
            self.spk1 = LIF(
                activation_fn=activation_fn,
                threshold_low=threshold_low,
                tau_mem=tau_mem,
                record_v_mem=True,
                norm_input=norm_input,
            )

        # @profile
        def forward(self, data):
            out = self.lin1(data)
            out = out.movedim(-1, 1)
            out = self.spk1(out)
            out = out.movedim(1, -1)
            return out

    return TestModel()


def build_split_model(
    n_channels=16,
    n_classes=10,
    tau_mem=10.,
    threshold=1.0,
    threshold_low=-1,
    norm_input=False,
):
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(5, 5, 1),
                bias=False,
            )
            activation_fn = ActivationFunction(spike_threshold=threshold)
            self.spk1 = LIFSplit(
                activation_fn=activation_fn,
                threshold_low=threshold_low,
                tau_mem=tau_mem,
                record_v_mem=True,
                norm_input=norm_input,
            )

        # @profile
        def forward(self, data):
            out = self.lin1(data)
            out = out.movedim(-1, 1)
            out = self.spk1(out)
            out = out.movedim(1, -1)
            return out

    return TestModel()


num_timesteps = 500
batch_size = 2
n_classes = 10
device = "cuda:0"


neuron_shape = (10, 128, 128)
n_channels = neuron_shape[0]

# Define inputs
input_data = (
    (torch.rand((batch_size, *neuron_shape, num_timesteps)) > 0.8).float().to(device)
)

# Define models
exodus_model = build_exodus_model(
    n_channels=n_channels,
    n_classes=n_classes,
    threshold_low=None,
).to(device)
split_model = build_split_model(
    n_channels=n_channels,
    n_classes=n_classes,
    threshold_low=None,
).to(device)

# Copy parameters
split_model.lin1.weight.data = exodus_model.lin1.weight.data.clone()

# # Optimizers for comparing gradients
# optim_exodus = torch.optim.SGD(exodus_model.parameters(), lr=1e-3)
# optim_slayer_orig = torch.optim.SGD(split_model.parameters(), lr=1e-3)
# optim_sinabs = torch.optim.SGD(sinabs_model.parameters(), lr=1e-3)


# @profile
def forward_pass():
    print("----- Forward pass -----")

    # Exodus
    print("Testing exodus")
    t0 = time()
    out_exodus = exodus_model(input_data)
    num_spikes_exodus = out_exodus.sum()
    print(f"\tNum spikes: {num_spikes_exodus}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    # Split
    print("Testing split model")
    t0 = time()
    out_split = split_model(input_data)
    num_spikes_split = out_split.sum()
    print(f"\tNum spikes: {num_spikes_split}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    return num_spikes_exodus, num_spikes_split


# @profile
def backward_pass(num_spikes_exodus, num_spikes_split):
    print("----- Backward pass -----")

    # Exodus
    print("Testing exodus")
    t0 = time()
    num_spikes_exodus.backward()
    grad_sum_exodus = sum(p.grad.sum() for p in exodus_model.parameters())
    print(f"\tGrad sum: {grad_sum_exodus}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    # Split
    print("Testing split model")
    t0 = time()
    num_spikes_split.backward()
    grad_sum_split = sum(p.grad.sum() for p in split_model.parameters())
    print(f"\tGrad sum: {grad_sum_split}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    ## Comparison of gradients
    g_exodus = [p.grad for p in exodus_model.parameters()][0]
    g_split = [p.grad for p in split_model.parameters()][0]

    # Correlation between gradients. (numpy seems more accurate here)
    corr = np.corrcoef(g_exodus.flatten().detach().cpu(), g_split.flatten().detach().cpu())
    print(f"\tGrad correlation: {corr[0,1]}")

    # Mean relative deviation
    mrd = torch.mean(torch.abs((g_exodus - g_split) / (g_exodus + 1e-5)))
    print(f"\tGrad mean relative deviation: {mrd}")

    # Max relative deviation
    maxrd = torch.max(torch.abs((g_exodus - g_split) / (g_exodus + 1e-5)))
    print(f"\tGrad max relative deviation: {maxrd}")

    # Compare mean difference to find systematic errors
    mrds = torch.mean((g_exodus - g_split) / (g_exodus + 1e-5))
    print(f"\tGrad mean signed rel diff: {mrds}")

if __name__ == "__main__":
    num_spikes_exodus, num_spikes_split = forward_pass()
    backward_pass(num_spikes_exodus, num_spikes_split)
