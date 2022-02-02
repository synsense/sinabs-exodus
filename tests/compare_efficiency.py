from math import prod
import torch
from torch import nn
from time import time
from sinabs.layers import IAF
from sinabs.exodus.layers import IAF as IAFS
from slayerSNN import layer as IAFSOrig


def build_sinabs_model(n_channels=16, n_classes=10, threshold=1.0, threshold_low=None):
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(5, 5, 1),
                bias=False,
            )
            self.spk1 = IAF(threshold=threshold, threshold_low=threshold_low)

        # @profile
        def forward(self, data):
            out = self.lin1(data)
            out = out.movedim(-1, 1)
            out = self.spk1(out)
            out = out.movedim(1, -1)
            return out

        def reset_states(self):
            for lyr in [self.spk1]:
                lyr.reset_states()

    return TestModel()


def build_exodus_model(
    n_channels=16,
    n_classes=10,
    num_timesteps=100,
    scale_grads=1.0,
    threshold=1.0,
    threshold_low=None,
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
            self.spk1 = IAFS(
                num_timesteps=num_timesteps,
                threshold=threshold,
                threshold_low=threshold_low,
                scale_grads=scale_grads,
            )

        # @profile
        def forward(self, data):
            out = self.lin1(data)
            out = out.movedim(-1, 1)
            out = self.spk1(out)
            out = out.movedim(1, -1)
            return out

    return TestModel()


def build_slayer_orig_model(
    n_channels=16, n_classes=10, num_timesteps=100, scale_grads=1.0, threshold=1.0
):

    neuron_params = {
        "theta": threshold,
        "scaleRho": scale_grads,
        "tauRho": 0.5,
        "tauSr": num_timesteps // 30,
        "tauRef": num_timesteps // 30,
        "scaleRef": 1,
    }
    sim_params = {"tSample": num_timesteps, "Ts": 1}

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Conv3d(
                in_channels=n_channels,
                out_channels=n_channels,
                kernel_size=(5, 5, 1),
                bias=False,
            )
            self.spk1 = IAFSOrig(neuron_params, sim_params)

        # @profile
        def forward(self, data):
            out = self.lin1(data)
            # out = out.movedim(1, -1)
            # shape = out.shape
            # if out.ndim == 3:
            #     out = out.reshape(shape[0], shape[1], 1, 1, shape[2])
            # elif out.ndim == 4:
            #     out = out.reshape(shape[0], shape[1], 1, shape[2], shape[3])
            # if out.ndim == 3:
            #     out = out.reshape(shape[0], shape[1], shape[2], shape[3], shape[4])
            # psp = self.spk1.psp(out.unsqueeze(-1).unsqueeze(-1).movedim(1, -1))
            psp = self.spk1.psp(out)
            out = self.spk1.spike(psp)
            # return out.movedim(-1, 1).squeeze(-1).squeeze(-1)
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
    num_timesteps=num_timesteps,
    threshold_low=-1,
).to(device)
slayer_orig_model = build_slayer_orig_model(
    n_channels=n_channels, n_classes=n_classes, num_timesteps=num_timesteps
).to(device)
sinabs_model = build_sinabs_model(
    n_channels=n_channels, n_classes=n_classes, threshold_low=-1
).to(device)

# Copy parameters
exodus_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()
slayer_orig_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()

# # Optimizers for comparing gradients
# optim_exodus = torch.optim.SGD(exodus_model.parameters(), lr=1e-3)
# optim_slayer_orig = torch.optim.SGD(slayer_orig_model.parameters(), lr=1e-3)
# optim_sinabs = torch.optim.SGD(sinabs_model.parameters(), lr=1e-3)


# @profile
def forward_pass():
    print("----- Forward pass -----")

    # Sinabs
    print("Testing sinabs")
    t0 = time()
    out_sinabs = sinabs_model(input_data)
    num_spikes_sinabs = out_sinabs.sum()
    print(f"\tNum spikes: {num_spikes_sinabs}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    # Exodus
    print("Testing exodus")
    t0 = time()
    out_exodus = exodus_model(input_data)
    num_spikes_exodus = out_exodus.sum()
    print(f"\tNum spikes: {num_spikes_exodus}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    # Sinabs
    print("Testing slayer orig")
    t0 = time()
    out_slyr_orig = slayer_orig_model(input_data)
    num_spikes_slyr_orig = out_slyr_orig.sum()
    print(f"\tNum spikes: {num_spikes_slyr_orig}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    return num_spikes_sinabs, num_spikes_exodus, num_spikes_slyr_orig


# @profile
def backward_pass(num_spikes_sinabs, num_spikes_exodus, num_spikes_slyr_orig):
    print("----- Backward pass -----")

    # Sinabs
    print("Testing sinabs")
    t0 = time()
    num_spikes_sinabs.backward()
    grad_sum_sinabs = sum(p.grad.sum() for p in sinabs_model.parameters())
    print(f"\tGrad sum: {grad_sum_sinabs}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    # Exodus
    print("Testing exodus")
    t0 = time()
    num_spikes_exodus.backward()
    grad_sum_exodus = sum(p.grad.sum() for p in exodus_model.parameters())
    print(f"\tGrad sum: {grad_sum_exodus}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")

    # Sinabs
    print("Testing slayer orig")
    t0 = time()
    num_spikes_slyr_orig.backward()
    grad_sum_slyr_orig = sum(p.grad.sum() for p in slayer_orig_model.parameters())
    print(f"\tGrad sum: {grad_sum_slyr_orig}")
    t1 = time()
    print(f"\tTook {t1 - t0:.2e} s")


if __name__ == "__main__":
    num_spikes_sinabs, num_spikes_exodus, num_spikes_slyr_orig = forward_pass()
    backward_pass(num_spikes_sinabs, num_spikes_exodus, num_spikes_slyr_orig)
