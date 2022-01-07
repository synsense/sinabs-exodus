import pytest
import time
import torch
import torch.nn as nn
import sinabs.slayer.layers as ssl
import sinabs.layers as sl
import sinabs.activation as sa
import numpy as np


atol = 1e-5
rtol = 1e-4


def test_lif_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / (1-alpha)
    layer = ssl.LIF(tau_mem=tau_mem).cuda()
    spike_output = layer(input_current)

    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_lif_squeeze():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.)
    alpha = torch.exp(-1/tau_mem)
    input_data = torch.rand(batch_size*time_steps, 2, 7, 7).cuda() / (1-alpha)
    layer = ssl.LIFSqueeze(tau_mem=tau_mem, batch_size=batch_size).cuda()
    spike_output = layer(input_data)

    assert input_data.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_sinabs_model():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.
    model = SinabsLIFModel(tau_mem, 
                            n_input_channels=n_input_channels, 
                            n_output_classes=n_output_classes).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output = model(input_data)

    assert input_data.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

def test_slayer_model():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.
    model = SlayerLIFModel(tau_mem, 
                            n_input_channels=n_input_channels, 
                            n_output_classes=n_output_classes).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output = model(input_data)

    assert input_data.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0

# test threshold_low
# test reset mechanisms

def test_slayer_sinabs_equal_output():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.
    sinabs_model = SinabsLIFModel(tau_mem, 
                            n_input_channels=n_input_channels, 
                            n_output_classes=n_output_classes).cuda()
    slayer_model = SlayerLIFModel(tau_mem, 
                            n_input_channels=n_input_channels, 
                            n_output_classes=n_output_classes).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output_sinabs = sinabs_model(input_data)
    spike_output_slayer = slayer_model(input_data)

    assert spike_output_sinabs.shape == spike_output_slayer.shape
    assert spike_output_sinabs.sum() == spike_output_slayer.sum()

def test_slayer_vs_sinabs_compare():
    num_timesteps = 500
    n_channels = 16
    batch_size = 100
    n_classes = 10
    tau_mem = 10.
    device = "cuda:0"

    # Define inputs
    input_data = (
        (torch.rand((num_timesteps * batch_size, n_channels)) > 0.95).float().to(device)
    )

    # Define models
    slayer_model = SlayerLIFModel(
        n_channels=n_channels, n_classes=n_classes, tau_mem=tau_mem
    ).to(device)
    sinabs_model = SinabsLIFModel(
        n_channels=n_channels, n_classes=n_classes, tau_mem=tau_mem
    ).to(device)

    def scale_all_weights_by_x(model, x):
        for param in model.parameters():
            param.data = param.data * x

    scale_all_weights_by_x(sinabs_model, 5.0)

    # Copy parameters
    slayer_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()
    slayer_model.lin2.weight.data = sinabs_model.lin2.weight.data.clone()
    slayer_model.lin3.weight.data = sinabs_model.lin3.weight.data.clone()

    # Optimizers for comparing gradients
    optim_slayer = torch.optim.SGD(slayer_model.parameters(), lr=1e-3)
    optim_sinabs = torch.optim.SGD(sinabs_model.parameters(), lr=1e-3)

    for i in range(3):
        # Sinabs
        sinabs_model.zero_grad()
        optim_sinabs.zero_grad()
        t_start = time.time()
        sinabs_out = sinabs_model(input_data)
        loss_sinabs = torch.nn.functional.mse_loss(
            sinabs_out, torch.ones_like(sinabs_out)
        )
        loss_sinabs.backward()
        grads_sinabs = [p.grad.data.clone() for p in sinabs_model.parameters()]
        optim_sinabs.step()

        t_stop = time.time()
        print(f"Runtime sinabs: {t_stop - t_start}")
        print("Sinabs model: ", sinabs_out.sum())

        # Slayer
        slayer_model.zero_grad()
        optim_slayer.zero_grad()
        t_start = time.time()
        slayer_out = slayer_model(input_data)
        loss_slayer = torch.nn.functional.mse_loss(
            slayer_out, torch.ones_like(slayer_out)
        )
        loss_slayer.backward()
        grads_slayer = [p.grad.data.clone() for p in slayer_model.parameters()]
        optim_slayer.step()
        t_stop = time.time()
        print(f"Runtime slayer: {t_stop - t_start}")
        print("Slayer model: ", slayer_out.sum())
        # print(slayer_out)

        ## Plot data
        # import matplotlib.pyplot as plt
        # plt.plot(sinabs_model.spk1.record[:, 0, 0].detach().cpu(), label="sinabs")
        # plt.plot(slayer_model.spk1.v_mem_recorded[0, 0, 0, 0].detach().cpu(), label="Slayer")
        # plt.legend()
        # plt.show()
        # plt.figure()
        # plt.scatter(*np.where(sinabs_out.cpu().detach().numpy()), marker=".")
        # plt.scatter(*np.where(slayer_out.cpu().detach().numpy()), marker="x")
        # plt.show()

        assert all(
            torch.allclose(l_sin.v_mem, l_slyr.v_mem, atol=atol, rtol=rtol)
            for (l_sin, l_slyr) in zip(
                slayer_model.spiking_layers, sinabs_model.spiking_layers
            )
        )
        assert (sinabs_out == slayer_out).all()

        # Compare gradients
        assert all(
            torch.allclose(g0, g1, atol=atol, rtol=rtol)
            for g0, g1 in zip(grads_sinabs, grads_slayer)
        )




class SinabsLIFModel(nn.Module):
    def __init__(self, 
                tau_mem, 
                n_input_channels=16, 
                n_output_classes=10, 
                threshold=1.0, 
                threshold_low=None,
                ):
        super().__init__()
        act_fn = sa.ActivationFunction(spike_threshold=threshold)

        self.lin1 = nn.Linear(n_input_channels, 16, bias=False)
        self.spk1 = sl.LIF(
            tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low
        )
        self.lin2 = nn.Linear(16, 32, bias=False)
        self.spk2 = sl.LIF(
            tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low
        )
        self.lin3 = nn.Linear(32, n_output_classes, bias=False)
        self.spk3 = sl.LIF(
            tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low
        )

    def forward(self, data):
        out = self.lin1(data)
        out = self.spk1(out)
        out = self.lin2(out)
        out = self.spk2(out)
        out = self.lin3(out)
        out = self.spk3(out)

        return out

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    def zero_grad(self):
        for lyr in self.spiking_layers:
            lyr.zero_grad()

    @property
    def spiking_layers(self):
        return [self.spk1, self.spk2, self.spk3]


class SlayerLIFModel(nn.Module):
    def __init__(self,     
                tau_mem,
                n_input_channels=16,
                n_output_classes=10,
                threshold=1.0,
                threshold_low=None,
                ):
        super().__init__()
        act_fn = sa.ActivationFunction(spike_threshold=threshold)

        self.lin1 = nn.Linear(n_input_channels, 16, bias=False)
        self.spk1 = ssl.LIF(
            tau_mem=tau_mem,
            activation_fn=act_fn,
            threshold_low=threshold_low,
        )

        self.lin2 = nn.Linear(16, 32, bias=False)
        self.spk2 = ssl.LIF(
            tau_mem=tau_mem,
            activation_fn=act_fn,
            threshold_low=threshold_low,
        )

        self.lin3 = nn.Linear(32, n_output_classes, bias=False)
        self.spk3 = ssl.LIF(
            tau_mem=tau_mem,
            activation_fn=act_fn,
            threshold_low=threshold_low,
        )

    def forward(self, data):
        # expected input shape  (batch, num_timesteps, n_channels)
        out = self.lin1(data)
        out = self.spk1(out)
        out = self.lin2(out)
        out = self.spk2(out)
        out = self.lin3(out)
        out = self.spk3(out)
        return out

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    def zero_grad(self):
        for lyr in self.spiking_layers:
            lyr.zero_grad()

    @property
    def spiking_layers(self):
        return [self.spk1, self.spk2, self.spk3]

