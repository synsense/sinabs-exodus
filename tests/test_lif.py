import pytest
import time
import torch
import torch.nn as nn
import sinabs.exodus.layers as ssl
import sinabs.layers as sl
import sinabs.activation as sa


atol = 1e-2
rtol = 1e-2


def test_lif_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / (1 - alpha)
    layer = ssl.LIF(tau_mem=tau_mem).cuda()
    spike_output = layer(input_current)

    assert layer.does_spike
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_lif_squeeze():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size * time_steps, 2, 7, 7).cuda() / (1 - alpha)
    layer = ssl.LIFSqueeze(tau_mem=tau_mem, batch_size=batch_size).cuda()
    spike_output = layer(input_data)

    assert input_data.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_threshold_low():
    batch_size, time_steps = 10, 1
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / -(1 - alpha)
    layer = ssl.LIF(tau_mem=tau_mem).cuda()
    layer(input_data)
    assert (layer.v_mem < -0.5).any()

    layer = ssl.LIF(tau_mem=tau_mem, threshold_low=-0.5).cuda()
    layer(input_data)
    assert not (layer.v_mem < -0.5).any()


def test_state_reset():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / (1 - alpha)
    layer = ssl.LIF(tau_mem=tau_mem).cuda()
    layer.reset_states()
    layer(input_data)
    assert (layer.v_mem != 0).any()

    layer.reset_states()
    assert (layer.v_mem == 0).all()


def test_exodus_sinabs_layer_equal_output():
    torch.set_printoptions(precision=10)
    batch_size, time_steps, n_neurons = 10, 100, 20
    tau_mem = 20.0
    sinabs_layer = sl.LIF(tau_mem=tau_mem).cuda()
    exodus_layer = ssl.LIF(tau_mem=tau_mem).cuda()
    input_data = torch.rand((batch_size, time_steps, n_neurons)).cuda() * 1e2
    spike_output_sinabs = sinabs_layer(input_data)
    spike_output_exodus = exodus_layer(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() > 0
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_exodus_sinabs_layer_equal_output_singlespike():
    torch.set_printoptions(precision=10)
    batch_size, time_steps, n_neurons = 10, 100, 20
    tau_mem = 20.0
    activation_fn = sa.ActivationFunction(spike_fn=sa.SingleSpike)
    sinabs_layer = sl.LIF(tau_mem=tau_mem, activation_fn=activation_fn).cuda()
    exodus_layer = ssl.LIF(tau_mem=tau_mem, activation_fn=activation_fn).cuda()
    input_data = torch.rand((batch_size, time_steps, n_neurons)).cuda() * 1e2
    spike_output_sinabs = sinabs_layer(input_data)
    spike_output_exodus = exodus_layer(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() > 0
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_exodus_sinabs_layer_equal_output_maxspike():
    torch.set_printoptions(precision=10)
    batch_size, time_steps, n_neurons = 10, 100, 20
    tau_mem = 20.0
    max_num_spikes = 3
    activation_fn = sa.ActivationFunction(spike_fn=sa.MaxSpike(max_num_spikes))
    sinabs_layer = sl.LIF(tau_mem=tau_mem, activation_fn=activation_fn).cuda()
    exodus_layer = ssl.LIF(tau_mem=tau_mem, activation_fn=activation_fn).cuda()
    input_data = torch.rand((batch_size, time_steps, n_neurons)).cuda() * 1e2
    spike_output_sinabs = sinabs_layer(input_data)
    spike_output_exodus = exodus_layer(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() > 0
    assert spike_output_sinabs.max() == max_num_spikes
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_sinabs_model():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.0
    tau_leak = 10.0

    model = SinabsLIFModel(
        tau_mem,
        tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output = model(input_data)

    assert spike_output.shape == (batch_size, time_steps, n_output_classes)
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_exodus_model():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.0
    tau_leak = 10.0

    model = ExodusLIFModel(
        tau_mem,
        tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output = model(input_data)

    assert spike_output.shape == (batch_size, time_steps, n_output_classes)
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_exodus_sinabs_model_equal_output():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.0
    tau_leak = 10.0

    sinabs_model = SinabsLIFModel(
        tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()
    exodus_model = ExodusLIFModel(
        tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()
    # make sure the weights for linear layers are the same
    for (sinabs_layer, exodus_layer) in zip(
        sinabs_model.linear_layers, exodus_model.linear_layers
    ):
        sinabs_layer.load_state_dict(exodus_layer.state_dict())
    assert (sinabs_model[0].weight == exodus_model[0].weight).all()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output_sinabs = sinabs_model(input_data)
    spike_output_exodus = exodus_model(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_exodus_vs_sinabs_compare_grads():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.0
    tau_leak = 10.0

    sinabs_model = SinabsLIFModel(
        tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()
    exodus_model = ExodusLIFModel(
        tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()

    # make sure the weights for linear layers are the same
    for (sinabs_layer, exodus_layer) in zip(
        sinabs_model.linear_layers, exodus_model.linear_layers
    ):
        sinabs_layer.load_state_dict(exodus_layer.state_dict())
    assert (sinabs_model[0].weight == exodus_model[0].weight).all()

    for layer in sinabs_model.spiking_layers:
        layer.tau_mem.requires_grad = False

    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda()

    t_start = time.time()
    sinabs_out = sinabs_model(input_data)
    loss_sinabs = torch.nn.functional.mse_loss(sinabs_out, torch.ones_like(sinabs_out))
    loss_sinabs.backward()
    grads_sinabs = [lyr.weight.grad.data.float() for lyr in sinabs_model.linear_layers]
    print(f"Runtime sinabs: {time.time() - t_start}")

    exodus_model.zero_grad()
    t_start = time.time()
    exodus_out = exodus_model(input_data)
    loss_exodus = torch.nn.functional.mse_loss(exodus_out, torch.ones_like(exodus_out))
    loss_exodus.backward()
    grads_exodus = [lyr.weight.grad.data.float() for lyr in exodus_model.linear_layers]
    print(f"Runtime exodus: {time.time() - t_start}")

    for (l_sin, l_slyr) in zip(
        exodus_model.spiking_layers, sinabs_model.spiking_layers
    ):
        assert torch.allclose(l_sin.v_mem, l_slyr.v_mem, atol=atol, rtol=rtol)

    assert (sinabs_out == exodus_out).all()

    for g0, g1 in zip(grads_sinabs, grads_exodus):
        assert torch.allclose(g0, g1, atol=atol, rtol=rtol)


class SinabsLIFModel(nn.Sequential):
    def __init__(
        self,
        tau_mem,
        tau_leak,
        n_input_channels=16,
        n_output_classes=10,
        threshold=1.0,
        threshold_low=None,
    ):
        act_fn = sa.ActivationFunction(spike_threshold=threshold)
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            sl.ExpLeak(tau_leak=tau_leak),
            sl.LIF(tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low),
            nn.Linear(16, 32, bias=False),
            sl.ExpLeak(tau_leak=tau_leak),
            sl.LIF(tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low),
            nn.Linear(32, n_output_classes, bias=False),
            sl.ExpLeak(tau_leak=tau_leak),
            sl.LIF(tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low),
        )

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    def zero_grad(self):
        for lyr in self.spiking_layers:
            lyr.zero_grad()

    @property
    def spiking_layers(self):
        return [self[2], self[5], self[8]]

    @property
    def linear_layers(self):
        return [self[0], self[3], self[6]]


class ExodusLIFModel(nn.Sequential):
    def __init__(
        self,
        tau_mem,
        tau_leak,
        n_input_channels=16,
        n_output_classes=10,
        threshold=1.0,
        threshold_low=None,
    ):
        act_fn = sa.ActivationFunction(spike_threshold=threshold)
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            ssl.ExpLeak(tau_leak=tau_leak),
            ssl.LIF(tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low),
            nn.Linear(16, 32, bias=False),
            ssl.ExpLeak(tau_leak=tau_leak),
            ssl.LIF(tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low),
            nn.Linear(32, n_output_classes, bias=False),
            ssl.ExpLeak(tau_leak=tau_leak),
            ssl.LIF(tau_mem=tau_mem, activation_fn=act_fn, threshold_low=threshold_low),
        )

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    def zero_grad(self):
        for lyr in self.spiking_layers:
            lyr.zero_grad()

    @property
    def spiking_layers(self):
        return [self[2], self[5], self[8]]

    @property
    def linear_layers(self):
        return [self[0], self[3], self[6]]
