import time
import torch
import torch.nn as nn
import sinabs.exodus.layers as el
import sinabs.layers as sl
import sinabs.activation as sa
from sinabs.from_torch import from_model


atol = 1e-5
rtol = 1e-4


def test_iaf_basic():
    batch_size, time_steps = 10, 100
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7).cuda()
    layer = el.IAF().cuda()
    spike_output = layer(input_current)

    assert layer.does_spike
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    assert "EXODUS" in layer.__repr__()


def test_iaf_squeeze():
    batch_size, time_steps = 10, 100
    input_data = torch.rand(batch_size * time_steps, 2, 7, 7).cuda()
    layer = el.IAFSqueeze(batch_size=batch_size).cuda()
    spike_output = layer(input_data)

    assert input_data.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_min_v_mem():
    batch_size, time_steps = 10, 1
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / -(1 - alpha)
    layer = el.LIF(tau_mem=tau_mem).cuda()
    layer(input_data)
    assert (layer.v_mem < -0.5).any()

    layer = el.LIF(tau_mem=tau_mem, min_v_mem=-0.5).cuda()
    layer(input_data)
    assert not (layer.v_mem < -0.5).any()


def test_state_reset():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / (1 - alpha)
    layer = el.LIF(tau_mem=tau_mem).cuda()
    layer.reset_states()
    layer(input_data)
    assert (layer.v_mem != 0).any()

    layer.reset_states()
    assert (layer.v_mem == 0).all()


def test_exodus_sinabs_layer_equal_output():
    batch_size, time_steps = 10, 100
    n_input_channels = 16
    sinabs_model = sl.IAF().cuda()
    exodus_model = el.IAF().cuda()
    input_data = torch.zeros((batch_size, time_steps, n_input_channels)).cuda()
    input_data[:, :10] = 1e4
    spike_output_sinabs = sinabs_model(input_data)
    spike_output_exodus = exodus_model(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() > 0
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_exodus_sinabs_layer_equal_output_singlespike():
    batch_size, time_steps = 10, 100
    n_input_channels = 16
    sinabs_model = sl.IAF(spike_fn=sa.SingleSpike).cuda()
    exodus_model = el.IAF(spike_fn=sa.SingleSpike).cuda()
    input_data = torch.zeros((batch_size, time_steps, n_input_channels)).cuda()
    input_data[:, :10] = 1e4
    spike_output_sinabs = sinabs_model(input_data)
    spike_output_exodus = exodus_model(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() > 0
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_exodus_sinabs_layer_equal_output_maxspike():
    batch_size, time_steps = 10, 100
    n_input_channels = 16
    max_num_spikes = 3
    sinabs_model = sl.IAF(spike_fn=sa.MaxSpike(max_num_spikes)).cuda()
    exodus_model = el.IAF(spike_fn=sa.MaxSpike(max_num_spikes)).cuda()
    input_data = torch.zeros((batch_size, time_steps, n_input_channels)).cuda()
    input_data[:, :10] = 1e4
    spike_output_sinabs = sinabs_model(input_data)
    spike_output_exodus = exodus_model(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() > 0
    assert spike_output_sinabs.max() == max_num_spikes
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


def test_sinabs_model():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    model = SNN(
        spike_layer_class=sl.IAFSqueeze,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        batch_size=batch_size,
    ).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output = model(input_data)

    assert spike_output.shape == (batch_size, time_steps, n_output_classes)
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_exodus_model():
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    model = SNN(
        spike_layer_class=el.IAFSqueeze,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        batch_size=batch_size,
    ).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda() * 1e5
    spike_output = model(input_data)

    assert spike_output.shape == (batch_size, time_steps, n_output_classes)
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0


def test_exodus_sinabs_model_equal_output():
    n_epochs = 3
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    sinabs_model = SNN(
        spike_layer_class=sl.IAFSqueeze,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        batch_size=batch_size,
    ).cuda()
    exodus_model = SNN(
        spike_layer_class=el.IAFSqueeze,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        batch_size=batch_size,
    ).cuda()
    # make sure the weights for linear layers are the same
    for sinabs_layer, exodus_layer in zip(
        sinabs_model.linear_layers, exodus_model.linear_layers
    ):
        sinabs_layer.load_state_dict(exodus_layer.state_dict())
    assert (
        sinabs_model.linear_layers[0].weight == exodus_model.linear_layers[0].weight
    ).all()
    input_data = (
        torch.rand((n_epochs, batch_size, time_steps, n_input_channels)).cuda() * 1e5
    )

    # Iterate over a few epochs to make sure states are propagated correctly
    for inp_epoch in input_data:
        spike_output_sinabs = sinabs_model(inp_epoch)
        spike_output_exodus = exodus_model(inp_epoch)

        assert spike_output_sinabs.shape == spike_output_exodus.shape
        assert spike_output_sinabs.sum() == spike_output_exodus.sum()
        assert (spike_output_sinabs == spike_output_exodus).all()


def test_exodus_vs_sinabs_compare_grads():
    n_epochs = 3
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    sinabs_model = SinabsIAFModel(
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()
    exodus_model = ExodusIAFModel(
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
    ).cuda()

    # make sure the weights for linear layers are the same
    for sinabs_layer, exodus_layer in zip(
        sinabs_model.linear_layers, exodus_model.linear_layers
    ):
        sinabs_layer.load_state_dict(exodus_layer.state_dict())
    assert (
        sinabs_model.linear_layers[0].weight == exodus_model.linear_layers[0].weight
    ).all()

    # Iterate over a few epochs to make sure states and gradients are propagated correctly
    input_data = torch.rand((n_epochs, batch_size, time_steps, n_input_channels)).cuda()

    t_start = time.time()
    for inp_epoch in input_data:
        sinabs_out = sinabs_model(inp_epoch)
    loss_sinabs = torch.nn.functional.mse_loss(sinabs_out, torch.ones_like(sinabs_out))
    loss_sinabs.backward()
    grads_sinabs = [
        p.grad.data.clone() for p in sinabs_model.parameters() if p.grad is not None
    ]
    print(f"Runtime sinabs: {time.time() - t_start}")

    t_start = time.time()
    for inp_epoch in input_data:
        exodus_out = exodus_model(inp_epoch)
    loss_exodus = torch.nn.functional.mse_loss(exodus_out, torch.ones_like(exodus_out))
    loss_exodus.backward()
    grads_exodus = [
        p.grad.data.clone() for p in exodus_model.parameters() if p.grad is not None
    ]
    print(f"Runtime exodus: {time.time() - t_start}")

    # - Compare neuron states
    for l_sin, l_slyr in zip(sinabs_model.spiking_layers, exodus_model.spiking_layers):
        assert torch.allclose(l_sin.v_mem, l_slyr.v_mem, atol=atol, rtol=rtol)

    # - Compare outputs
    assert (sinabs_out == exodus_out).all()

    for g0, g1 in zip(grads_sinabs, grads_exodus):
        assert torch.allclose(g0, g1, atol=atol, rtol=rtol)


class ANN(nn.Sequential):
    def __init__(
        self,
        n_input_channels=16,
        n_output_classes=10,
    ):
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            nn.ReLU(),
            nn.Linear(16, 32, bias=False),
            nn.ReLU(),
            nn.Linear(32, n_output_classes, bias=False),
            nn.ReLU(),
        )


class SNN(nn.Module):
    def __init__(
        self,
        spike_layer_class,
        batch_size,
        n_input_channels=16,
        n_output_classes=10,
        threshold=1.0,
        min_v_mem=None,
    ):
        super().__init__()
        ann = ANN(n_input_channels=n_input_channels, n_output_classes=n_output_classes)
        self.network = from_model(
            ann,
            spike_layer_class=spike_layer_class,
            spike_threshold=threshold,
            min_v_mem=min_v_mem,
            batch_size=batch_size,
        ).spiking_model

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    def forward(self, data):
        return self.network(data)

    @property
    def spiking_layers(self):
        return [self.network[i] for i in (1, 3, 5)]

    @property
    def linear_layers(self):
        return [self.network[i] for i in (0, 2, 4)]


class SinabsIAFModel(nn.Sequential):
    def __init__(
        self,
        n_input_channels=16,
        n_output_classes=10,
        threshold=1.0,
        min_v_mem=None,
    ):
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            sl.IAF(spike_threshold=threshold, min_v_mem=min_v_mem),
            nn.Linear(16, 32, bias=False),
            sl.IAF(spike_threshold=threshold, min_v_mem=min_v_mem),
            nn.Linear(32, n_output_classes, bias=False),
            sl.IAF(spike_threshold=threshold, min_v_mem=min_v_mem),
        )

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    @property
    def spiking_layers(self):
        return [self[1], self[3], self[5]]

    @property
    def linear_layers(self):
        return [self[0], self[2], self[4]]


class ExodusIAFModel(nn.Sequential):
    def __init__(
        self,
        n_input_channels=16,
        n_output_classes=10,
        threshold=1.0,
        min_v_mem=None,
    ):
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            el.IAF(spike_threshold=threshold, min_v_mem=min_v_mem),
            nn.Linear(16, 32, bias=False),
            el.IAF(spike_threshold=threshold, min_v_mem=min_v_mem),
            nn.Linear(32, n_output_classes, bias=False),
            el.IAF(spike_threshold=threshold, min_v_mem=min_v_mem),
        )

    def reset_states(self):
        for lyr in self.spiking_layers:
            lyr.reset_states()

    @property
    def spiking_layers(self):
        return [self[1], self[3], self[5]]

    @property
    def linear_layers(self):
        return [self[0], self[2], self[4]]
