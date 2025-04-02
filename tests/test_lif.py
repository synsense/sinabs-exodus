import time
from itertools import product
import pytest
import torch
import torch.nn as nn
import sinabs.exodus.layers as el
import sinabs.layers as sl
import sinabs.activation as sa


atol = 1e-8
rtol = 1e-3


def test_lif_basic():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.rand(batch_size, time_steps, 2, 7, 7).cuda() / (1 - alpha)
    layer = el.LIF(tau_mem=tau_mem).cuda()
    spike_output = layer(input_current)

    assert layer.does_spike
    assert input_current.shape == spike_output.shape
    assert torch.isnan(spike_output).sum() == 0
    assert spike_output.sum() > 0
    assert "EXODUS" in layer.__repr__()


def test_lif_squeeze():
    batch_size, time_steps = 10, 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_data = torch.rand(batch_size * time_steps, 2, 7, 7).cuda() / (1 - alpha)
    layer = el.LIFSqueeze(tau_mem=tau_mem, batch_size=batch_size).cuda()
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
    torch.set_printoptions(precision=10)
    batch_size, time_steps, n_neurons = 10, 100, 20
    tau_mem = 20.0
    sinabs_layer = sl.LIF(tau_mem=tau_mem).cuda()
    exodus_layer = el.LIF(tau_mem=tau_mem).cuda()
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
    sinabs_layer = sl.LIF(tau_mem=tau_mem, spike_fn=sa.SingleSpike).cuda()
    exodus_layer = el.LIF(tau_mem=tau_mem, spike_fn=sa.SingleSpike).cuda()
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
    sinabs_layer = sl.LIF(tau_mem=tau_mem, spike_fn=sa.MaxSpike(max_num_spikes)).cuda()
    exodus_layer = el.LIF(tau_mem=tau_mem, spike_fn=sa.MaxSpike(max_num_spikes)).cuda()
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


@pytest.mark.parametrize("norm_input", (True, False))
def test_exodus_sinabs_model_equal_output(norm_input):
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.0
    tau_leak = 10.0

    sinabs_model = SinabsLIFModel(
        tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        norm_input=norm_input,
    ).cuda()
    exodus_model = ExodusLIFModel(
        tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        norm_input=norm_input,
    ).cuda()
    # make sure the weights for linear layers are the same
    for sinabs_layer, exodus_layer in zip(
        sinabs_model.linear_layers, exodus_model.linear_layers
    ):
        sinabs_layer.load_state_dict(exodus_layer.state_dict())
    assert (sinabs_model[0].weight == exodus_model[0].weight).all()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda()
    if not norm_input:
        input_data *= 1e-2
    spike_output_sinabs = sinabs_model(input_data)
    spike_output_exodus = exodus_model(input_data)

    assert spike_output_sinabs.shape == spike_output_exodus.shape
    assert spike_output_sinabs.sum() == spike_output_exodus.sum()
    assert (spike_output_sinabs == spike_output_exodus).all()


args = product((True, False), (True, False), (True, False))


@pytest.mark.parametrize("train_alphas,norm_input,train_time_consts", args)
def test_exodus_sinabs_state_transfer(train_alphas, norm_input, train_time_consts):
    batch_size, time_steps = 10, 100
    n_input_channels, n_output_classes = 16, 10
    tau_mem = 20.0
    tau_leak = 10.0

    model_kwargs = dict(
        tau_mem=tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        train_alphas=train_alphas,
        norm_input=norm_input,
    )
    sinabs_model = SinabsLIFModel(**model_kwargs).cuda()
    exodus_model = ExodusLIFModel(**model_kwargs).cuda()

    if not train_time_consts:
        for n, p in sinabs_model.named_parameters():
            if "tau" in n or "alpha" in n:
                p.requires_grad_(False)

    # make sure the weights for linear layers are the same
    sinabs_model.load_state_dict(exodus_model.state_dict())

    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda()
    if not norm_input:
        input_data *= 5e-2

    sinabs_out = sinabs_model(input_data)
    exodus_out = exodus_model(input_data)

    # - Export parameters from exodus to sinabs
    new_sinabs_model = SinabsLIFModel(**model_kwargs).cuda()
    new_sinabs_model(input_data)  # Enforce state initialization
    new_sinabs_model.load_state_dict(exodus_model.state_dict())
    # - Export parameters from sinabs to exodus
    new_exodus_model = ExodusLIFModel(**model_kwargs).cuda()
    new_exodus_model(input_data)  # Enforce state initialization
    new_exodus_model.load_state_dict(sinabs_model.state_dict())
    # - Evolve all four models
    sinabs_out = sinabs_model(input_data)
    exodus_out = exodus_model(input_data)
    new_sinabs_out = new_sinabs_model(input_data)
    new_exodus_out = new_exodus_model(input_data)
    for out in (exodus_out, new_sinabs_out, new_exodus_out):
        assert (out == sinabs_out).all()


args = product((True, False), (True, False), (True, False))


@pytest.mark.parametrize("train_alphas,norm_input,train_time_consts", args)
def test_exodus_vs_sinabs_compare_grads(train_alphas, norm_input, train_time_consts):
    n_epochs = 2
    batch_size, time_steps = 2, 5
    n_input_channels, n_output_classes = 2, 2
    tau_mem = 20.0
    tau_leak = 10.0

    model_kwargs = dict(
        tau_mem=tau_mem,
        tau_leak=tau_leak,
        n_input_channels=n_input_channels,
        n_output_classes=n_output_classes,
        train_alphas=train_alphas,
        norm_input=norm_input,
    )
    sinabs_model = SinabsLIFModel(**model_kwargs).cuda()
    exodus_model = ExodusLIFModel(**model_kwargs).cuda()

    if not train_time_consts:
        for n, p in sinabs_model.named_parameters():
            if "tau" in n or "alpha" in n:
                p.requires_grad_(False)

    # make sure the weights for linear layers are the same
    for sinabs_layer, exodus_layer in zip(
        sinabs_model.linear_layers, exodus_model.linear_layers
    ):
        sinabs_layer.load_state_dict(exodus_layer.state_dict())
    assert (sinabs_model[0].weight == exodus_model[0].weight).all()

    input_data = torch.rand((n_epochs, batch_size, time_steps, n_input_channels)).cuda()
    if not norm_input:
        input_data *= 5e-2

    t_start = time.time()
    for inp_epoch in input_data:
        sinabs_out = sinabs_model(inp_epoch)
    loss_sinabs = torch.nn.functional.mse_loss(sinabs_out, torch.ones_like(sinabs_out))
    loss_sinabs.backward()
    grads_sinabs = {
        k: p.grad for k, p in sinabs_model.named_parameters() if p.grad is not None
    }
    print(f"Runtime sinabs: {time.time() - t_start}")

    exodus_model.zero_grad()
    t_start = time.time()
    for inp_epoch in input_data:
        exodus_out = exodus_model(inp_epoch)
    loss_exodus = torch.nn.functional.mse_loss(exodus_out, torch.ones_like(exodus_out))
    loss_exodus.backward()
    grads_exodus = {
        k: p.grad for k, p in exodus_model.named_parameters() if p.grad is not None
    }
    print(f"Runtime exodus: {time.time() - t_start}")

    if norm_input:
        atol = 1e-7
        rtol = 1e-4
    else:
        atol = 1e-7
        rtol = 1e-2

    for l_sin, l_slyr in zip(exodus_model.spiking_layers, sinabs_model.spiking_layers):
        assert torch.allclose(l_sin.v_mem, l_slyr.v_mem, atol=atol, rtol=rtol)

    assert (sinabs_out == exodus_out).all()

    for k, g_sin in grads_sinabs.items():
        abs_diff = torch.abs(g_sin - grads_exodus[k])
        max_diff = torch.max(abs_diff).item()
        rel_diff = abs_diff / (torch.abs(torch.max(g_sin, grads_exodus[k])) + 1e-12)
        max_rel_diff = torch.max(rel_diff).item()
        print(k, "max difference:", max_diff, "max rel. diff.:", max_rel_diff)
        assert torch.allclose(g_sin, grads_exodus[k], atol=atol, rtol=rtol)

        corr, scale = correlation_and_scale(g_sin, grads_exodus[k])
        # Gradients must differ by less than 8 degree in direction and 5 per cent in length
        assert (1 - corr) < 1e-2, f"Correlation is {corr}"
        assert torch.abs(scale - 1) < 5e-2, f"Relative scale is {scale}"

    # - Export parameters from exodus to sinabs
    new_sinabs_model = SinabsLIFModel(**model_kwargs).cuda()
    new_sinabs_model(input_data[0])  # Enforce state initialization
    new_sinabs_model.load_state_dict(exodus_model.state_dict())
    # - Export parameters from sinabs to exodus
    new_exodus_model = ExodusLIFModel(**model_kwargs).cuda()
    new_exodus_model(input_data[0])  # Enforce state initialization
    new_exodus_model.load_state_dict(sinabs_model.state_dict())
    # - Evolve all four models
    sinabs_out = sinabs_model(input_data[0])
    exodus_out = exodus_model(input_data[0])
    new_sinabs_out = new_sinabs_model(input_data[0])
    new_exodus_out = new_exodus_model(input_data[0])
    for out in (exodus_out, new_sinabs_out, new_exodus_out):
        assert (out == sinabs_out).all()


args = product((True, False), (True, False), (None, 30))


@pytest.mark.parametrize("train_alphas,norm_input,tau_syn", args)
def test_exodus_vs_sinabs_compare_grads_single_layer(train_alphas, norm_input, tau_syn):
    batch_size, time_steps = 2, 10
    n_channels = 16
    tau_mem = 20.0
    spike_threshold = 1
    min_v_mem = -1

    sinabs_model = sl.LIF(
        tau_mem=torch.ones((n_channels,)) * tau_mem,
        tau_syn=tau_syn,
        spike_threshold=spike_threshold,
        min_v_mem=min_v_mem,
        norm_input=norm_input,
        train_alphas=train_alphas,
        record_states=True,
    ).cuda()
    exodus_model = el.LIF(
        tau_mem=torch.ones((n_channels,)) * tau_mem,
        tau_syn=tau_syn,
        spike_threshold=spike_threshold,
        min_v_mem=min_v_mem,
        norm_input=norm_input,
        train_alphas=train_alphas,
        record_states=True,
    ).cuda()

    input_data = torch.rand((batch_size, time_steps, n_channels)).cuda()
    # Non-zero initial state
    initial_state_v_mem = torch.rand_like(input_data[:, 0])
    sinabs_model.v_mem = initial_state_v_mem.clone()
    exodus_model.v_mem = initial_state_v_mem.clone()
    if tau_syn is not None:
        initial_state_i_syn = torch.rand_like(input_data[:, 0])
        sinabs_model.i_syn = initial_state_i_syn.clone()
        exodus_model.i_syn = initial_state_i_syn.clone()

    t_start = time.time()
    sinabs_out = sinabs_model(input_data)
    loss_sinabs = torch.nn.functional.mse_loss(sinabs_out, torch.ones_like(sinabs_out))
    loss_sinabs.backward()
    grads_sinabs = {
        k: p.grad for k, p in sinabs_model.named_parameters() if p.grad is not None
    }
    print(f"Runtime sinabs: {time.time() - t_start}")

    exodus_model.zero_grad()
    t_start = time.time()
    exodus_out = exodus_model(input_data)
    loss_exodus = torch.nn.functional.mse_loss(exodus_out, torch.ones_like(exodus_out))
    loss_exodus.backward()
    grads_exodus = {
        k: p.grad for k, p in exodus_model.named_parameters() if p.grad is not None
    }
    print(f"Runtime exodus: {time.time() - t_start}")

    # assert torch.allclose(sinabs_model.v_mem, exodus_model.v_mem, atol=atol, rtol=rtol)

    assert (sinabs_out.sum() == exodus_out.sum()).all()

    for k, g_sin in grads_sinabs.items():
        max_diff = torch.max(torch.abs(g_sin - grads_exodus[k])).item()
        print(k, "max difference:", max_diff)

        assert torch.allclose(g_sin, grads_exodus[k], atol=atol, rtol=5e-2)
        corr, scale = correlation_and_scale(g_sin, grads_exodus[k])
        assert torch.abs(corr - 1) < 1e-5, f"Correlation is {corr}"
        assert torch.abs(scale - 1) < 2e-2, f"Relative scale is {scale}"


def test_exodus_vs_sinabs_compare_grads_single_layer_simplified():
    batch_size, time_steps = 1, 20
    n_channels = 1
    tau_mem = 20.0
    spike_threshold = 1.2
    min_v_mem = 0
    train_alphas = True
    norm_input = False
    tau_syn = 30

    sinabs_model = sl.LIF(
        tau_mem=torch.ones((n_channels,)) * tau_mem,
        tau_syn=tau_syn,
        spike_threshold=spike_threshold,
        min_v_mem=min_v_mem,
        norm_input=norm_input,
        train_alphas=train_alphas,
        record_states=True,
    ).cuda()
    exodus_model = el.LIF(
        tau_mem=torch.ones((n_channels,)) * tau_mem,
        tau_syn=tau_syn,
        spike_threshold=spike_threshold,
        min_v_mem=min_v_mem,
        norm_input=norm_input,
        train_alphas=train_alphas,
        record_states=True,
    ).cuda()

    input_data = torch.zeros((batch_size, time_steps, n_channels)).cuda()
    input_data[:, 1] = 2
    initial_state = torch.rand_like(input_data[:, 0])

    # Non-zero initial state
    initial_state_v_mem = torch.rand_like(input_data[:, 0])
    if tau_syn is not None:
        initial_state_i_syn = torch.rand_like(input_data[:, 0])

    # Alpha-gradients for each time step
    def get_alpha_grads(model):
        """Get alpha gradients at specific time step"""
        model.zero_grad()
        model.reset_states()
        model.v_mem = initial_state_v_mem.clone()
        if tau_syn is not None:
            model.i_syn = initial_state_i_syn.clone()
        out = model(input_data)
        grads = []
        for t in range(input_data.shape[1]):
            ls = out[0, t, 0]
            ls.backward(retain_graph=True)
            grad = model.alpha_mem.grad if train_alphas else model.tau_mem.grad
            grads.append(grad.item())
            model.zero_grad()
        return torch.as_tensor(grads)

    exodus_grads = get_alpha_grads(exodus_model)
    sinabs_grads = get_alpha_grads(sinabs_model)

    assert torch.allclose(exodus_grads, sinabs_grads, atol=atol, rtol=rtol)


class SinabsLIFModel(nn.Sequential):
    def __init__(
        self,
        tau_mem,
        tau_leak,
        n_input_channels=16,
        n_output_classes=10,
        threshold=1.0,
        min_v_mem=None,
        train_alphas=False,
        norm_input=True,
    ):
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            sl.ExpLeak(
                tau_mem=torch.ones((16,)) * tau_leak,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            sl.LIF(
                tau_mem=torch.ones((16,)) * tau_mem,
                spike_threshold=threshold,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            nn.Linear(16, 32, bias=False),
            sl.ExpLeak(
                tau_mem=torch.ones((32,)) * tau_leak,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            sl.LIF(
                tau_mem=torch.ones((32,)) * tau_mem,
                spike_threshold=threshold,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            nn.Linear(32, n_output_classes, bias=False),
            sl.ExpLeak(
                tau_mem=torch.ones((n_output_classes,)) * tau_leak,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            sl.LIF(
                tau_mem=torch.ones((n_output_classes,)) * tau_mem,
                spike_threshold=threshold,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
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
        min_v_mem=None,
        norm_input=True,
        train_alphas=False,
    ):
        super().__init__(
            nn.Linear(n_input_channels, 16, bias=False),
            el.ExpLeak(
                tau_mem=torch.ones((16,)) * tau_leak,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            el.LIF(
                tau_mem=torch.ones((16,)) * tau_mem,
                spike_threshold=threshold,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            nn.Linear(16, 32, bias=False),
            el.ExpLeak(
                tau_mem=torch.ones((32,)) * tau_leak,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            el.LIF(
                tau_mem=torch.ones((32,)) * tau_mem,
                spike_threshold=threshold,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            nn.Linear(32, n_output_classes, bias=False),
            el.ExpLeak(
                tau_mem=torch.ones((n_output_classes,)) * tau_leak,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
            el.LIF(
                tau_mem=torch.ones((n_output_classes,)) * tau_mem,
                spike_threshold=threshold,
                min_v_mem=min_v_mem,
                norm_input=norm_input,
                train_alphas=train_alphas,
            ),
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


def correlation_and_scale(a, b):
    sum_of_sqrs_a = (a**2).sum()
    sum_of_sqrs_b = (b**2).sum()
    if sum_of_sqrs_a == 0 and sum_of_sqrs_b == 0:
        return torch.tensor(1), torch.tensor(1)  # both a and b are 0

    corr = (a * b).sum() / (torch.sqrt(sum_of_sqrs_a) * torch.sqrt(sum_of_sqrs_b))
    scale = sum_of_sqrs_a / sum_of_sqrs_b

    return corr, scale
