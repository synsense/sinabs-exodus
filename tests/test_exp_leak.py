from itertools import product
import time
import pytest
import torch
import sinabs.layers as sl
import sinabs.exodus.layers as el


def test_leaky_basic():
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    input_current = torch.rand(time_steps, 2, 7, 7).cuda()
    layer = el.ExpLeak(tau_mem=tau_mem).cuda()
    membrane_output = layer(input_current)

    assert not layer.does_spike
    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0
    assert "EXODUS" in layer.__repr__()


def test_leaky_basic_early_decay():
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    input_current = torch.rand(time_steps, 2, 7, 7).cuda()
    layer_dec = el.ExpLeak(tau_mem=tau_mem, decay_early=True).cuda()
    layer = el.ExpLeak(tau_mem=tau_mem, decay_early=False).cuda()
    membrane_output = layer(input_current)
    membrane_output_dec = layer_dec(input_current)

    assert input_current.shape == membrane_output_dec.shape
    assert torch.isnan(membrane_output_dec).sum() == 0
    assert membrane_output_dec.sum() > 0
    assert torch.allclose(
        membrane_output_dec, membrane_output * layer_dec.alpha_mem_calculated
    )


def test_leaky_squeezed():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    input_current = torch.rand(batch_size * time_steps, 2, 7, 7).cuda()
    layer = el.ExpLeakSqueeze(tau_mem=tau_mem, batch_size=batch_size).cuda()
    membrane_output = layer(input_current)

    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0


def test_leaky_membrane_decay():
    batch_size = 10
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_mem)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7).cuda()
    input_current[:, 0] = 1 / (1 - alpha)  # only inject current in the first time step
    layer = el.ExpLeak(tau_mem=tau_mem, norm_input=True, decay_early=False).cuda()
    membrane_output = layer(input_current)

    # first time step is not decayed
    membrane_decay = alpha ** (time_steps - 1)

    # account for rounding errors with .isclose()
    assert torch.allclose(
        membrane_output[:, 0], torch.tensor(1.0)
    ), "Output for first time step is not correct."
    assert (
        membrane_output[:, -1] == layer.v_mem
    ).all(), "Output of last time step does not correspond to last layer state."
    assert torch.isclose(
        layer.v_mem, membrane_decay, atol=1e-08
    ).all(), "Neuron membrane potentials do not seems to decay correctly."


def test_exodus_sinabs_layer_equal_output():
    batch_size, time_steps = 10, 100
    n_input_channels = 16
    tau_mem = 10.0
    sinabs_model = sl.ExpLeak(tau_mem=tau_mem, norm_input=True).cuda()
    exodus_model = el.ExpLeak(tau_mem=tau_mem, norm_input=True).cuda()
    input_data = torch.zeros((batch_size, time_steps, n_input_channels)).cuda()
    input_data[:, :10] = 1e4
    output_sinabs = sinabs_model(input_data)
    output_exodus = exodus_model(input_data)

    assert output_sinabs.shape == output_exodus.shape
    assert (output_sinabs != 0).any()
    assert torch.allclose(output_sinabs, output_exodus)


atol = 1e-8
rtol = 1e-3

args = product((True, False), (True, False))


@pytest.mark.parametrize("train_alphas,norm_input", args)
def test_exodus_vs_sinabs_compare_grads(train_alphas, norm_input):
    batch_size, time_steps = 10, 100
    n_input_channels = 16
    tau_mem = 10.0
    sinabs_model = sl.ExpLeak(
        tau_mem=tau_mem, norm_input=norm_input, train_alphas=train_alphas
    ).cuda()
    exodus_model = el.ExpLeak(
        tau_mem=tau_mem, norm_input=norm_input, train_alphas=train_alphas
    ).cuda()
    input_data = torch.rand((batch_size, time_steps, n_input_channels)).cuda()

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

    assert torch.allclose(sinabs_model.v_mem, exodus_model.v_mem, atol=atol, rtol=rtol)
    assert torch.allclose(sinabs_out, exodus_out)

    for k, g_sin in grads_sinabs.items():
        assert torch.allclose(g_sin, grads_exodus[k], atol=atol, rtol=rtol)


def test_exodus_vs_sinabs_compare_grads_single_layer_simplified():
    batch_size, time_steps = 1, 20
    n_channels = 1
    tau_mem = 20.0
    train_alphas = True
    norm_input = False

    sinabs_model = sl.ExpLeak(
        tau_mem=torch.ones((n_channels,)) * tau_mem,
        norm_input=norm_input,
        train_alphas=train_alphas,
        record_states=True,
    ).cuda()
    exodus_model = el.ExpLeak(
        tau_mem=torch.ones((n_channels,)) * tau_mem,
        norm_input=norm_input,
        train_alphas=train_alphas,
        record_states=True,
    ).cuda()

    input_data = torch.zeros((batch_size, time_steps, n_channels)).cuda()
    input_data[:, 1] = 2
    # initial_state = torch.rand_like(input_data[:, 0])

    # Alpha-gradients for each time step
    def get_alpha_grads(model):
        """Get alpha gradients at specific time step"""
        model.zero_grad()
        model.reset_states()
        # model.v_mem = initial_state.clone()
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
