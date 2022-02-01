import torch
import sinabs.layers as sl
import sinabs.slayer.layers as el


def test_leaky_basic():
    time_steps = 100
    tau_leak = torch.tensor(30.0)
    input_current = torch.rand(time_steps, 2, 7, 7).cuda()
    layer = el.ExpLeak(tau_leak=tau_leak).cuda()
    membrane_output = layer(input_current)

    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0


def test_leaky_squeezed():
    batch_size = 10
    time_steps = 100
    tau_leak = torch.tensor(30.0)
    input_current = torch.rand(batch_size * time_steps, 2, 7, 7).cuda()
    layer = el.ExpLeakSqueeze(tau_leak=tau_leak, batch_size=batch_size).cuda()
    membrane_output = layer(input_current)

    assert input_current.shape == membrane_output.shape
    assert torch.isnan(membrane_output).sum() == 0
    assert membrane_output.sum() > 0


def test_leaky_membrane_decay():
    batch_size = 10
    time_steps = 100
    tau_leak = torch.tensor(30.0)
    alpha = torch.exp(-1 / tau_leak)
    input_current = torch.zeros(batch_size, time_steps, 2, 7, 7).cuda()
    input_current[:, 0] = 1 / (1 - alpha)  # only inject current in the first time step
    layer = el.ExpLeak(tau_leak=tau_leak).cuda()
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


def test_slayer_sinabs_layer_equal_output():
    batch_size, time_steps = 10, 100
    n_input_channels = 16
    tau_leak = 10
    sinabs_model = sl.ExpLeak(tau_leak=tau_leak).cuda()
    slayer_model = el.ExpLeak(tau_leak=tau_leak).cuda()
    input_data = torch.zeros((batch_size, time_steps, n_input_channels)).cuda()
    input_data[:, :10] = 1e4
    output_sinabs = sinabs_model(input_data)
    output_slayer = slayer_model(input_data)

    assert output_sinabs.shape == output_slayer.shape
    assert (output_sinabs != 0).any()
    assert torch.allclose(output_sinabs, output_slayer)
