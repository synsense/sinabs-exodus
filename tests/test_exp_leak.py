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


def test_leaky_basic_early_decay():
    time_steps = 100
    tau_mem = torch.tensor(30.0)
    input_current = torch.rand(time_steps, 2, 7, 7).cuda()
    layer_dec = el.ExpLeak(tau_mem=tau_mem, decay_early=True).cuda()
    layer = el.ExpLeak(tau_mem=tau_mem).cuda()
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
    layer = el.ExpLeak(tau_mem=tau_mem, norm_input=True).cuda()
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
