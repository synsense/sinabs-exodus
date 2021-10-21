import pytest


def test_lif_inference():
    import torch
    from sinabs.slayer.layers import LIF, LIFSqueeze

    num_timesteps = 100
    tau_mem = 10
    tau_syn = [5.0, 15.0]
    threshold = 0.2
    threshold_low = -0.2
    batch_size = 32
    n_neurons = 45

    device = "cuda:0"

    rand_data = torch.rand((batch_size, num_timesteps, len(tau_syn), n_neurons))
    input_data = (rand_data > 0.95).float().to(device)
    input_data_squeeze = input_data.reshape(-1, len(tau_syn), n_neurons)

    layer = LIF(
        num_timesteps=num_timesteps,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
    ).to(device)
    layer_squeeze = LIFSqueeze(
        num_timesteps=num_timesteps,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
    ).to(device)
    layer_squeeze_thr_low = LIFSqueeze(
        num_timesteps=num_timesteps,
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        threshold=threshold,
        threshold_low=threshold_low,
    ).to(device)

    # Make sure wrong input dimensions are detected
    with pytest.raises(ValueError):
        output = layer(
            torch.rand((batch_size, num_timesteps, len(tau_syn) + 1, n_neurons))
        )

    output = layer(input_data)
    assert output.shape == (batch_size, num_timesteps, n_neurons)

    output_squeeze = layer_squeeze(input_data_squeeze)
    assert output_squeeze.shape == (batch_size * num_timesteps, n_neurons)
    assert (output_squeeze == output.reshape(-1, n_neurons)).all()

    output_thrlow = layer_squeeze_thr_low(input_data_squeeze)
    assert (output_thrlow != output_squeeze).any()

    # # Make sure vmem is not below threshold_low for two consecutive timesteps
    # # This test might fail even if the layer works correctly
    # vmem = layer_squeeze_thr_low.vmem
    # assert not (
    #     torch.logical_and(vmem[:, 1:] < threshold_low, vmem[:, :-1] < threshold_low)
    # ).any()


def build_sinabs_model(
    tau_mem,
    tau_syn,
    n_channels=16,
    n_classes=10,
    batch_size=1,
    threshold=1.0,
    threshold_low=None,
):
    import torch.nn as nn
    from sinabs.layers.cpp.lif_bptt import SpikingLayer

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_syn = len(tau_syn)
            self.lin1 = nn.Linear(n_channels, 2 * 16, bias=False)
            self.spk1 = SpikingLayer(
                tau_mem, tau_syn, threshold, threshold_low=threshold_low
            )
            self.lin2 = nn.Linear(16, 64, bias=False)
            self.spk2 = SpikingLayer(
                tau_mem, tau_syn, threshold, threshold_low=threshold_low
            )
            self.lin3 = nn.Linear(32, self.n_syn * n_classes, bias=False)
            self.spk3 = SpikingLayer(
                tau_mem, tau_syn, threshold, threshold_low=threshold_low
            )

        def forward(self, data):
            out = self.lin1(data)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1] // self.n_syn))
            out = self.spk1(out)
            out = self.lin2(out)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1] // self.n_syn))
            out = self.spk2(out)
            out = self.lin3(out)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1] // self.n_syn))
            out = self.spk3(out)
            return out

    return TestModel()


@pytest.mark.skip("sinabs-cpp not stable")
def test_sinabs_model():
    import torch
    import numpy as np

    num_timesteps = 100
    n_channels = 16
    batch_size = 1
    n_classes = 10
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_sinabs_model(
        tau_mem, tau_syn, n_channels=n_channels, n_classes=n_classes, batch_size=1
    ).to(device)
    input_data = torch.rand((batch_size, num_timesteps, n_channels)).to(device)
    out = model(input_data)
    assert out.shape == (batch_size, num_timesteps, n_classes)


def build_slayer_model(
    tau_mem,
    tau_syn,
    num_timesteps=None,
    batch_size=None,
    n_channels=16,
    n_classes=10,
    scale_grads=1.0,
    threshold=1.0,
    threshold_low=None,
):
    import torch.nn as nn
    from sinabs.slayer.layers import LIF

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_syn = len(tau_syn)
            self.lin1 = nn.Linear(n_channels, 16 * 2, bias=False)
            self.spk1 = LIF(
                num_timesteps=num_timesteps,
                batch_size=batch_size,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                threshold=threshold,
                threshold_low=threshold_low,
                scale_grads=scale_grads,
            )

            self.lin2 = nn.Linear(16, 2 * 32, bias=False)
            self.spk2 = LIF(
                num_timesteps=num_timesteps,
                batch_size=batch_size,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                threshold=threshold,
                threshold_low=threshold_low,
                scale_grads=scale_grads,
            )

            self.lin3 = nn.Linear(32, self.n_syn * n_classes, bias=False)
            self.spk3 = LIF(
                num_timesteps=num_timesteps,
                batch_size=batch_size,
                tau_mem=tau_mem,
                tau_syn=tau_syn,
                threshold=threshold,
                threshold_low=threshold_low,
                scale_grads=scale_grads,
            )

        def forward(self, data):
            # expected input shape  (batch, num_timesteps, n_channels)
            (n_batches, num_timesteps, n_channels) = data.shape
            out = self.lin1(data)
            out = out.view((n_batches, num_timesteps, self.n_syn, -1))
            out = self.spk1(out)
            out = self.lin2(out)
            out = out.view((n_batches, num_timesteps, self.n_syn, -1))
            out = self.spk2(out)
            out = self.lin3(out)
            out = out.view((n_batches, num_timesteps, self.n_syn, -1))
            out = self.spk3(out)
            return out

    return TestModel()


def test_slayer_model():
    import torch
    import numpy as np

    num_timesteps = 100
    n_channels = 16
    batch_size = 1
    n_classes = 10
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_slayer_model(
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
    ).to(device)
    input_data = torch.rand((batch_size, num_timesteps, n_channels)).to(device)

    out = model(input_data)
    assert out.shape == (batch_size, num_timesteps, n_classes)


def test_slayer_model_batch():
    import torch
    import numpy as np

    num_timesteps = 100
    n_channels = 16
    batch_size = 1
    n_classes = 10
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_slayer_model(
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        batch_size=batch_size,
    ).to(device)
    input_data = torch.rand((batch_size, num_timesteps, n_channels)).to(device)

    out = model(input_data)
    assert out.shape == (batch_size, num_timesteps, n_classes)


def test_gradient_scaling():
    import torch
    import numpy as np

    torch.manual_seed(0)
    num_timesteps = 100
    n_channels = 16
    batch_size = 1
    n_classes = 2
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_slayer_model(
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
        scale_grads=1.0,
    ).to(device)
    initial_weights = [p.data.clone() for p in model.parameters()]
    input_data = torch.rand((batch_size, num_timesteps, n_channels)).to(device)

    out = model(input_data).cpu()
    loss = torch.nn.functional.mse_loss(out, torch.ones_like(out))
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    # Calculate ratio of std of first and last layer gradients
    grad_ratio = torch.std(grads[0]) / torch.std(grads[-1])

    # Generate identical model, except for gradient scaling
    model_new = build_slayer_model(
        tau_mem=tau_mem,
        tau_syn=tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
        scale_grads=0.1,
    ).to(device)
    for p_new, p_old in zip(model_new.parameters(), initial_weights):
        p_new.data = p_old.clone()

    out_new = model_new(input_data).cpu()
    # Make sure output is the same as for original model
    assert (out_new == out).all()

    # Compare gradient ratios
    loss_new = torch.nn.functional.mse_loss(out_new, torch.ones_like(out))

    # Make sure loss is the same as for original model
    assert (loss_new == loss).all()

    loss_new.backward()
    grads_new = [p.grad for p in model_new.parameters()]
    grad_ratio_new = torch.std(grads_new[0]) / torch.std(grads_new[-1])

    # Deepest layer gradient should be much smaller than before
    assert grad_ratio_new < 0.5 * grad_ratio


@pytest.mark.skip("sinabs-cpp not stable")
def test_slayer_vs_sinabs_compare():
    import torch
    import time
    import numpy as np

    num_timesteps = 500
    n_channels = 16
    batch_size = 100
    n_classes = 10
    device = "cuda:0"

    tau_mem = 50.0
    tau_syn = np.array([50.0, 50.0])
    # Define inputs
    input_data = (
        (torch.rand((batch_size, num_timesteps, n_channels)) > 0.95).float().to(device)
    )

    # Define models
    slayer_model = build_slayer_model(
        tau_mem,
        tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
    ).to(device)
    sinabs_model = build_sinabs_model(
        tau_mem,
        tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
    ).to(device)

    def scale_all_weights_by_x(model, x):
        for param in model.parameters():
            param.data = param.data * x

    scale_all_weights_by_x(sinabs_model, 0.02)

    # Copy parameters
    slayer_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()
    slayer_model.lin2.weight.data = sinabs_model.lin2.weight.data.clone()
    slayer_model.lin3.weight.data = sinabs_model.lin3.weight.data.clone()

    t_start = time.time()
    sinabs_out = sinabs_model(input_data)
    t_stop = time.time()
    print(f"Runtime sinabs: {t_stop - t_start}")

    t_start = time.time()
    slayer_out = slayer_model(input_data)
    t_stop = time.time()
    print(f"Runtime slayer: {t_stop - t_start}")

    print("Sinabs model: ", sinabs_out.sum())
    print("Slayer model: ", slayer_out.sum())
    print(slayer_out)

    print(sinabs_model.spk1.vmem_rec.shape)
    print(slayer_model.spk1.vmem.shape)

    # Plot data
    # import matplotlib.pyplot as plt
    # plt.plot(sinabs_model.spk1.vmem_rec[:, 0, 0].detach().cpu(), label="sinabs")
    # plt.plot(slayer_model.spk1.vmem[0, 0, 0, 0].detach().cpu(), label="Slayer")
    # plt.legend()
    # plt.show()

    assert abs(sinabs_out.sum() - slayer_out.sum()) <= 10 * sinabs_out.sum() / 100.0


@pytest.mark.skip("sinabs-cpp not stable")
def test_slayer_vs_sinabs_compare_thr_low():
    import torch
    import time
    import numpy as np

    num_timesteps = 500
    n_channels = 16
    batch_size = 100
    n_classes = 10
    device = "cuda:0"
    threshold = 0.7
    threshold_low = -0.3

    tau_mem = 50.0
    tau_syn = np.array([50.0, 50.0])
    # Define inputs
    input_data = (
        (torch.rand((batch_size, num_timesteps, n_channels)) > 0.95).float().to(device)
    )

    # Define models
    slayer_model = build_slayer_model(
        tau_mem,
        tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
        threshold=threshold,
        threshold_low=threshold_low,
    ).to(device)
    slayer_model_nothrlow = build_slayer_model(
        tau_mem,
        tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
        threshold=threshold,
    ).to(device)
    sinabs_model = build_sinabs_model(
        tau_mem,
        tau_syn,
        n_channels=n_channels,
        n_classes=n_classes,
        num_timesteps=num_timesteps,
        threshold=threshold,
    ).to(device)

    def scale_all_weights_by_x(model, x):
        for param in model.parameters():
            param.data = param.data * x

    scale_all_weights_by_x(sinabs_model, 0.02)

    # Copy parameters
    slayer_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()
    slayer_model.lin2.weight.data = sinabs_model.lin2.weight.data.clone()
    slayer_model.lin3.weight.data = sinabs_model.lin3.weight.data.clone()

    t_start = time.time()
    sinabs_out = sinabs_model(input_data)
    t_stop = time.time()
    print(f"Runtime sinabs: {t_stop - t_start}")

    t_start = time.time()
    slayer_out = slayer_model(input_data)
    t_stop = time.time()
    print(f"Runtime slayer: {t_stop - t_start}")

    t_start = time.time()
    slayer_out_nothrlow = slayer_model_nothrlow(input_data)
    t_stop = time.time()
    print(f"Runtime slayer, no lower threshold: {t_stop - t_start}")

    print("Sinabs model: ", sinabs_out.sum())
    print("Slayer model: ", slayer_out.sum())
    print("Slayer model, no lower threshold: ", slayer_out_nothrlow.sum())
    print(slayer_out)

    print(sinabs_model.spk1.vmem_rec.shape)
    print(slayer_model.spk1.vmem.shape)

    # Plot data
    # import matplotlib.pyplot as plt
    # plt.plot(sinabs_model.spk1.vmem_rec[:, 0, 0].detach().cpu(), label="sinabs")
    # plt.plot(slayer_model.spk1.vmem[0, 0, 0, 0].detach().cpu(), label="Slayer")
    # plt.legend()
    # plt.show()

    assert abs(sinabs_out.sum() - slayer_out.sum()) <= 10 * sinabs_out.sum() / 100.0
    # Make sure there is actually a difference from adding the lower threshold
    assert (torch.abs(slayer_out_nothrlow - slayer_out) > 1e-3).any()
