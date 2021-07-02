def test_iaf_inference():
    import torch
    import pytest
    from sinabs.slayer.layers import IAFSqueeze, IAF

    num_timesteps = 100
    threshold = 1.0
    batch_size = 32
    n_neurons = (3, 3, 5)

    device = "cuda:0"

    input_data = torch.rand((batch_size, num_timesteps, *n_neurons)).to(device)
    input_data_squeeze = input_data.reshape(-1, *n_neurons)
    layer_squeeze = IAFSqueeze(num_timesteps, threshold).to(device)
    layer = IAF(num_timesteps, threshold).to(device)

    # Make sure wrong input dimensions are detected
    with pytest.raises(ValueError):
        output = layer(torch.rand((batch_size, num_timesteps + 1, *n_neurons)))

    output = layer(input_data)
    assert output.shape == input_data.shape

    output_squeeze = layer_squeeze(input_data_squeeze)
    assert output_squeeze.shape == input_data_squeeze.shape
    assert (output_squeeze == output.reshape(-1, *n_neurons)).all()


def build_sinabs_model(n_channels=16, n_classes=10, batch_size=1):
    import torch.nn as nn
    from sinabs.layers import IAFSqueeze

    threshold = 1.0

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(n_channels, 16, bias=False)
            self.spk1 = IAFSqueeze(
                threshold=threshold, threshold_low=None, batch_size=batch_size
            )
            self.lin2 = nn.Linear(16, 32, bias=False)
            self.spk2 = IAFSqueeze(
                threshold=threshold, threshold_low=None, batch_size=batch_size
            )
            self.lin3 = nn.Linear(32, n_classes, bias=False)
            self.spk3 = IAFSqueeze(
                threshold=threshold, threshold_low=None, batch_size=batch_size
            )

        def forward(self, data):
            out = self.lin1(data)
            out = self.spk1(out)
            out = self.lin2(out)
            out = self.spk2(out)
            out = self.lin3(out)
            out = self.spk3(out)
            return out

    return TestModel()


def test_sinabs_model():
    import torch

    num_timesteps = 100
    n_channels = 16
    batch_size = 2
    n_classes = 10
    device = "cuda:0"
    model = build_sinabs_model(
        n_channels=n_channels, n_classes=n_classes, batch_size=1
    ).to(device)
    input_data = torch.rand((batch_size * num_timesteps, n_channels)).to(device)
    out = model(input_data)
    assert out.shape == (batch_size * num_timesteps, n_classes)


def build_slayer_model(n_channels=16, n_classes=10, num_timesteps=100, scale_grads=1.0):
    import torch.nn as nn
    from sinabs.slayer.layers import IAFSqueeze

    threshold = 1.0

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin1 = nn.Linear(n_channels, 16, bias=False)
            self.spk1 = IAFSqueeze(
                num_timesteps=num_timesteps,
                threshold=threshold,
                scale_grads=scale_grads,
            )
            self.lin2 = nn.Linear(16, 32, bias=False)
            self.spk2 = IAFSqueeze(
                num_timesteps=num_timesteps,
                threshold=threshold,
                scale_grads=scale_grads,
            )
            self.lin3 = nn.Linear(32, n_classes, bias=False)
            self.spk3 = IAFSqueeze(
                num_timesteps=num_timesteps,
                threshold=threshold,
                scale_grads=scale_grads,
            )

        def forward(self, data):
            out = self.lin1(data)
            out = self.spk1(out)
            out = self.lin2(out)
            out = self.spk2(out)
            out = self.lin3(out)
            out = self.spk3(out)
            return out

    return TestModel()


def test_slayer_model():
    import torch

    num_timesteps = 100
    n_channels = 16
    batch_size = 2
    n_classes = 10
    device = "cuda:0"
    model = build_slayer_model(
        n_channels=n_channels, n_classes=n_classes, num_timesteps=num_timesteps
    ).to(device)

    input_data = torch.rand((num_timesteps * batch_size, n_channels)).to(device)

    out = model(input_data)
    assert out.shape == (num_timesteps * batch_size, n_classes)


def test_gradient_scaling():
    import torch

    torch.manual_seed(0)
    num_timesteps = 100
    n_channels = 16
    batch_size = 1
    n_classes = 2
    device = "cuda:0"
    model = build_slayer_model(
        n_channels=n_channels, n_classes=n_classes, num_timesteps=num_timesteps
    ).to(device)
    initial_weights = [p.data.clone() for p in model.parameters()]
    input_data = torch.rand((num_timesteps * batch_size, n_channels)).to(device)

    out = model(input_data).cpu()
    loss = torch.nn.functional.mse_loss(out, torch.ones_like(out))
    loss.backward()
    grads = [p.grad for p in model.parameters()]
    # Calculate ratio of std of first and last layer gradients
    grad_ratio = torch.std(grads[0]) / torch.std(grads[-1])

    # Generate identical model, except for gradient scaling
    model_new = build_slayer_model(
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


def test_slayer_vs_sinabs_compare():
    import torch
    import time

    num_timesteps = 500
    n_channels = 16
    batch_size = 100
    n_classes = 10
    device = "cuda:0"

    # Define inputs
    input_data = (
        (torch.rand((num_timesteps * batch_size, n_channels)) > 0.95).float().to(device)
    )

    # Define models
    slayer_model = build_slayer_model(
        n_channels=n_channels, n_classes=n_classes, num_timesteps=num_timesteps
    ).to(device)
    sinabs_model = build_sinabs_model(
        n_channels=n_channels, n_classes=n_classes, batch_size=batch_size
    ).to(device)

    def scale_all_weights_by_x(model, x):
        for param in model.parameters():
            param.data = param.data * x

    scale_all_weights_by_x(sinabs_model, 1.0)

    # Copy parameters
    slayer_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()
    slayer_model.lin2.weight.data = sinabs_model.lin2.weight.data.clone()
    slayer_model.lin3.weight.data = sinabs_model.lin3.weight.data.clone()

    t_start = time.time()
    sinabs_out = sinabs_model(input_data.view((-1, n_channels)))
    t_stop = time.time()
    print(f"Runtime sinabs: {t_stop - t_start}")

    t_start = time.time()
    slayer_out = slayer_model(input_data)
    t_stop = time.time()
    print(f"Runtime slayer: {t_stop - t_start}")

    print("Sinabs model: ", sinabs_out.sum())
    print("Slayer model: ", slayer_out.sum())
    print(slayer_out)

    ## Plot data
    # import matplotlib.pyplot as plt
    # plt.plot(sinabs_model.spk1.record[:, 0, 0].detach().cpu(), label="sinabs")
    # plt.plot(slayer_model.spk1.vmem[0, 0, 0, 0].detach().cpu(), label="Slayer")
    # plt.legend()
    # plt.show()
    # plt.figure()
    # plt.scatter(*np.where(sinabs_out.cpu().detach().numpy()), marker=".")
    # plt.scatter(*np.where(slayer_out.cpu().detach().numpy()), marker="x")
    # plt.show()

    assert (sinabs_out == slayer_out).all()
