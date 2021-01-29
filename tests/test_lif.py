def test_lif_inference():
    import torch
    from sinabs2.slayer.layers.lif import SpikingLayer

    t_sim = 100
    tau_mem = 10
    tau_syn = [5.0, 15.0]
    threshold = 1.0
    batch_size = 32
    h,w = 45, 55

    device = "cuda:0"

    input_data = torch.rand((batch_size, h, w, t_sim)).to(device)
    layer = SpikingLayer(tau_mem, tau_syn, threshold, batch_size).to(device)

    output = layer(input_data)
    assert(output.shape == input_data.shape)


def build_sinabs_model(tau_mem, tau_syn, n_channels=16, n_classes=10, batch_size=1):
    import torch.nn as nn
    from sinabs.layers.cpp.lif_bptt import SpikingLayer

    threshold = 1.0


    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_syn = len(tau_syn)
            self.lin1 = nn.Linear(n_channels, 2*16, bias=False)
            self.spk1 = SpikingLayer(tau_mem, tau_syn, threshold, batch_size=batch_size)
            self.lin2 = nn.Linear(16, 64, bias=False)
            self.spk2 = SpikingLayer(tau_mem, tau_syn, threshold, batch_size=batch_size)
            self.lin3 = nn.Linear(32, self.n_syn*n_classes, bias=False)
            self.spk3 = SpikingLayer(tau_mem, tau_syn, threshold, batch_size=batch_size)

        def forward(self, data):
            out = self.lin1(data)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1]//self.n_syn))
            out = self.spk1(out)
            out = self.lin2(out)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1]//self.n_syn))
            out = self.spk2(out)
            out = self.lin3(out)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1]//self.n_syn ))
            out = self.spk3(out)
            return out

    return TestModel()


def test_sinabs_model():
    import torch
    import numpy as np
    t_sim = 100
    n_channels = 16
    batch_size = 1
    n_classes = 10
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_sinabs_model(tau_syn, tau_mem, n_channels=n_channels, n_classes=n_classes, batch_size=1).to(device)
    input_data = torch.rand((batch_size, t_sim, n_channels)).to(device).reshape((-1, n_channels))
    out = model(input_data)
    assert out.shape == (batch_size*t_sim, n_classes)


def build_slayer_model(tau_mem, tau_syn, n_channels=16, n_classes=10, batch_size=1):
    import torch.nn as nn
    from sinabs2.slayer.layers.lif import SpikingLayer

    threshold = 1.0


    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_syn = len(tau_syn)
            self.lin1 = nn.Linear(n_channels, 16*2, bias=False)
            self.spk1 = SpikingLayer(tau_mem, tau_syn, threshold, batch_size=batch_size)
            self.lin2 = nn.Linear(16, 2*32, bias=False)
            self.spk2 = SpikingLayer(tau_mem, tau_syn, threshold, batch_size=batch_size)
            self.lin3 = nn.Linear(32, self.n_syn*n_classes, bias=False)
            self.spk3 = SpikingLayer(tau_mem, tau_syn, threshold, batch_size=batch_size)

        def forward(self, data):
            # reshape to (time, batch, dim, n_channels)
            out = data.movedim(-1, 0)
            out = self.lin1(out)
            # reshape to (n_syn, batch, dim, n_channels, time)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1]//self.n_syn))
            out = out.movedim(0, -1)
            out = out.movedim(-3, 0)
            out = self.spk1(out)
            out = out.movedim(-1, 0)
            out = self.lin2(out)
            # reshape to (n_syn, batch, dim, n_channels, time)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1]//self.n_syn))
            out = out.movedim(0, -1)
            out = out.movedim(-3, 0)
            out = self.spk2(out)
            out = out.movedim(-1, 0)
            out = self.lin3(out)
            # reshape to (n_syn, batch, dim, n_channels, time)
            shape = out.shape
            out = out.view((*shape[:-1], self.n_syn, shape[-1]//self.n_syn))
            out = out.movedim(0, -1)
            out = out.movedim(-3, 0)
            out = self.spk3(out)
            return out

    return TestModel()


def test_slayer_model():
    import torch
    import numpy as np
    t_sim = 100
    n_channels = 16
    n_dims = 1
    batch_size = 1
    n_classes = 10
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_slayer_model(tau_mem=tau_mem, tau_syn=tau_syn, n_channels=n_channels, n_classes=n_classes, batch_size=1).to(device)
    input_data = torch.rand((batch_size, n_dims, 1, n_channels, t_sim)).to(device)

    out = model(input_data)
    assert out.shape == (batch_size, n_dims, 1, n_classes, t_sim)


def test_slayer_vs_sinabs_compare():
    import torch
    import time
    import numpy as np
    t_sim = 500
    n_channels = 16
    batch_size = 100
    n_classes = 10
    device = "cuda:0"

    tau_mem = 50.
    tau_syn = np.array([50.0, 50.0])
    # Define inputs
    input_data = (torch.rand((batch_size, n_channels, t_sim)) > 0.95).float().to(device)
    input_data_sinabs = input_data.movedim(-1, 1).reshape((-1, n_channels))
    input_data_slayer = input_data.unsqueeze(1).unsqueeze(1)  # Add an additional dimension to input
    assert len(input_data_slayer.shape) == 5

    # Define models
    slayer_model = build_slayer_model(tau_mem, tau_syn, n_channels=n_channels, n_classes=n_classes, batch_size=batch_size).to(device)
    sinabs_model = build_sinabs_model(tau_mem, tau_syn, n_channels=n_channels, n_classes=n_classes, batch_size=batch_size).to(device)

    assert(input_data_slayer.sum() == input_data_sinabs.sum())

    def scale_all_weights_by_x(model, x):
        for param in model.parameters():
            param.data = param.data*x

    scale_all_weights_by_x(sinabs_model, 0.04)

    # Copy parameters
    slayer_model.lin1.weight.data = sinabs_model.lin1.weight.data.clone()
    slayer_model.lin2.weight.data = sinabs_model.lin2.weight.data.clone()
    slayer_model.lin3.weight.data = sinabs_model.lin3.weight.data.clone()


    t_start = time.time()
    sinabs_out = sinabs_model(input_data_sinabs)
    t_stop = time.time()
    print(f"Runtime sinabs: {t_stop - t_start}")

    t_start = time.time()
    slayer_out = slayer_model(input_data_slayer)
    t_stop = time.time()
    print(f"Runtime slayer: {t_stop - t_start}")



    print("Sinabs model: ", sinabs_out.sum())
    print("Slayer model: ", slayer_out.sum())
    print(slayer_out)

    print(sinabs_model.spk1.vmem_rec.shape)
    print(slayer_model.spk1.vmem.shape)

    # Plot data
    import matplotlib.pyplot as plt
    plt.plot(sinabs_model.spk1.vmem_rec[:, 0, 0].detach().cpu(), label="sinabs")
    plt.plot(slayer_model.spk1.vmem[0,0,0,0].detach().cpu(), label="Slayer")
    plt.legend()
    plt.show()

    assert abs(sinabs_out.sum() - slayer_out.sum()) <= 5*sinabs_out.sum()/100.0
