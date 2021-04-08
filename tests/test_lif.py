def test_lif_inference():
    import torch
    from sinabs.slayer.layers.lif import SpikingLayer

    t_sim = 100
    tau_mem = 10
    tau_syn = [5.0, 15.0]
    threshold = 1.0
    batch_size = 32
    n_neurons = 45

    device = "cuda:0"

    input_data = torch.rand((t_sim, batch_size, len(tau_syn), n_neurons)).to(device)
    layer = SpikingLayer(tau_mem, tau_syn, threshold, batch_size).to(device)

    output = layer(input_data)
    assert (output.shape == (t_sim, batch_size, n_neurons))


def build_sinabs_model(tau_mem, tau_syn, n_channels=16, n_classes=10, batch_size=1):
    import torch.nn as nn
    from sinabs.layers.cpp.lif_bptt import SpikingLayer

    threshold = 1.0

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_syn = len(tau_syn)
            self.lin1 = nn.Linear(n_channels, 2 * 16, bias=False)
            self.spk1 = SpikingLayer(tau_mem, tau_syn, threshold)
            self.lin2 = nn.Linear(16, 64, bias=False)
            self.spk2 = SpikingLayer(tau_mem, tau_syn, threshold)
            self.lin3 = nn.Linear(32, self.n_syn * n_classes, bias=False)
            self.spk3 = SpikingLayer(tau_mem, tau_syn, threshold)

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
    model = build_sinabs_model(tau_mem, tau_syn, n_channels=n_channels, n_classes=n_classes, batch_size=1).to(device)
    input_data = torch.rand((batch_size, t_sim, n_channels)).to(device)
    out = model(input_data)
    assert out.shape == (batch_size, t_sim, n_classes)


def build_slayer_model(tau_mem, tau_syn, n_channels=16, n_classes=10, batch_size=1):
    import torch.nn as nn
    from sinabs.slayer.layers.lif import SpikingLayer

    threshold = 1.0

    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.n_syn = len(tau_syn)
            self.lin1 = nn.Linear(n_channels, 16 * 2, bias=False)
            self.spk1 = SpikingLayer(tau_mem, tau_syn, threshold)
            self.lin2 = nn.Linear(16, 2 * 32, bias=False)
            self.spk2 = SpikingLayer(tau_mem, tau_syn, threshold)
            self.lin3 = nn.Linear(32, self.n_syn * n_classes, bias=False)
            self.spk3 = SpikingLayer(tau_mem, tau_syn, threshold)

        def forward(self, data):
            # expected input shape  (time, batch, n_channels)
            (t_sim, n_batches, n_channels) = data.shape
            out = self.lin1(data)
            out = out.view((t_sim, n_batches, self.n_syn, -1))
            out = self.spk1(out)
            out = self.lin2(out)
            out = out.view((t_sim, n_batches, self.n_syn, -1))
            out = self.spk2(out)
            out = self.lin3(out)
            out = out.view((t_sim, n_batches, self.n_syn, -1))
            out = self.spk3(out)
            return out

    return TestModel()


def test_slayer_model():
    import torch
    import numpy as np
    t_sim = 100
    n_channels = 16
    batch_size = 1
    n_classes = 10
    device = "cuda:0"
    tau_mem = 10
    tau_syn = np.array([5.0, 15.0])
    model = build_slayer_model(tau_mem=tau_mem, tau_syn=tau_syn, n_channels=n_channels, n_classes=n_classes,
                               batch_size=1).to(device)
    input_data = torch.rand((t_sim, batch_size, n_channels)).to(device)

    out = model(input_data)
    assert out.shape == (t_sim, batch_size, n_classes)


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
    input_data = (torch.rand((t_sim, batch_size, n_channels)) > 0.95).float().to(device)
    
    # Define models
    slayer_model = build_slayer_model(tau_mem, tau_syn, n_channels=n_channels, n_classes=n_classes,
                                      batch_size=batch_size).to(device)
    sinabs_model = build_sinabs_model(tau_mem, tau_syn, n_channels=n_channels, n_classes=n_classes,
                                      batch_size=batch_size).to(device)
    
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
    #import matplotlib.pyplot as plt
    #plt.plot(sinabs_model.spk1.vmem_rec[:, 0, 0].detach().cpu(), label="sinabs")
    #plt.plot(slayer_model.spk1.vmem[0, 0, 0, 0].detach().cpu(), label="Slayer")
    #plt.legend()
    #plt.show()
    
    assert abs(sinabs_out.sum() - slayer_out.sum()) <= 10 * sinabs_out.sum() / 100.0
