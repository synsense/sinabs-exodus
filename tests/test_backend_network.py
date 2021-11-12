from copy import deepcopy

import torch
from torch import nn

from sinabs.from_torch import from_model


class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        seq = [
            nn.Conv2d(
                2, 8, kernel_size=(2, 2), stride=(2, 2), padding=(0, 0), bias=False
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                8, 16, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Conv2d(
                16, 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False
            ),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(2, 2)),
            nn.Flatten(),
            nn.Linear(8 * 8 * 8, 11, bias=True),
            nn.ReLU(),
        ]
        self.main = nn.Sequential(*seq)

    def forward(self, x):
        return self.main(x)


class NestedANN(nn.Module):
    def __init__(self):
        super().__init__()

        self.main = nn.Sequential(ANN(), nn.ReLU())

    def forward(self, x):
        return self.main(x)


def generate_network(backend):
    ann = NestedANN()
    input_shape = (2, 128, 128)
    return from_model(
        ann,
        input_shape=input_shape,
        threshold=0.08,
        threshold_low=-0.12,
        bias_rescaling=1.3,
        batch_size=2,
        synops=True,
        backend=backend,
    )


def compare_networks(net0, net1):
    for module0, module1 in zip(net0.modules(), net1.modules()):
        for p0, p1 in zip(module0.parameters(), module1.parameters()):
            assert (p0 == p1).all()
        for b0, b1 in zip(module0.buffers(), module1.buffers()):
            assert (b0 == b1).all()
        if hasattr(module0, "_param_dict"):
            neuron_params1 = module1._param_dict
            for key, val in module0._param_dict.items():
                if key in neuron_params1:
                    assert neuron_params1[key] == val


def test_sinabs_slyr_sinabs():
    """Convert network from sinabs to slayer and back to sinabs"""

    snn_orig = generate_network("sinabs")

    snn_copy = deepcopy(snn_orig)
    snn_slayer = snn_copy.to_backend("slayer")
    # Networks convert in-place
    assert snn_copy is snn_slayer

    snn_slayer_copy = deepcopy(snn_slayer)
    snn_sinabs = snn_slayer_copy.to_backend("sinabs")
    # Networks convert in-place
    assert snn_slayer_copy is snn_sinabs

    # Make sure parameters have not changed
    compare_networks(snn_orig, snn_slayer)
    compare_networks(snn_orig, snn_sinabs)

    # Make sure all networks produce the same output
    input_data = torch.rand([2, 2, 128, 128])
    output_orig = snn_orig(input_data)
    output_sinabs = snn_sinabs(input_data)
    output_slayer = snn_slayer(input_data)

    assert (output_orig == output_sinabs).all()
    assert (output_orig == output_slayer).all()


def test_slyr_sinabs_slyr():
    """Convert network from slayer to sinabs and back to slayer"""

    snn_orig = generate_network("slayer")

    snn_copy = deepcopy(snn_orig)
    snn_sinabs = snn_copy.to_backend("sinabs")
    # Networks convert in-place
    assert snn_copy is snn_sinabs

    snn_sinabs_copy = deepcopy(snn_sinabs)
    snn_slayer = snn_sinabs_copy.to_backend("slayer")
    # Networks convert in-place
    assert snn_sinabs_copy is snn_slayer

    # Make sure parameters have not changed
    compare_networks(snn_orig, snn_sinabs)
    compare_networks(snn_orig, snn_slayer)

    # Make sure all networks produce the same output
    input_data = torch.rand([2, 2, 128, 128])
    output_orig = snn_orig(input_data)
    output_slayer = snn_slayer(input_data)
    output_sinabs = snn_sinabs(input_data)

    assert (output_orig == output_sinabs).all()
    assert (output_orig == output_slayer).all()
