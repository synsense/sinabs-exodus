import sinabs.exodus.layers as el
import sinabs.layers as sl
import torch.nn as nn
from sinabs.exodus import conversion


def test_sinabs_to_exodus_layer_replacement():
    batch_size = 12
    sinabs_model = nn.Sequential(
        nn.Conv2d(2, 8, 5, 1),
        sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
        sl.SumPool2d(2, 2),
        nn.Conv2d(8, 16, 3, 1),
        sl.IAFSqueeze(batch_size=batch_size, min_v_mem=-2),
        sl.SumPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64, 10),
    )

    exodus_model = conversion.sinabs_to_exodus(sinabs_model)

    assert type(sinabs_model[1]) == sl.IAFSqueeze
    assert type(exodus_model[1]) == el.IAFSqueeze
    assert len(sinabs_model) == len(exodus_model)
    assert exodus_model[1].min_v_mem == sinabs_model[1].min_v_mem
    assert exodus_model[4].min_v_mem == sinabs_model[4].min_v_mem
    assert exodus_model[1].batch_size == sinabs_model[1].batch_size


def test_exodus_to_sinabs_layer_replacement():
    batch_size = 12
    exodus_model = nn.Sequential(
        nn.Conv2d(2, 8, 5, 1),
        el.IAFSqueeze(batch_size=batch_size, min_v_mem=-1),
        sl.SumPool2d(2, 2),
        nn.Conv2d(8, 16, 3, 1),
        el.IAFSqueeze(batch_size=batch_size, min_v_mem=-2),
        sl.SumPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(64, 10),
    )

    sinabs_model = conversion.exodus_to_sinabs(exodus_model)

    assert type(sinabs_model[1]) == sl.IAFSqueeze
    assert type(exodus_model[1]) == el.IAFSqueeze
    assert len(exodus_model) == len(sinabs_model)
    assert exodus_model[1].min_v_mem == sinabs_model[1].min_v_mem
    assert exodus_model[4].min_v_mem == sinabs_model[4].min_v_mem
    assert exodus_model[1].batch_size == sinabs_model[1].batch_size
