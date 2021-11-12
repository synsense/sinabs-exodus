import pytest
import torch

input_data = torch.randint(10, size=(2, 10, 3, 4, 5))
input_data_squeezed = input_data.view((20, 3, 4, 5))


def compare_layers(lyr0, lyr1):
    for p0, p1 in zip(lyr0.parameters(), lyr1.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(lyr0.buffers(), lyr1.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    neuron_params1 = lyr1._param_dict
    for name, val in lyr0._param_dict.items():
        # Skip attributes which are not part of the other backend
        if name in neuron_params1:
            assert neuron_params1[name] == val


def test_backend_iaf_sinabs_to_slayer():
    from sinabs.layers import IAF, IAFSqueeze

    layer = IAF()

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_slayer_backend, layer)

    ## Squeezed layer

    layer = IAFSqueeze(num_timesteps=10)

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data_squeezed)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_slayer_backend, layer)


def test_backend_iaf_slayer_to_sinabs():
    from sinabs.slayer.layers import IAF, IAFSqueeze

    layer = IAF()

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_sinabs_backend, layer)

    ## Squeezed layer

    layer = IAFSqueeze(num_timesteps=10)

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data_squeezed)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_sinabs_backend, layer)


def test_backend_lif_sinabs_to_slayer():

    from sinabs.layers import LIF, LIFSqueeze

    layer = LIF(alpha_mem=0.8)

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_slayer_backend, layer)

    ## Squeezed layer

    layer = LIFSqueeze(alpha_mem=0.8, num_timesteps=10)

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data_squeezed)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_slayer_backend, layer)


def test_backend_lif_slayer_to_sinabs():

    from sinabs.slayer.layers import LIF, LIFSqueeze

    layer = LIF(alpha_mem=0.8)

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_sinabs_backend, layer)

    ## Squeezed layer

    layer = LIFSqueeze(alpha_mem=0.8, num_timesteps=10)

    # Modify default parameters and buffers
    # for b in layer.buffers():
    #     b += 1
    for p in layer.parameters():
        p += 1

    layer.forward(input_data_squeezed)

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    compare_layers(layer_sinabs_backend, layer)
