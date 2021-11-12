import pytest


def test_backend_iaf_sinabs_to_slayer():
    from sinabs.layers import IAF, IAFSqueeze

    layer = IAF()

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_slayer_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_slayer_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_slayer_backend._param_dict.items():
        # Skip `scale_grads` and `record` attributes, which are not part of sinabs backend
        if name not in ("scale_grads", "record"):
            assert layer._param_dict[name] == val

    ## Squeezed layer

    layer = IAFSqueeze(num_timesteps=10)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_slayer_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_slayer_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_slayer_backend._param_dict.items():
        # Skip `scale_grads` and `record` attributes, which are not part of sinabs backend
        if name not in ("scale_grads", "record"):
            assert layer._param_dict[name] == val


def test_backend_iaf_slayer_to_sinabs():
    from sinabs.slayer.layers import IAF, IAFSqueeze

    layer = IAF()

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer._param_dict[name] == val

    ## Squeezed layer

    layer = IAFSqueeze(num_timesteps=10)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer._param_dict[name] == val


def test_backend_lif_sinabs_to_slayer():

    from sinabs.layers import LIF, LIFSqueeze

    layer = LIF(alpha_mem=0.8)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_slayer_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_slayer_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_slayer_backend._param_dict.items():
        # Skip attributes which are not part of sinabs backend
        if name not in ("scale_grads", "record", "window"):
            assert layer._param_dict[name] == val

    ## Squeezed layer

    layer = LIFSqueeze(alpha_mem=0.8, num_timesteps=10)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_slayer_backend = layer.to_backend("slayer")
    # Make sure object has not changed in-place
    assert layer is not layer_slayer_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_slayer_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_slayer_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_slayer_backend._param_dict.items():
        # Skip attributes which are not part of sinabs backend
        if name not in ("scale_grads", "record", "window"):
            assert layer._param_dict[name] == val


def test_backend_lif_slayer_to_sinabs():

    from sinabs.slayer.layers import LIF, LIFSqueeze

    layer = LIF(alpha_mem=0.8)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer._param_dict[name] == val

    ## Squeezed layer

    layer = LIFSqueeze(alpha_mem=0.8, num_timesteps=10)

    # Modify default parameters and buffers
    for b in layer.buffers():
        b += 1
    for p in layer.parameters():
        p += 1

    # Expect warning because slayer backend has additional neuron parameter
    # `scale_grads` that is not part of sinabs backend
    with pytest.warns(RuntimeWarning):
        layer_sinabs_backend = layer.to_backend("sinabs")
    # Make sure object has not changed in-place
    assert layer is not layer_sinabs_backend

    # Make sure that parameters and buffers have not been changed during `to_backend` call
    for p0, p1 in zip(layer_sinabs_backend.parameters(), layer.parameters()):
        assert (p0 == p1).all()
        assert p0 is not p1
    for b0, b1 in zip(layer_sinabs_backend.buffers(), layer.buffers()):
        assert (b0 == b1).all()
        assert b0 is not b1

    for name, val in layer_sinabs_backend._param_dict.items():
        assert layer._param_dict[name] == val
