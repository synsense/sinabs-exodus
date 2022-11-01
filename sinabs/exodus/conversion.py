import sinabs.layers as sl
import sinabs.exodus.layers as el
import torch
import sinabs

module_map = {
    sl.IAF: el.IAF,
    sl.IAFSqueeze: el.IAFSqueeze,
    sl.LIF: el.LIF,
    sl.LIFSqueeze: el.LIFSqueeze,
    sl.ExpLeak: el.ExpLeak,
    sl.ExpLeakSqueeze: el.ExpLeakSqueeze,
}


def exodus_to_sinabs(model: torch.nn.Module):
    """
    Replace all EXODUS layers with the Sinabs equivalent if available.
    This can be useful if for example you want to convert your model to a
    DynapcnnNetwork or you want to work on a machine without GPU.
    All layer attributes will be copied over.

    Parameters:
        model: The model that contains EXODUS layers.
    """

    mapping_list = [
        (
            exodus_class,
            lambda module, replacement=sinabs_class: replacement(**module.arg_dict),
        )
        for sinabs_class, exodus_class in module_map.items()
    ]
    for class_to_replace, mapper_fn in mapping_list:
        model = sinabs.conversion.replace_module(
            model, class_to_replace, mapper_fn=mapper_fn
        )
    return model


def sinabs_to_exodus(model: torch.nn.Module):
    """
    Replace all Sinabs layers with EXODUS equivalents if available.
    This will typically speed up training by a factor of 2-5.
    All layer attributes will be copied over.

    Parameters:
        model: The model that contains Sinabs layers.
    """

    mapping_list = [
        (
            sinabs_class,
            lambda module, replacement=exodus_class: replacement(**module.arg_dict),
        )
        for sinabs_class, exodus_class in module_map.items()
    ]
    for class_to_replace, mapper_fn in mapping_list:
        model = sinabs.conversion.replace_module(
            model, class_to_replace, mapper_fn=mapper_fn
        )
    return model
