from .spiking_layer import IntegrateFireBase
from .iaf import IAF, IAFSqueeze
from .lif import LIF, LIFSqueeze

# from sinabs import layers

# _layers_with_backend = (IAF, IAFSqueeze, LIF, LIFSqueeze)

# for lyr in _layers_with_backend:
#     # Find equivalent sinabs layer classes by name
#     lyr_sinabs = getattr(layers, lyr.__name__)
#     # Add slayer version to sinabs layer class' external backends
#     lyr_sinabs.external_backends[lyr.backend] = lyr
#     # Add sinabs layer class to slayer version's external backends
#     lyr.external_backends[lyr_sinabs.backend] = lyr_sinabs
