class IAFLayer():
    def __init__(self):
        pass


    def epsp_kernel(self):
        pass

    def ref_kernel(self):
        pass


class SpikingLayer:
    def __init__(self):
        # TODO: load params
        pass

    def forward(self, vmem):
        t_sim = len(vmem)
        output_spikes = spikeFunction.apply(vmem, -self.ref_kernel, self.threshold, t_sim, self.tauRho, self.scaleRho)
