def test_exp_kernel():
    from src.sinabs2.slayer.kernels import exp_kernel

    kernel = exp_kernel(10, 1.0)


    import matplotlib.pyplot as plt

    plt.plot(kernel)
    plt.show()


def test_psp_kernel():
    from src.sinabs2.slayer.kernels import psp_kernel

    kernel = psp_kernel(tau_mem=30, tau_syn=10, dt=1.0)

    import matplotlib.pyplot as plt

    plt.plot(kernel)
    plt.show()

def test_generateEpsp():
    import torch
    from src.sinabs2.slayer.psp import generateEpsp
    from src.sinabs2.slayer.kernels import psp_kernels

    device = "cuda:0"

    kernels = psp_kernels([10, 15], 10.0, 1).to(device)

    # n_syn, n_neurons, t_sim
    input_spikes = torch.rand(2, 7, 100).to(device)
    t_sim = input_spikes.shape[-1]

    vsyn = generateEpsp(input_spikes, kernels, t_sim=t_sim)
    assert vsyn.shape == (2, 7, 100)
