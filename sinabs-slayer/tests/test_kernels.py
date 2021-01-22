def test_exp_kernel():
    from src.sinabs.slayer.kernels import exp_kernel

    kernel = exp_kernel(10, 1.0)


    import matplotlib.pyplot as plt

    plt.plot(kernel)
    plt.show()


def test_psp_kernel():
    from src.sinabs.slayer.kernels import psp_kernel

    kernel = psp_kernel(tau_mem=30, tau_syn=10, dt=1.0)

    import matplotlib.pyplot as plt

    plt.plot(kernel)
    plt.show()
