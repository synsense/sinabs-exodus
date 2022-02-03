import pytest
import torch
from sinabs.exodus.spike import IntegrateAndFire
from sinabs import activation as sa


def test_integratefire():
    inp = torch.rand((2, 10), requires_grad=True, device="cuda")
    v_mem_initial = torch.zeros(2, device="cuda")
    activations_initial = torch.zeros(2, device="cuda")
    surrogate_gradient_fn = sa.Heaviside(0)

    thr = 1
    alpha = 0.9

    out, v_mem = IntegrateAndFire.apply(
        inp,
        thr,
        alpha,
        v_mem_initial,
        activations_initial,
        thr,
        -thr,
        surrogate_gradient_fn,
    )

    out.sum().backward()


def test_integratefire_backprop_vmem():
    inp = torch.rand((2, 10), requires_grad=True, device="cuda")
    v_mem_initial = torch.zeros(2, device="cuda")
    activations_initial = torch.zeros(2, device="cuda")
    surrogate_gradient_fn = sa.Heaviside(0)

    thr = 1
    alpha = 0.9

    out, v_mem = IntegrateAndFire.apply(
        inp,
        thr,
        alpha,
        v_mem_initial,
        activations_initial,
        thr,
        -thr,
        surrogate_gradient_fn,
    )

    with pytest.raises(NotImplementedError):
        # Gradients for membrane potential are not implemented
        v_mem.sum().backward()

