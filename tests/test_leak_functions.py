import pytest
import torch
from sinabs.exodus.leaky import LeakyIntegrator
from sinabs.layers.functional.lif import lif_forward


def test_leakyintegrator():

    torch.random.manual_seed(1)

    inp = torch.rand((2, 10), requires_grad=True, device="cuda")
    v_mem_initial = torch.zeros(2, device="cuda", requires_grad=True)

    alpha = torch.rand_like(v_mem_initial, requires_grad=True).contiguous().cuda()

    out = LeakyIntegrator.apply(
        inp,
        alpha,
        v_mem_initial,
    )

    out.sum().backward()

    for p in (inp, alpha, v_mem_initial):
        assert p.grad is not None


def test_compare_leakyintegrator_backward():

    torch.manual_seed(1)

    time_steps = 100
    batchsize = 10
    n_neurons = 8
    num_epochs = 3

    # Input data and initialization
    input_sinabs = torch.rand(
        (num_epochs, batchsize, time_steps, n_neurons),
        requires_grad=True,
        device="cuda",
    )
    v_mem_init_sinabs = torch.rand(
        batchsize, n_neurons, requires_grad=True, device="cuda"
    )
    alpha_sinabs = torch.rand(n_neurons, requires_grad=True, device="cuda")

    # Copy data without connecting gradients
    input_exodus = input_sinabs.clone().detach().requires_grad_(True)
    v_mem_init_exodus = v_mem_init_sinabs.clone().detach().requires_grad_(True)
    alpha_exodus = alpha_sinabs.clone().detach().requires_grad_(True)

    out_exodus = evolve_exodus(
        data=input_exodus,
        alpha=alpha_exodus,
        v_mem_init=v_mem_init_exodus,
    )

    out_sinabs = evolve_sinabs(
        data=input_sinabs,
        alpha=alpha_sinabs,
        v_mem_init=v_mem_init_sinabs,
    )

    assert torch.allclose(out_exodus, out_sinabs)

    # random weights so the output grad tensor is less uniform
    rand_weights = torch.rand_like(out_exodus)
    # - Test backward pass through output spikes
    (rand_weights * out_exodus).sum().backward()
    (rand_weights * out_sinabs).sum().backward()

    assert torch.allclose(input_sinabs.grad, input_exodus.grad)
    assert torch.allclose(v_mem_init_sinabs.grad, v_mem_init_exodus.grad)
    assert torch.allclose(alpha_sinabs.grad, alpha_exodus.grad)


def evolve_exodus(
    data: torch.tensor,
    alpha: torch.tensor,
    v_mem_init: torch.tensor,
):
    alpha = alpha.expand(v_mem_init.shape).flatten().contiguous()
    v_mem_init = v_mem_init.flatten().contiguous()
    batchsize, timesteps, *trailing_dim = data.shape[1:]

    for inp in data:
        inp = inp.movedim(1, -1).reshape(-1, timesteps)
        output = LeakyIntegrator.apply(
            inp.contiguous(),
            alpha,
            v_mem_init,
        )

        v_mem_init = output[:, -1].contiguous()

    return output.reshape(batchsize, *trailing_dim, timesteps).movedim(-1, 1)


def evolve_sinabs(
    data: torch.tensor,
    alpha: torch.tensor,
    v_mem_init: torch.tensor,
):
    state = {"v_mem": v_mem_init}

    for inp in data:
        # Add batch dimension and move time to dimension 1

        output, state, *__ = lif_forward(
            input_data=inp,
            alpha_mem=alpha,
            alpha_syn=None,
            state=state,
            spike_threshold=None,
            spike_fn=None,
            reset_fn=None,
            surrogate_grad_fn=None,
            min_v_mem=None,
            norm_input=False,
            record_states=False,
        )

    return output
