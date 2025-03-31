import pytest
from itertools import product
import torch
from sinabs.exodus.spike import IntegrateAndFire
from sinabs import activation as sa
from sinabs.layers.functional.lif import lif_forward


v_mem_initials = (torch.zeros(2).cuda(), torch.rand(2).cuda() - 0.5)
alphas = (torch.ones(2).cuda() * 0.9, torch.rand(2).cuda())
thresholds = (torch.ones(2).cuda(), torch.tensor([0.3, 0.9]).cuda())
min_v_mem = (
    None,
    -torch.ones(2).cuda(),
    torch.tensor([-0.3, -0.4]).cuda(),
    torch.tensor([1.2, 0]).cuda(),
)
membrane_subtract = (None, torch.tensor([0.1, 0.2]).cuda())

argvals = (v_mem_initials, alphas, thresholds, min_v_mem, membrane_subtract)
combined_args = product(*argvals)
argnames = "v_mem_initial,alpha,threshold,min_v_mem,membrane_subtract"


@pytest.mark.parametrize(argnames, combined_args)
def test_integratefire(v_mem_initial, alpha, threshold, min_v_mem, membrane_subtract):
    inp = torch.rand((2, 10), requires_grad=True, device="cuda")
    surrogate_gradient_fn = sa.Heaviside(0)

    if membrane_subtract is None:
        membrane_subtract = threshold

    def apply():
        return IntegrateAndFire.apply(
            inp,
            alpha,
            v_mem_initial,
            threshold,
            membrane_subtract,
            min_v_mem,
            surrogate_gradient_fn,
        )

    if min_v_mem is not None and (min_v_mem >= threshold).any():
        with pytest.raises(ValueError):
            apply()
    else:
        # Test forward pass and backpropagation through output
        out, v_mem = apply()
        out.sum().backward()

        # Test forward pass and backpropagation through v_mem
        out, v_mem = apply()
        v_mem.sum().backward()


backward_varnames = ("spikes", "vmem", "sum")
argvals_ext = (
    v_mem_initials,
    alphas,
    thresholds,
    min_v_mem[:-1],  # Avoid min_v_mem > thr error
    membrane_subtract,
    backward_varnames,
)
combined_args_ext = product(*argvals_ext)


@pytest.mark.parametrize(argnames + ",backward_var", combined_args_ext)
def test_compare_integratefire(
    v_mem_initial, alpha, threshold, min_v_mem, membrane_subtract, backward_var
):

    torch.manual_seed(1)

    time_steps = 100
    batchsize = 10
    n_neurons = 2
    num_epochs = 3
    surrogate_grad_fn = sa.PeriodicExponential()
    max_num_spikes_per_bin = 2

    # Input data and initialization
    input_sinabs = torch.rand(
        (num_epochs, batchsize, time_steps, n_neurons),
        requires_grad=True,
        device="cuda",
    )
    v_mem_init_sinabs = v_mem_initial.clone().requires_grad_(True)
    alpha_sinabs = alpha.clone().requires_grad_(True)

    # Copy data without connecting gradients
    input_exodus = input_sinabs.clone().detach().requires_grad_(True)
    v_mem_init_exodus = v_mem_initial.clone().requires_grad_(True)
    alpha_exodus = alpha.clone().requires_grad_(True)

    out_exodus, vmem_exodus = evolve_exodus(
        data=input_exodus,
        alpha=alpha_exodus,
        v_mem_init=v_mem_init_exodus,
        threshold=threshold,
        min_v_mem=min_v_mem,
        surrogate_grad_fn=surrogate_grad_fn,
        max_num_spikes_per_bin=max_num_spikes_per_bin,
        membrane_subtract=membrane_subtract,
    )

    out_sinabs, vmem_sinabs = evolve_sinabs(
        data=input_sinabs,
        alpha=alpha_sinabs,
        v_mem_init=v_mem_init_sinabs,
        threshold=threshold,
        min_v_mem=min_v_mem,
        surrogate_grad_fn=surrogate_grad_fn,
        max_num_spikes_per_bin=max_num_spikes_per_bin,
        membrane_subtract=membrane_subtract,
    )

    assert torch.allclose(out_exodus, out_sinabs)
    assert torch.allclose(vmem_exodus, vmem_sinabs, rtol=1e-4, atol=1e-8)

    if backward_var == "spikes":
        # random weights so the output grad tensor is less uniform
        rand_weights = torch.rand_like(out_exodus)
        # - Test backward pass through output spikes
        (rand_weights * out_exodus).sum().backward()
        (rand_weights * out_sinabs).sum().backward()
    elif backward_var == "vmem":
        # random weights so the output grad tensor is less uniform
        rand_weights = torch.rand_like(vmem_exodus)
        # - Test backward pass through membrane potential
        (rand_weights * vmem_exodus).sum().backward()
        (rand_weights * vmem_sinabs).sum().backward()
    elif backward_var == "sum":
        # random weights so the output grad tensor is less uniform
        w_out = torch.rand_like(out_exodus)
        w_vmem = torch.rand_like(vmem_exodus)
        # - Test backward pass through membrane potential
        (out_exodus * w_out + (vmem_exodus * w_vmem).unsqueeze(1)).sum().backward()
        (out_sinabs * w_out + (vmem_sinabs * w_vmem).unsqueeze(1)).sum().backward()

    assert torch.allclose(input_sinabs.grad, input_exodus.grad, rtol=1e-4, atol=1e-6)
    assert torch.allclose(
        v_mem_init_sinabs.grad, v_mem_init_exodus.grad, rtol=1e-4, atol=1e-6
    )
    assert torch.allclose(alpha_sinabs.grad, alpha_exodus.grad, rtol=1e-4, atol=1e-6)


def evolve_exodus(
    data: torch.tensor,
    alpha: torch.tensor,
    v_mem_init: torch.tensor,
    threshold: torch.tensor,
    min_v_mem: torch.tensor,
    surrogate_grad_fn,
    max_num_spikes_per_bin=None,
    membrane_subtract=None,
):
    # This normally happens inside the Exodus LIF layer,
    # which is being circumvented here
    batchsize, timesteps, *trailing_dim = data.shape[1:]
    expanded_shape = (batchsize, *trailing_dim)
    alpha = alpha.expand(expanded_shape).flatten().contiguous()
    v_mem_init = v_mem_init.expand(expanded_shape).flatten().contiguous()
    threshold = threshold.expand(expanded_shape).flatten().contiguous()
    if min_v_mem is not None:
        min_v_mem = min_v_mem.expand(expanded_shape).flatten().contiguous()

    if membrane_subtract is None:
        membrane_subtract = threshold
    else:
        membrane_subtract = (
            membrane_subtract.expand(expanded_shape).flatten().contiguous()
        )

    for inp in data:
        inp = inp.movedim(1, -1).reshape(-1, timesteps)
        output_spikes, v_mem = IntegrateAndFire.apply(
            inp.contiguous(),
            alpha,
            v_mem_init,
            threshold,
            membrane_subtract,
            min_v_mem,
            surrogate_grad_fn,
            max_num_spikes_per_bin,
        )

        v_mem = v_mem - membrane_subtract.unsqueeze(1) * output_spikes
        v_mem_init = v_mem[:, -1].contiguous()

    return (
        output_spikes.reshape(batchsize, *trailing_dim, timesteps).movedim(-1, 1),
        v_mem[:, -1].reshape(batchsize, *trailing_dim),
    )


def evolve_sinabs(
    data: torch.tensor,
    alpha: torch.tensor,
    v_mem_init: torch.tensor,
    threshold: float,
    min_v_mem: float,
    surrogate_grad_fn,
    max_num_spikes_per_bin=None,
    membrane_subtract=None,
):
    if max_num_spikes_per_bin is not None:
        spike_fn = sa.MaxSpike(max_num_spikes_per_bin)
    else:
        spike_fn = sa.MultiSpike

    state = {"v_mem": v_mem_init}

    for inp in data:
        # Add batch dimension and move time to dimension 1

        output_spikes, state, record_dict = lif_forward(
            input_data=inp,
            alpha_mem=alpha,
            alpha_syn=None,
            state=state,
            spike_threshold=threshold,
            spike_fn=spike_fn,
            reset_fn=sa.MembraneSubtract(subtract_value=membrane_subtract),
            surrogate_grad_fn=surrogate_grad_fn,
            min_v_mem=min_v_mem,
            norm_input=False,
            record_states=True,
        )

    # Squeeze batch dimensions and move time to last
    # return output_spikes.squeeze(0).movedim(-1,0), record_dict["v_mem"].squeeze(0).movedim(-1, 0)[:, -1]
    return output_spikes, state["v_mem"]
