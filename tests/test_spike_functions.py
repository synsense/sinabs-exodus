import pytest
from itertools import product
import torch
from sinabs.exodus.spike import IntegrateAndFire
from sinabs import activation as sa
from sinabs.layers.functional.lif import lif_forward


v_mem_initials = (torch.zeros(2).cuda(), torch.rand(2).cuda() - 0.5)
alphas = (torch.ones(2).cuda() * 0.9, torch.rand(2).cuda())
thresholds = (torch.ones(2).cuda(), torch.tensor([0.3, 0.9]))
min_v_mem = (None, -torch.ones(2).cuda(), torch.tensor([-0.3, 0.4]).cuda())
membrane_subtract = (None, torch.tensor([0.1, 0.2]).cuda())

argvals = (v_mem_initials, alphas, thresholds, min_v_mem, membrane_subtract)
combined_args = product(*argvals)
argnames = "v_mem_initial,alpha,threshold,min_v_mem,membrane_subtract"

@pytest.mark.parametrize(argnames, combined_args)
def test_integratefire(
    v_mem_initial, alpha, threshold, min_v_mem, membrane_subtract
):
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

    if (membrane_subtract >= threshold).any():
        with pytest.raises(ValueError):
            apply()
    else:
        # Test forward pass and backpropagation through output
        out, v_mem = apply()
        out.sum().backward()

        # Make sure backprop through vmem raises error
        out, v_mem = apply()
        with pytest.raises(NotImplementedError):
            # Gradients for membrane potential are not implemented
            v_mem.sum().backward()


backward_varnames = ("spikes", "vmem", "sum")
argvals_ext = (*argvals, backward_varnames)
combined_args_ext = product(*argvals_ext)
@pytest.mark.parametrize(argnames + ",backward_var", combined_args_ext)
def test_compare_integratefire_backward(
    v_mem_initial, alpha, threshold, min_v_mem, membrane_subtract
):

    torch.manual_seed(1)
    
    time_steps = 100
    n_neurons = 2
    surrogate_gradient_fn = sa.Heaviside(0)
    max_num_spikes_per_bin = 2
    
    # Input data and initialization 
    input_sinabs = torch.rand((time_steps, n_neurons), requires_grad=True, device= "cuda")
    v_mem_init_sinabs = v_mem_initial.clone().requires_grad_(True)
    alpha_sinabs = alpha.clone().requires_grad_(True)

    # Copy data without connecting gradients
    input_exodus = input_sinabs.clone().detach().requires_grad_(True)
    v_mem_init_exodus = v_mem_initial.clone().requires_grad_(True)
    alpha_exodus = alpha.clone().requires_grad_(True)

    out_exodus, vmem_exodus = evolve_exodus(
        inp=input_exodus,
        alpha=alpha_exodus,
        v_mem_init=v_mem_init_exodus,
        threshold=threshold,
        min_v_mem=min_v_mem,
        surrogate_grad_fn=surrogate_gradient_fn,
        max_num_spikes_per_bin=max_num_spikes_per_bin,
        membrane_subtract=membrane_subtract,
    )

    out_sinabs, vmem_sinabs = evolve_sinabs(
        inp=input_sinabs,
        alpha=alpha_sinabs,
        v_mem_init=v_mem_init_sinabs,
        threshold=thr,
        min_v_mem=min_v_mem,
        surrogate_grad_fn=surrogate_gradient_fn,
        max_num_spikes_per_bin=max_num_spikes_per_bin,
        membrane_subtract=membrane_subtract,
    )

    assert torch.allclose(out_exodus, out_sinabs)
    assert torch.allclose(vmem_exodus, vmem_sinabs)

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
        (out_exodus * w_out + vmem_exodus * w_vmem).sum().backward()
        (out_sinabs * w_out + vmem_sinabs * w_vmem).sum().backward()

    assert torch.allclose(input_sinabs.grad, input_exodus.grad)
    assert torch.allclose(v_mem_init_sinabs.grad, v_mem_init_exodus.grad)
    assert torch.allclose(alpha_sinabs.grad, alpha_exodus.grad)


def evolve_exodus(
    inp: torch.tensor,
    alpha: torch.tensor,
    v_mem_init: torch.tensor,
    threshold: torch.tensor,
    min_v_mem: torch.tensor,
    surrogate_grad_fn,
    max_num_spikes_per_bin = None,
    membrane_subtract = None,
):
    if membrane_subtract is None:
        membrane_subtract = torch.ones_like(v_mem_init) * threshold

    output_spikes, v_mem = IntegrateAndFire.apply(
        inp,
        alpha,
        v_mem_init,
        threshold,
        membrane_subtract,
        min_v_mem,
        surrogate_grad_fn,
        max_num_spikes_per_bin,
    )

    v_mem = v_mem - membrane_subtract.unsqueeze(0) * output_spikes

    return output_spikes, v_mem

def evolve_sinabs(
    inp: torch.tensor,
    alpha: torch.tensor,
    v_mem_init: torch.tensor,
    threshold: torch.tensor,
    min_v_mem: torch.tensor,
    surrogate_grad_fn, 
    max_num_spikes_per_bin = None,
    membrane_subtract = None,
):
    if max_num_spikes_per_bin is not None:
        spike_fn = sa.MaxSpike(max_num_spikes_per_bin)
    else:
        spike_fn = sa.MultiSpike

    output_spikes, state, record_dict = lif_forward(
        input_data=inp.unsqueeze(0), # Add batch dimension
        alpha_mem=alpha,
        alpha_syn=None,
        state={"v_mem": v_mem_init},
        spike_threshold=threshold,
        spike_fn=spike_fn,
        reset_fn=sa.MembraneSubtract(subtract_value=membrane_subtract),
        surrogate_grad_fn=surrogate_grad_fn,
        min_v_mem=min_v_mem,
        norm_input=False,
        record_states=True
    )

    return output_spikes, record_dict["v_mem"]
