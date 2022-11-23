import pytest
import torch
from sinabs.exodus.spike import IntegrateAndFire
from sinabs import activation as sa
from sinabs.layers.functional.lif import lif_forward


def test_integratefire():
    inp = torch.rand((2, 10), requires_grad=True, device="cuda")
    v_mem_initial = torch.zeros(2, device="cuda")
    surrogate_gradient_fn = sa.Heaviside(0)

    thr = 1.0
    alpha = torch.as_tensor(0.9).expand(v_mem_initial.shape).contiguous().cuda()
    membrane_subtract = torch.as_tensor(thr).expand(v_mem_initial.shape)

    out, v_mem = IntegrateAndFire.apply(
        inp,
        alpha,
        v_mem_initial,
        thr,
        membrane_subtract.contiguous().float().cuda(),
        -thr,
        surrogate_gradient_fn,
    )

    out.sum().backward()


def test_integratefire_backprop_vmem():
    inp = torch.rand((2, 10), requires_grad=True, device="cuda")
    v_mem_initial = torch.zeros(2, device="cuda")
    surrogate_gradient_fn = sa.Heaviside(0)

    thr = 1
    alpha = torch.as_tensor(0.9).expand(v_mem_initial.shape).contiguous().cuda()
    membrane_subtract = torch.as_tensor(thr).expand(v_mem_initial.shape)

    out, v_mem = IntegrateAndFire.apply(
        inp,
        alpha,
        v_mem_initial,
        thr,
        membrane_subtract.contiguous().float().cuda(),
        -thr,
        surrogate_gradient_fn,
    )

    with pytest.raises(NotImplementedError):
        # Gradients for membrane potential are not implemented
        v_mem.sum().backward()

args = ("spikes", "vmem", "sum")
@pytest.mark.parametrize("backward_var", args)
def test_compare_integratefire_backward(backward_var):
    time_steps = 100
    n_neurons = 8
    thr = 1
    min_v_mem = -1
    surrogate_gradient_fn = sa.Heaviside(0)
    max_num_spikes_per_bin = 2
    
    # Input data and initialization 
    input_sinabs = torch.rand((time_steps, n_neurons), requires_grad=True, device= "cuda")
    v_mem_init_sinabs = torch.rand(n_neurons, requires_grad=True, device= "cuda")
    alpha_sinabs = torch.rand(n_neurons, requires_grad=True, device= "cuda")

    # Copy data without connecting gradients
    input_exodus = input_sinabs.clone().detach().requires_grad_(True)
    v_mem_init_exodus = v_mem_init_sinabs.clone().detach().requires_grad_(True)
    alpha_exodus = alpha_sinabs.clone().detach().requires_grad_(True)

    out_exodus, vmem_exodus = evolve_exodus(
        inp=input_exodus,
        alpha=alpha_exodus,
        v_mem_init=v_mem_init_exodus,
        threshold=thr,
        min_v_mem=min_v_mem,
        surrogate_grad_fn=surrogate_gradient_fn,
        max_num_spikes_per_bin=max_num_spikes_per_bin,
    )

    out_sinabs, vmem_sinabs = evolve_sinabs(
        inp=input_sinabs,
        alpha=alpha_sinabs,
        v_mem_init=v_mem_init_sinabs,
        threshold=thr,
        min_v_mem=min_v_mem,
        surrogate_grad_fn=surrogate_gradient_fn,
        max_num_spikes_per_bin=max_num_spikes_per_bin,
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
    threshold: float,
    min_v_mem: float,
    surrogate_grad_fn,
    max_num_spikes_per_bin = None,
):
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
    threshold: float,
    min_v_mem: float,
    surrogate_grad_fn, 
    max_num_spikes_per_bin = None,
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
        reset_fn=sa.MembraneSubtract(),
        surrogate_grad_fn=surrogate_grad_fn,
        min_v_mem=min_v_mem,
        norm_input=False,
        record_states=True
    )

    return output_spikes, record_dict["v_mem"]
