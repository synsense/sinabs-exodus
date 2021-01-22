import torch
from src.sinabs.slayer.kernels import psp_kernels, exp_kernel
from src.sinabs.slayer.psp import generateEpsp
from src.sinabs.slayer.spike import spikeFunction
import matplotlib.pyplot as plt
import numpy as np


# Network parameters
t_sim = 100
batch_size = 1
n_neurons = (batch_size, 1, 1)
tau_mem = 10
tau_syn = [15, 2]

device = "cuda:0"

# Input spikes
input_spikes = (torch.rand(*n_neurons, t_sim) < 0.1).float().to(device)

# Generate kernels
epsp_kernel = psp_kernels(tau_mem=tau_mem, tau_syn=tau_syn, dt=1.0).to(device)

# Generate synaptic currents
isyn = generateEpsp(input_spikes, epsp_kernel, t_sim)


# TODO: Weighted sum of synaptic currents using some linear or convolutional weights
vmem = isyn.sum(0)
print(vmem.shape)


# Spiking and learning parameters
threshold = 1000.0
tau_ref = tau_mem
tauRho = 2.0
scaleRho = 1.0
ref_kernel = (exp_kernel(tau_mem, dt=1.0)*threshold).to(device)

vmem_copy = vmem.clone()
vmem_copy = vmem_copy.reshape(*n_neurons, t_sim).contiguous()  # Expects atleast 4 dimensional tensor


plt.figure()
plt.plot(epsp_kernel.cpu().numpy().T, label="EPSP Kernels")
plt.plot(ref_kernel.cpu().numpy().T/threshold, label="refractory kernel")
plt.legend()

# Generate output spikes
output_spikes = spikeFunction.apply(vmem_copy, -ref_kernel, threshold, t_sim, tauRho, scaleRho)
print(output_spikes)


# Plot data
print(np.where(output_spikes.squeeze().cpu().numpy()))

plt.figure()
plt.plot(epsp_kernel.cpu().numpy().T, label="EPSP Kernels")
plt.plot(ref_kernel.cpu().numpy().T/threshold, label="refractory kernel")
plt.legend()

plt.figure()
plt.plot(isyn.transpose(-1, 0).cpu().numpy().squeeze(), label="Isyn")
plt.plot(vmem.cpu().numpy().reshape(-1, t_sim).T, label="Vmem")
plt.plot(vmem_copy.cpu().numpy().reshape(-1, t_sim).T, label="Vmem_copy")
plt.eventplot(np.where(output_spikes.cpu().numpy())[-1], linelengths=threshold)
plt.legend()

plt.show()
