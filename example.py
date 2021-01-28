import torch
from src.sinabs2.slayer.kernels import psp_kernels, exp_kernel
from src.sinabs2.slayer.psp import generateEpsp
from src.sinabs2.slayer.spike import spikeFunction
import matplotlib.pyplot as plt
import numpy as np


# Network parameters
t_sim = 100
batch_size = 1
n_neurons = (batch_size, 1, 1, 1)
tau_mem = 10
tau_syn = [15, 2]
n_syn = len(tau_syn)

device = "cuda:0"

# Input spikes
input_spikes = (torch.rand(n_syn, *n_neurons, t_sim) < 0.1).float().to(device)

# Generate kernels
epsp_kernel = psp_kernels(tau_mem=tau_mem, tau_syn=tau_syn, dt=1.0).to(device)

# Generate synaptic currents
vsyn = generateEpsp(input_spikes, epsp_kernel, t_sim)
# TODO: Weighted sum of synaptic currents using some linear or convolutional weights
vmem = vsyn.sum(0)

# Spiking and learning parameters
threshold = 5.0
tau_ref = tau_mem*5
tauRho = 2.0
scaleRho = 1.0
ref_kernel = (exp_kernel(tau_ref, dt=1.0)*threshold).to(device)

vmem_copy = vmem.clone().contiguous()
# Expects atleast 5 dimensional tensor (batch, feature, height, width, time)


plt.figure()
plt.plot(epsp_kernel.cpu().numpy().T, label="EPSP Kernels")
plt.plot(ref_kernel.cpu().numpy().T/threshold, label="refractory kernel")
plt.legend()

# Generate output spikes
output_spikes = spikeFunction(vmem_copy, -ref_kernel, threshold, tauRho, scaleRho)
# print(np.where(output_spikes.squeeze().cpu().numpy()))
print(output_spikes)


# Plot data

plt.figure()
plt.plot(epsp_kernel.cpu().numpy().T, label="EPSP Kernels")
plt.plot(ref_kernel.cpu().numpy().T/threshold, label="refractory kernel")
plt.plot(output_spikes.cpu().squeeze().numpy(), label="output_spikes")
plt.legend()

plt.figure()
plt.plot(vsyn.transpose(-1, 0).cpu().numpy().squeeze(), label="Isyn")
plt.plot(vmem.cpu().numpy().reshape(-1, t_sim).T, label="Vmem")
plt.plot(vmem_copy.cpu().numpy().reshape(-1, t_sim).T, label="Vmem_copy")
plt.eventplot(np.where(output_spikes.cpu().numpy())[-1], linelengths=threshold)
plt.legend()

plt.show()
