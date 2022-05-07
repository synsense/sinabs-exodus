/*
 * Author: Felix Bauer, loosely based on slayerPytorch library by Sumit Bam Shrestha
 * Contains routines to compute forward and backward passes
 * for the following neuron dynamics
 * - non-spiking leaky integrator
 */
#ifndef LEAKYKERNELS_H_INCLUDED
#define LEAKYKERNELS_H_INCLUDED

#include <stdio.h>

// Kernel functions


/** Forward kernel for leaky integrator dynamic.
 * 	
 * Forward evolution for (leaky) integrator dynamic, over time.
 * vmem_t = alpha * vmem_{t-1} + input_t
 *
 * @param vmem 2D-tensor (nNeurons x nTimesteps) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x nTimesteps) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void leakyForwardKernel(
	scalarType* __restrict__ vmemAll,
	const scalarType* __restrict__ input,
	const scalarType* __restrict__ vmemInitial,
	float alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	scalarType vmemCurr = vmemInitial[neuronID];

	for(unsigned t=0; t<nTimesteps; ++t){

		// Decay state
		vmemCurr *= alpha;

		// ID of neuron and current timestep
		unsigned linearID = t + neuronID * nTimesteps;

		// Add current input to vmemCurr
		vmemCurr += input[linearID];

		// Write current vmemCurr into tensor
		vmemAll[linearID] = vmemCurr;
	}

}


/** Backward kernel for leaky integrator dynamics.
 *
 * Using that 
 * \frac{dOut_j}{dIn_i} = alhpa^{j-i} if j>=i, else 0
 * and
 * gradInput_i = sum_{j=i}^{N_s-1} \frac{dOut_j}{dIn_i} gradOutput_j
 * = sum_{j=i}^{N_s-1} alpha^{j-i} * gradOutput_j
 * = alpha * sum_{j=i+1}^{N_s-1} alpha^{j-(i+1)} * gradOutput_j + gradOutput_i
 * gradInput can be calculated recursively as 
 * gradInput_i = alpha * gradInput_{i+1} + gradOutput_i
 * gradInput_{N_s-1} = gradOutput_{N_s-1}
 *
 * @param vmem 2D-tensor (nNeurons x nTimesteps) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x nTimesteps) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void leakyBackwardKernel(
	scalarType* __restrict__ gradInput,
	const scalarType* __restrict__ gradOutput,
	float alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	scalarType grad_curr = 0;

	for(unsigned t=nTimesteps-1; t<nTimesteps; --t){

		// ID of neuron and current timestep
		unsigned linearID = t + neuronID * nTimesteps;

		// Add corresponding element of gradOutput and multiply by alpha
		grad_curr = grad_curr * alpha + gradOutput[linearID];

		// Write current grad into gradInput
		gradInput[linearID] = grad_curr;
	}

}


// Host functions


/** Forward pass for exponential leak
 *
 * v_t = alpha * v_{t-1} + I_t
 * Parallelize across neurons/batches
 */
template <class scalarType>
void leakyForwardCuda(
	scalarType* vmemAll,
	const scalarType* input,
	const scalarType* vmemInitial,
	float alpha, unsigned nNeurons, unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyForwardKernel<scalarType><<< block, thread >>>(
			vmemAll,
			input,
			vmemInitial,
			alpha, nNeurons, nTimesteps);
}

/** Backward pass for exponential leak
 *
 * v_t = alpha * v_{t-1} + I_t
 * Parallelize across neurons/batches
 */
template <class scalarType>
void leakyBackwardCuda(
	scalarType* inputGrad,
	const scalarType* outputGrad,
	float alpha, unsigned nNeurons, unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyBackwardKernel<scalarType><<< block, thread >>>(
			inputGrad,
			outputGrad,
			alpha, nNeurons, nTimesteps);
}


#endif // LEAKYKERNELS_H_INCLUDED
