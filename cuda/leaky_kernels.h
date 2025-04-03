/*
 * This file contains routines to compute forward and backward passes
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
 * @param alhpa 1D-tensor (nNeurons) with decay factor of the neuron state (exp(-dt/tau)).
 * 		  For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void leakyForwardKernel(
	scalarType* __restrict__ vmemAll,
	const scalarType* __restrict__ input,
	const scalarType* __restrict__ vmemInitial,
	const scalarType* __restrict__ alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	scalarType vmemCurr = vmemInitial[neuronID];

	for(unsigned t=0; t<nTimesteps; ++t){

		// Decay state
		vmemCurr *= alpha[neuronID];

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
 * inputGrad_i = \sum_{j=i}^{N_s-1} \frac{dOut_j}{dIn_i} outputGrad_j
 * = \sum_{j=i}^{N_s-1} alpha^{j-i} * outputGrad_j
 * = alpha * \sum_{j=i+1}^{N_s-1} alpha^{j-(i+1)} * outputGrad_j + outputGrad_i
 * inputGrad can be calculated recursively as
 * inputGrad_i = alpha * inputGrad_{i+1} + outputGrad_i
 * inputGrad_{N_s-1} = outputGrad_{N_s-1}
 *
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed input gradients
 * 		  are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) with the output gradients
 * @param alhpa 1D-tensor (nNeurons) with decay factor of the neuron state (exp(-dt/tau)).
 * 		  For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void leakyBackwardKernel(
	scalarType* __restrict__ inputGrad,
	const scalarType* __restrict__ outputGrad,
	const scalarType* __restrict__ alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

    // Index of first element in current row of 2D tensors (i.e. for current neuron)
    unsigned linearRowID = neuronID * nTimesteps;

	scalarType grad = 0;

	for(unsigned t=nTimesteps-1; t<nTimesteps; --t){

		// ID of neuron and current timestep
		unsigned tIndex = t + linearRowID;

		// Add corresponding element of outputGrad and multiply by alpha
		grad = grad * alpha[neuronID] + outputGrad[tIndex];

		// Write current grad into inputGrad
		inputGrad[tIndex] = grad;
	}

}


/** Backward kernel for alpha gradients in leaky integrator dynamics.
 *
 * The gradients are given by
 * \frac{d Out_t}{d \alpha} = \sum_{i=0}^{t-1} (t-i) \alpha^{t-i-1} In_i ,
 * where In_0 is vmemInitial. The gradients can be calculated recursively as
 * \frac{d Out_1}{d \alpha} = vmemInitial
 * \frac{d Out_{t}}{d \alpha} = \alpha * frac{d Out_{t-1}{d \alpha} + Out_{t-1}
 * because Out_{t} = \sum_{i=0}^{T} \alpha^{t-i} * In_i
 *
 * @param alphaGrad 1D-tensor (nNeurons) to which the computed alpha gradients
 * 		  are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) with the output gradients
 * @param output 2D-tensor (nNeurons x nTimesteps) with the output of the forward pass
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param alhpa 1D-tensor (nNeurons) with decay factor of the neuron state (exp(-dt/tau)).
 * 		  For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void leakyBackwardAlphaKernel(
	scalarType* __restrict__ alphaGrad,
	const scalarType* __restrict__ outputGrad,
	const scalarType* __restrict__ output,
	const scalarType* __restrict__ vmemInitial,
	const scalarType* __restrict__ alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (i.e. for current neuron)
	unsigned linearRowID = neuronID * nTimesteps;

	// At t=0, gradient is vmemInitial
	scalarType grad = vmemInitial[neuronID];
	alphaGrad[neuronID] = grad * outputGrad[linearRowID];

	for(unsigned t=1; t<nTimesteps; ++t){

		// 2D-index of neuron and current timestep
		unsigned tIndex = t + linearRowID;

		// Scale previous grad with alpha and add output at previous timestep
		grad = alpha[neuronID] * grad + output[tIndex - 1];

		// Add corresponding element of outputGrad and multiply by alpha
		alphaGrad[neuronID] += grad * outputGrad[tIndex];
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
	const scalarType* alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyForwardKernel<scalarType><<< block, thread >>>(
			vmemAll,
			input,
			vmemInitial,
			alpha,
			nNeurons, nTimesteps);
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
	const scalarType* alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyBackwardKernel<scalarType><<< block, thread >>>(
			inputGrad,
			outputGrad,
			alpha,
			nNeurons, nTimesteps);
}

/** Backward pass for exponential leak to get alpha gradients
 *
 * v_t = alpha * v_{t-1} + I_t
 * Parallelize across neurons/batches
 */
template <class scalarType>
void leakyBackwardAlphaCuda(
	scalarType* alphaGrad,
	const scalarType* outputGrad,
	const scalarType* output,
	const scalarType* vmemInitial,
	const scalarType* alpha,
	unsigned nNeurons,
	unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyBackwardAlphaKernel<scalarType><<< block, thread >>>(
			alphaGrad,
			outputGrad,
			output,
			vmemInitial,
			alpha,
			nNeurons, nTimesteps);
}


#endif // LEAKYKERNELS_H_INCLUDED
