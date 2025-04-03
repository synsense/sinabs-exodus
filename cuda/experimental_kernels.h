/*
 * This file contain routines in experimental stage to compute forward and backward passes
 * for the following neuron dynamics
 * - leaky and non-leaky integrate-and-fire (different implementation)
 * - spike-function for given membrane potential
 */
#ifndef EXPERIMENTALKERNELS_H_INCLUDED
#define EXPERIMENTALKERNELS_H_INCLUDED

#include <stdio.h>

// Kernel functions

/** Spike generation forward kernel */
template <class scalarType>
__global__ void spikeForwardKernel(
	scalarType* __restrict__ d_s,
	scalarType* __restrict__ d_u,
	float alpha,
	float membrSubtract,
	unsigned nNeurons,
	unsigned nTimesteps,
	float theta,
	float theta_low,
	bool applyThetaLow,
	unsigned maxNumSpikes)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	float reset_decay;

	for(unsigned i=0; i<nTimesteps; ++i)
	{
		// ID of neuron and current timestep
		unsigned linearID = i + neuronID * nTimesteps;

		// Spike mechanism
		if(d_u[linearID] >= theta)
		{
            unsigned activation = min(unsigned(d_u[linearID] / theta), maxNumSpikes);
			d_s[linearID] = static_cast<float>(activation);

			// Reset mechanism
			reset_decay = alpha;
			for(unsigned j=1; j<nTimesteps; ++j)
			{
				if(i + j < nTimesteps)	d_u[linearID + j] -= reset_decay * membrSubtract * activation;
				reset_decay *= alpha;
			}

		// Lower bound
		} else if(applyThetaLow && (d_u[linearID] < theta_low))
		{
			float difference = theta_low - d_u[linearID];
			for(unsigned j=1; j<nTimesteps; ++j)
			{
				difference *= alpha;
				if(i + j < nTimesteps) d_u[linearID + j] += difference;
			}
		}
	}

}


/** Spike generation backward kernel
 *
 * Assuming a function that calculates the output spikes of a SRM-neuron (exponential
 * refractory response, arbitrary surrogate gradients) for a given (synaptic) input that
 * has already been convolved with the input-spike response kernel, this kernel computes
 * a single element (corresponding to one time step) of the the input gradient for
 * one neuron and/or batch. It amounts to the scalar product of the output gradient with
 * the derivative of the spike output wrt. the (pre-spiking) membrane potential at the
 * i-th timestep.
 *
 * inputGrad_i = \sum_{j=i}^{N_s - 1} outputGrad_j * dOutput_j / dV_i
 * 			   = surr_i * outputGrad_i +
 * 				 \sum_{j=i+1}^{N_s - 1} outputGrad_j *
 * 				 * (-g_i) * g_j * \prod_{k=i+1}^{j - 1} beta_k
 * where gamma_i = surr_i * notClipped_i
 * and beta_i = (alpha - membrSubtract_i * surr_i) * notClipped_i - 1
 *
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
__global__ void spikeBackwardKernel(
	scalarType* __restrict__ inputGrad,
	const scalarType* __restrict__ outputGrad,
	const scalarType* __restrict__ surr,
	const scalarType* __restrict__ notClipped,
	float alpha, float membrSubtract, unsigned nNeurons, unsigned nTimesteps)
{
	// Identifier corresponding to the element of the input gradient that is
	// computed as well as the denominator in the derivatives
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= nTimesteps)	return;

	// Identifier for the current neuron and/or batch
	unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;
	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (i.e. for current neuron)
	unsigned linearRowID = neuronID * nTimesteps;

	// Index at which input-gradient is to be calculated (corresponds to i in formula)
	unsigned inputGradID = i + linearRowID;

	// ID for current surrogate gradient and output gradient (corresponds to j in formula)
	unsigned linearID = i + linearRowID;

	// First summand of input gradient is surrogate gradient * output gradient
	// Corresponds to j=i
	inputGrad[inputGradID] = surr[linearID] * notClipped[linearID] * outputGrad[inputGradID];

	// The product over k in the formula in the function description. Will be incremented
	// iteratively while looping over j in the sum
	float gradProd = surr[linearID] * notClipped[linearID] * membrSubtract;

	// Iterate over sum, over different derivative enumerators.
	// Stop early when accumulated product is 0
	for(unsigned j=i + 1; (j<nTimesteps and gradProd != 0.0f); ++j)
	{
		// Update linearID with current j
		linearID = j + linearRowID;

		// New summand to inputGrad_i
		inputGrad[inputGradID] -= outputGrad[linearID] * surr[linearID] * notClipped[linearID] * gradProd;

		// Update product for next step
		gradProd *= (alpha - membrSubtract * notClipped[linearID] * surr[linearID]);
	}
}


/** Spike generation backward pass for general reset kernels (less efficient)
 *
 * Assuming a function that calculates the output spikes of a SRM-neuron (arbitrary
 * response and surrogate gradients) for a given (synaptic) input that
 * has already been convolved with the input-spike response kernel, this kernel computes
 * all elements of the the input gradient for one neuron and/or batch. It amounts to
 * the scalar product of the output gradient with the derivative of the spike output
 * wrt. the (pre-spiking) membrane potential at the i-th timestep.
 *
 * Does not support neurons with lower bound for their membrane potential, but
 * supports arbitrary refractory responses.
 *
 * inputGrad_i = \sum_{j=i}^{N_s - 1} outputGrad_j * dOutput_j / dV_i
 * 			   = surr_i * outputGrad_i +
 * 				 \sum_{j=i+1}^{N_s - 1} outputGrad_j *
 * 				 * surr_j * \sum_{k=i}^{j - 1} refr[j-k] * dOutput_k / dV_i
 *
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param jaco 3D-tensor (nNeurons x nTimesteps x nTimesteps) to temporarily store elements of the Jacobian
 * 			   matrix for each neuron.
 * @param surr 2D-tensor (nNeurons x nTimesteps) that holds the surrogate gradients at different points in time
 * @param refr 1D-tensor (refrSize) Refractory response over time
 * @param nNeurons Number of neurons/batches
 * @param refrSize Number of timesteps of the refractory response
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
__global__ void spikeBackwardRefrKernel(
	scalarType* __restrict__ inputGrad,
	const scalarType* __restrict__ outputGrad,
	scalarType* __restrict__ jaco,
	const scalarType* __restrict__ surr,
	const scalarType* __restrict__ refr,
	unsigned nNeurons, unsigned refrSize, unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	// First ID in current row of surr
	unsigned linearSurrRowID = neuronID * nTimesteps;
	// First ID in current transposed jacobian matrix
	unsigned linearJacoRowID = neuronID * nTimesteps;

	// iterate over rows of transposed jacobian (i.e. 'denominator' of derivative)
	for(unsigned i=0; i<nTimesteps; ++i)
	{
		// diagonal entry (j=i) equal to i-th surrogate gradient
		jaco[i + linearJacoRowID] = surr[i + linearSurrRowID];

		inputGrad[i + linearSurrRowID] += surr[i + linearSurrRowID] * outputGrad[i + linearSurrRowID];

		// above diagonal entries, iterate over coloumns (i.e. 'numerator' of derivative)
		for(unsigned j=i+1; j<nTimesteps; ++j)
		{
			unsigned linearSurrID = j + linearSurrRowID;
			unsigned linearJacoID = j + linearJacoRowID;

			float inner_sum = 0;
			for(unsigned k=i; k<j; ++k)
			{
				if(j-k < refrSize) inner_sum += jaco[k + linearJacoRowID] * refr[j - k];
			}

			// d(a_j) / d(V_i)
			jaco[linearJacoID] = surr[linearSurrID] * inner_sum;

			//Add to i-th component in input gradient
			inputGrad[i + linearSurrRowID] += jaco[linearJacoID] * outputGrad[linearSurrID];

		}
	}
}


/**
 * WIP
 * Equivalent to lifBackwardKernel but with a recursive formula:
 * inputGrad_i = notClipped_i * (surr_i * ouputGrad_i + (alpha - surr_i * membrSubtract))
 * inputGrad_{N_s-1} = surr_{N_s-1} * notClipped_{N_s-1} * ouputGrad_{N_s-1}
 *
 * Parallelize across neurons and batches
 */
template <class scalarType>
__global__ void lifBackwardKernelRecursive(
	scalarType* __restrict__ inputGrad,
	const scalarType* __restrict__ outputGrad,
	const scalarType* __restrict__ surr,
	const scalarType* __restrict__ notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned nTimesteps)
{
	// Identifier for the current neuron and/or batch
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (i.e. for current neuron)
	unsigned linearRowID = neuronID * nTimesteps;

	scalarType new_summand;  // notClipped_i * ouputGrad_i
	scalarType new_factor;  // (alpha - surr_i * membrSubtract)
	scalarType grad = 1.0;  // Current gradient.

	// ID corresponding to current gradient component
	unsigned linearID;

	for(unsigned j=nTimesteps-1; j-- > 0; )
	{
		linearID = linearRowID + j;

		new_summand = outputGrad[linearID] * surr[linearID];
		new_factor = (alpha - surr[linearID]);
		grad = (new_summand + new_factor * grad) * notClipped[linearID];
		inputGrad[linearID] = grad;
	}
}


// Host functions

	
template <class scalarType>
void spikeForwardCuda(
	scalarType* d_s,
	scalarType* d_u,
	float alpha,
	float membrSubtract,
	unsigned nNeurons,
	unsigned nTimesteps,
	float theta,
	float theta_low,
	bool applyThetaLow,
	unsigned maxNumSpikes)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	spikeForwardKernel<scalarType><<< block, thread >>>(
		d_s, d_u, alpha, membrSubtract, nNeurons, nTimesteps, theta, theta_low, applyThetaLow, maxNumSpikes);
}


/** Gradients for spikeForward function, but with constant refractory response.
 *
 * Assuming a function that calculates the output spikes of an LIF neuron (step-function
 * for refractory response, arbitrary surrogate gradients) for a given (synaptic) input,
 * that has already been convolved with the spike respones kernel, use the
 * spikeBackwardKernel kernel to compute the input gradient.
 * It amounts to the product of the transposed Jacobian (derivative of output spikes wrt.
 * convolved synaptic input and the output gradient.
 *
 * Parallelize over neurons/batches (thread.y) and elements of the input gradient (thread.x)
 *
 * Neuron-grid logic is taken from conv/corr functions in convKernels.h and ensures that
 * maximum block sizes are not exceeded, even for large number of parallel units.
 *
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) indicating for each time step whether the
 * 					 membrane potential has been clipped to a constant, which will
 * 					 result in 0 gradients at this point.
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
void spikeBackwardCuda(
	scalarType* inputGrad,
	const scalarType* outputGrad,
	const scalarType* surr,
	const scalarType* notClipped,
	float alpha, float membrSubtract, unsigned nNeurons, unsigned nTimesteps)
{
	dim3 thread(128, 8, 1);

	int nGrid = ceil(1.0f * nNeurons / thread.y / 65535);
	int neuronsPerGrid = ceil(1.0f * nNeurons / nGrid);

	for(auto i=0; i<nGrid; ++i)
	{
		int startOffset = i * neuronsPerGrid;
		int neuronsInGrid = (startOffset + neuronsPerGrid <= nNeurons) ? neuronsPerGrid : nNeurons - startOffset;

		if(neuronsInGrid < 0)	break;

		dim3 block( ceil( 1.0f * nTimesteps    / thread.x ),
					ceil( 1.0f * neuronsInGrid / thread.y ),
					1 );

		// these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

		spikeBackwardKernel<scalarType><<< block, thread >>>( inputGrad + startOffset * nTimesteps,
													outputGrad  + startOffset * nTimesteps,
													surr + startOffset * nTimesteps,
													notClipped + startOffset * nTimesteps,
													alpha, membrSubtract, neuronsInGrid, nTimesteps);
	}
}


/**
 * Gradients for spikeForward and spikeForwardLowBound functions
 */
template <class scalarType>
void spikeBackwardRefr(scalarType* inputGrad, const scalarType* outputGrad, scalarType* jaco, const scalarType* surr, const scalarType* refr, unsigned nNeurons, unsigned refrSize, unsigned nTimesteps)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	spikeBackwardRefrKernel<scalarType><<< block, thread >>>(inputGrad, outputGrad, jaco, surr, refr, nNeurons, refrSize, nTimesteps);
}


/**
 * WIP
 * Like lifBackward, but using a different computation method
 */
template <class scalarType>
void lifBackwardRecursive(
	scalarType* inputGrad,
	const scalarType* outputGrad,
	const scalarType* surr,
	const scalarType* notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	lifBackwardKernelRecursive<scalarType><<< block, thread >>>(
			inputGrad,
			outputGrad,
			surr,
			notClipped,
			membrSubtract, alpha, nNeurons, nTimesteps);
}

#endif // EXPERIMENTALKERNELS_H_INCLUDED
