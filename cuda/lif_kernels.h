/*
 * Author: Felix Bauer, loosely based on slayerPytorch library by Sumit Bam Shrestha
 * Contains routines to compute forward and backward passes
 * for the following neuron dynamics
 * - leaky and non-leaky integrate-and-fire
 */
#ifndef LIFKERNELS_H_INCLUDED
#define LIFKERNELS_H_INCLUDED

#include <stdio.h>


// Kernel functions


/** LIF or IAF forward kernel
 *
 * Forward evolution for a single IAF or LIF neuron, over time. Including state decay,
 * spike generation and subtract mechanism. No synaptic dynamics.
 * vmem_t = alpha * (vmem_{t-1} - spikes_{t-1}) + input_t
 * spikes_t = (vmem_t // theta) * (vmem_t > 0)
 *
 * @param outputSpikes 2D-tensor (nNeurons x nTimesteps) to which the computed output spikes
 					   are to be written
 * @param vmem 2D-tensor (nNeurons x nTimesteps) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x nTimesteps) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param activationsPrev 1D-tensor (nNeurons) with the spikes of the preceding time step
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param theta Firing threshold
 * @param thetaLow Lower bound to vmem
 * @param applyThetaLow Flag whether vmem is lower bounded
 * @param maxNumSpikes Maximum number of spikes a neuron can emit per time step
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void lifForwardKernel(
	scalarType* __restrict__ outputSpikes,
	scalarType* __restrict__ vmemAll,
	scalarType* __restrict__ input,
	scalarType* __restrict__ vmemInitial,
	scalarType* __restrict__ activationsPrev,
	float membrSubtract,
	float alpha,
	float theta,
	float thetaLow,
	bool applyThetaLow,
	unsigned maxNumSpikes,
	unsigned nNeurons,
	unsigned nTimesteps)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	scalarType vmemCurr = vmemInitial[neuronID];
	unsigned activation = unsigned(activationsPrev[neuronID]);

	for(unsigned t=0; t<nTimesteps; ++t){
		// Subtract spikes
		vmemCurr -= activation * membrSubtract;

		// Decay state
		vmemCurr *= alpha;

		// ID of neuron and current timestep
		unsigned linearID = t + neuronID * nTimesteps;

		// Add current input to vmemCurr
		vmemCurr += input[linearID];

		// Apply lower threshold
		if (applyThetaLow && (vmemCurr < thetaLow)){
			vmemCurr = thetaLow;
		}

		// Generate spikes
		if(vmemCurr >= theta){
			activation = min(unsigned(vmemCurr / theta), maxNumSpikes);
		} else {
			activation = 0;
		}

		// Write activation into tensor
		outputSpikes[linearID] = static_cast<float>(activation);

		// Write current vmemCurr into tensor
		vmemAll[linearID] = vmemCurr;
	}

}


/** LIF or IAF backward kernel
 *
 * Assuming a function that calculates the output spikes of an IAF or LIF neuron (step-function
 * or exponential for input spike response and step-function for refractory response, arbitrary
 * surrogate gradients) for a given (synaptic) input, this kernel computes a single element
 * (corresponding to one time step) of the the input gradient for one neuron and/or batch.
 * It amounts to the scalar product of the output gradient with the derivative of
 * the spike output wrt. the input at the i-th timestep.
 *
 * inputGrad_i = surr_i * outputGrad_i * notClipped_{i} +
 * 				 \sum_{j=i+1}^{N_s - 1} outputGrad_j * surr_j *
 * 				 * \prod_{k=i}^{j-1} (alpha - surr_k * membrSubtract) * notClipped_{k}
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
__global__ void lifBackwardKernel(
	scalarType* __restrict__ inputGrad,
	const scalarType* __restrict__ outputGrad,
	const scalarType* __restrict__ surr,
	const scalarType* __restrict__ notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned nTimesteps)
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
	// Index at which input-gradient is to be calculated
	unsigned inputGradID = i + linearRowID;

	// Accumulate product of past (alpha - surr * membrSubtract) * notClipped terms
	float accGrad = notClipped[inputGradID];

	// First summand of input gradient is surrogate gradient * output gradient * notClipped
	inputGrad[inputGradID] = surr[inputGradID] * outputGrad[inputGradID] * accGrad;

	float newFactor;
	unsigned linearSurrID;

	// Iterate through sum, over different derivative enumerators.
	// Stop early when accumulated product is 0
	for(unsigned j=i + 1; (j<nTimesteps and accGrad != 0.0f); ++j)
	{
		// ID for current surrogate gradient and output gradient
		linearSurrID = j + linearRowID;
		// New factor to be accumulated
		newFactor = alpha - membrSubtract * surr[linearSurrID - 1];
		accGrad *= (newFactor * notClipped[linearSurrID]);
		// Add new term to current gradient
		inputGrad[inputGradID] += accGrad * surr[linearSurrID] * outputGrad[linearSurrID];
	}
}


// Host functions


/** Forward pass for IAF or LIF neuron dynaimcs.
 *
 * Forward evolution for IAF or LIF neurons, including state decay, spike generation
 * and subtract mechanism. No synaptic dynamics.
 * vmem_t = alpha * (vmem_{t-1} - spikes_{t-1}) + input_t
 * spikes_t = (vmem_t // theta) * (vmem_t > 0)
 *
 * For IAF dynamics set alpha = 1, for LIF 0 < alpha < 1
 *
 * Parallelize over neurons/batches
 *
 * @param outputSpikes 2D-tensor (nNeurons x nTimesteps) to which the computed output spikes
 					   are to be written
 * @param vmem 2D-tensor (nNeurons x nTimesteps) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x nTimesteps) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param activationsPrev 1D-tensor (nNeurons) with the spikes of the preceding time step
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param theta Firing threshold
 * @param thetaLow Lower bound to vmem
 * @param applyThetaLow Flag whether vmem is lower bounded
 * @param multipleSpikes Flag whether multiple spikes can be emitted in a single time step
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
void lifForwardCuda(
	scalarType* outputSpikes,
	scalarType* vmem,
	scalarType* const input,
	scalarType* const vmemInitial,
	scalarType* const activationsPrev,
	const float membrSubtract,
	const float alpha,
	const float theta,
	const float thetaLow,
	const bool applyThetaLow,
	const unsigned maxNumSpikes,
	const unsigned nNeurons,
	const unsigned nTimesteps)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	lifForwardKernel<scalarType><<< block, thread >>>(
			outputSpikes,
			vmem,
			input,
			vmemInitial,
			activationsPrev,
			membrSubtract, alpha, theta, thetaLow, applyThetaLow, maxNumSpikes, nNeurons, nTimesteps);
}


/** Backward pass for IAF or LIF neuron dynaimcs.
 * 
 * Corresponding backward function for lifForward. Will backpropagate a gradient wrt. outputSpikes
 * to a gradient wrt. input. Works for arbitrary choices of surrogate gradients.
 *
 * Parallelize over neurons/batches (thread.y) and elements of the input gradient (thread.x)
 * by using lifBackwardKernel.
 *
 * The call to lifBackwardKernel can be replaced with spikeBackwardKernel, which will
 * give the derivatives wrt. to the synaptic inputs after they have been convolved with
 * an input-spike response kernel, so only for the spiking/resetting mechanism, allowing for
 * arbitrary spike response kernels.
 *
 * Neuron-grid logic ensures that maximum block sizes are not exceeded, even for large 
 * number of parallel units.
 *
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) indicating for each time step whether the
 * 					 membrane potential has been clipped to a constant, which will
 * 					 result in 0 gradients at this point.
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
void lifBackwardCuda(
	scalarType* inputGrad,
	const scalarType* outputGrad,
	const scalarType* surr,
	const scalarType* notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned nTimesteps)
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

		lifBackwardKernel<scalarType><<< block, thread >>>( inputGrad + startOffset * nTimesteps,
													outputGrad  + startOffset * nTimesteps,
													surr + startOffset * nTimesteps,
													notClipped + startOffset * nTimesteps,
													membrSubtract, alpha, neuronsInGrid, nTimesteps);
	}
}


#endif // LIFKERNELS_H_INCLUDED
