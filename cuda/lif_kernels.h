/*
 * This file contains routines to compute forward and backward passes
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
 * vmem_t = \alpha * (vmem_{t-1} - spikes_{t-1}) + input_t
 * spikes_t = (vmem_t // theta) * (vmem_t > 0)
 *
 * @param outputSpikes 2D-tensor (nNeurons x nTimesteps) to which the computed output spikes
 					   are to be written
 * @param vmemAll 2D-tensor (nNeurons x nTimesteps) to which the computed membrane
 *                potentials are to be written
 * @param input 2D-tensor (nNeurons x nTimesteps) with the input
 * @param vmemPostInitial 1D-tensor (nNeurons) with the initial membrane potentials (after reset)
 * @param alhpa 1D-tensor with decay factor of the neuron states (exp(-dt/tau)).
 *              For IAF neurons set to 1.
 * @param theta 1D-tensor of firing thresholds
 * @param membrSubtract 1D tensor with values that are subtracted from the membrane
 *        potential when spiking
 * @param thetaLow 1D-tensor of lower bounds to vmem
 * @param applyThetaLow Flag whether vmem is lower bounded
 * @param maxNumSpikes Maximum number of spikes a neuron can emit per time step
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
**/
template <class scalarType>
__global__ void lifForwardKernel(
    scalarType* __restrict__ outputSpikes,
    scalarType* __restrict__ vmemAll,
    const scalarType* __restrict__ input,
    const scalarType* __restrict__ vmemPostInitial,
    const scalarType* __restrict__ alpha,
    const scalarType* __restrict__ membrSubtract,
    const scalarType* __restrict__ theta,
    const scalarType* __restrict__ thetaLow,
    bool applyThetaLow,
    unsigned maxNumSpikes,
    unsigned nNeurons,
    unsigned nTimesteps)
{
    unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
    
    if(neuronID >= nNeurons)	return;
    
    scalarType vmemCurr = vmemPostInitial[neuronID];
    unsigned activation = 0;
    
    for(unsigned t=0; t<nTimesteps; ++t){
    	// Subtract spikes
    	vmemCurr -= activation * membrSubtract[neuronID];
    
    	// Decay state
    	vmemCurr *= alpha[neuronID];
    
    	// ID of neuron and current timestep
    	unsigned linearID = t + neuronID * nTimesteps;
    
    	// Add current input to vmemCurr
    	vmemCurr += input[linearID];
    
    	// Apply lower threshold
    	if (applyThetaLow && (vmemCurr < thetaLow[neuronID])){
    		vmemCurr = thetaLow[neuronID];
    	}
    
    	// Generate spikes
    	if(vmemCurr >= theta[neuronID]){
    		activation = min(unsigned(vmemCurr / theta[neuronID]), maxNumSpikes);
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
 *               \sum_{j=i+1}^{nTimesteps - 1} outputGrad_j * surr_j *
 *               * \prod_{k=i}^{j-1} (\alpha - surr_k * membrSubtract) * notClipped_{k}
 * @param inputGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 *                  input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients
 *             ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) indicating whether vmem has been clipped
 * @param alhpa 1D-tensor with decay factor of the neuron states (exp(-dt/tau)).
 *              For IAF neurons set to 1.
 * @param membrSubtract 1D-tensor of value that is subtracted from the membrane potential
 *        when spiking
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
__global__ void lifBackwardKernel(
    scalarType* __restrict__ inputGrad,
    const scalarType* __restrict__ outputGrad,
    const scalarType* __restrict__ surr,
    const scalarType* __restrict__ notClipped,
    const scalarType* __restrict__ alpha,
    const scalarType* __restrict__ membrSubtract,
    const unsigned nNeurons,
    const unsigned nTimesteps)
{
    // Identifier corresponding to the element of the input gradient that is
    // computed as well as the denominator in the derivatives
    unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= nTimesteps) return;

    // Identifier for the current neuron and/or batch
    unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;
    if(neuronID >= nNeurons)    return;

    // Index of first element in current row of 2D tensors (i.e. for current neuron)
    unsigned linearRowID = neuronID * nTimesteps;
    // Index at which input-gradient is to be calculated
    unsigned iIndex = i + linearRowID;

    // Accumulate product of past (alpha - surr * membrSubtract) * notClipped terms
    float accGrad = notClipped[iIndex];

    // First summand of input gradient is surrogate gradient * output gradient * notClipped
    inputGrad[iIndex] = outputGrad[iIndex] * accGrad;

    float newFactor;
    unsigned jIndex;

    // Iterate through sum, over different derivative enumerators.
    // Stop early when accumulated product is 0
    for(unsigned j=i + 1; (j<nTimesteps and accGrad != 0.0f); ++j)
    {
        // ID for current surrogate gradient and output gradient
        jIndex = j + linearRowID;
        // New factor to be accumulated
        newFactor = alpha[neuronID] - membrSubtract[neuronID] * surr[jIndex - 1];
        accGrad *= (newFactor * notClipped[jIndex]);
        // Add new term to current gradient
        inputGrad[iIndex] += accGrad * outputGrad[jIndex];
    }

}


/** LIF or IAF backward kernel for calculating alpha gradients
 *
 * Assuming a function that calculates the output spikes of an IAF or LIF neuron (step-function
 * or exponential for input spike response and step-function for refractory response, arbitrary
 * surrogate gradients) for a given (synaptic) input, this kernel computes the alpha gradient
 * for one neuron and/or batch.
 * It amounts to the scalar product of the output gradient with the derivative of
 * the spike output wrt. alpha.
 * alphaGrad = \sum_{i=0}^{nTimesteps - 1} d(out_j) / d(alhpa) * outputGrad_j
 * \frac{d out_j}{d \alhpa} = accGrad_j * surr_j
 * accGrad_0 = 0,
 * accGrad_{j+1} = notClipped_{j+1} * (accGrad_j * (\alpha - surr_j * membrSubtract) + vmem_j)
 * @param alphaGrad 1D-tensor (nNeurons) to which the computed
 *                  alpha gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients
 * @param vmemPost 2D-tensor (nNeurons x nTimesteps) with membrane potentials after reset
 * @param vmemPostInitial 1D-tensor (nNeurons) with initial membrane potentials (after reset)
 *             ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) indicating whether vmem has been clipped
 * @param alhpa 1D-tensor with decay factor of the neuron states (exp(-dt/tau)).
 *              For IAF neurons set to 1.
 * @param membrSubtract 1D-tensor of value that is subtracted from the membrane potential
 *        when spiking
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
__global__ void lifBackwardAlphaKernel(
    scalarType* __restrict__ alphaGrad,
    const scalarType* __restrict__ outputGrad,
    const scalarType* __restrict__ vmemPost,
    const scalarType* __restrict__ vmemPostInitial,
    const scalarType* __restrict__ surr,
    const scalarType* __restrict__ notClipped,
    const scalarType* __restrict__ alpha,
    const scalarType* __restrict__ membrSubtract,
    const unsigned nNeurons,
    const unsigned nTimesteps)
{
    // Identifier for the current neuron and/or batch
    unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
    if(neuronID >= nNeurons) return;

    // Index of first element in current row of 2D tensors (i.e. for current neuron)
    unsigned linearRowID = neuronID * nTimesteps;

    // accGrad_{i+1} = alpha * (1 - membrSubtract * surr_{i}) * accGrad_i + vmemPost_i
    float accGrad = vmemPostInitial[neuronID];

    // Summand of input gradient is accGrad * surrogate gradient * output gradient * notClipped
    float newGrad = notClipped[linearRowID] * accGrad;
    alphaGrad[neuronID] = outputGrad[linearRowID] * newGrad;

    unsigned jIndex;

    // Iterate through sum, over different derivative enumerators.
    // Stop early when accumulated product is 0
    for(unsigned j=1; j < nTimesteps; ++j)
    {
        // ID for current surrogate gradient and output gradient
        jIndex = j + linearRowID;
        // Multiply dv_t / dv_{t-1} - term to accumulated gradient
        accGrad *= alpha[neuronID] * (1.0 - membrSubtract[neuronID] * surr[jIndex - 1]);
        // Add membane potential
        accGrad += vmemPost[jIndex - 1];
        // Multiply with 0 if clipped
        accGrad *= notClipped[jIndex];
        // // New gradient for current time step
        // newGrad = accGrad * surr[jIndex];
        
	// Add new term to current gradient scalar product
        alphaGrad[neuronID] += accGrad * outputGrad[jIndex];
    }

}


// Host functions


/** Forward pass for IAF or LIF neuron dynaimcs.
 *
 * Forward evolution for IAF or LIF neurons, including state decay, spike generation
 * and subtract mechanism. No synaptic dynamics.
 * vmem_t = \alpha * (vmem_{t-1} - spikes_{t-1}) + input_t
 * spikes_t = (vmem_t // theta) * (vmem_t > 0)
 *
 * For IAF dynamics set \alpha = 1, for LIF 0 < \alpha < 1
 *
 * Parallelize over neurons/batches
 *
 * @param outputSpikes 2D-tensor (nNeurons x nTimesteps) to which the computed output spikes
 					   are to be written
 * @param vmem 2D-tensor (nNeurons x nTimesteps) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x nTimesteps) with the input
 * @param vmemPostInitial 1D-tensor (nNeurons) with the initial membrane potentials (after reset)
 * @param alhpa 1D-tensor with decay factor of the neuron states (exp(-dt/tau)).
 *              For IAF neurons set to 1.
 * @param membrSubtract 1D-tensor of value that is subtracted from the membrane potential
 *        when spiking
 * @param theta 1D-tensor of firing thresholds
 * @param thetaLow 1D-tensor of lower bounds to vmem
 * @param applyThetaLow Flag whether vmem is lower bounded
 * @param multipleSpikes Flag whether multiple spikes can be emitted in a single time step
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
void lifForwardCuda(
	scalarType* outputSpikes,
	scalarType* vmem,
	const scalarType* input,
	const scalarType* vmemPostInitial,
	const scalarType* alpha,
	const scalarType* membrSubtract,
	const scalarType* theta,
	const scalarType* thetaLow,
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
	    vmemPostInitial,
        alpha,
        membrSubtract,
        theta,
        thetaLow,
        applyThetaLow,
        maxNumSpikes,
        nNeurons,
        nTimesteps);
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
 *                  input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) indicating for each time step whether the
 *                   membrane potential has been clipped to a constant, which will
 *                   result in 0 gradients at this point.
 * @param alhpa 1D-tensor with decay factor of the neuron states (exp(-dt/tau)).
 *              For IAF neurons set to 1.
 * @param membrSubtract 1D-tensor of value that is subtracted from the membrane potential
 *        when spiking
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
void lifBackwardCuda(
    scalarType* inputGrad,
    const scalarType* outputGrad,
    const scalarType* surr,
    const scalarType* notClipped,
    const scalarType* alpha,
    const scalarType* membrSubtract,
    const unsigned nNeurons,
    const unsigned nTimesteps)
{
    dim3 thread(128, 8, 1);

    int nGrid = ceil(1.0f * nNeurons / thread.y / 65535);
    int neuronsPerGrid = ceil(1.0f * nNeurons / nGrid);

    for(auto i=0; i<nGrid; ++i)
    {
        int startOffset = i * neuronsPerGrid;
        int neuronsInGrid = (startOffset + neuronsPerGrid <= nNeurons) ? neuronsPerGrid : nNeurons - startOffset;

        if(neuronsInGrid < 0)   break;

        dim3 block( ceil( 1.0f * nTimesteps    / thread.x ),
                    ceil( 1.0f * neuronsInGrid / thread.y ),
                    1 );

        // these should never be trigerred
        if(block.y >= 65535)    AT_ERROR("maximum blockDim.y exceeded.");
        if(block.z >= 65535)    AT_ERROR("maximum blockDim.z exceeded.");

        lifBackwardKernel<scalarType><<< block, thread >>>(
            inputGrad + startOffset * nTimesteps,
            outputGrad  + startOffset * nTimesteps,
            surr + startOffset * nTimesteps,
            notClipped + startOffset * nTimesteps,
            alpha + startOffset,
            membrSubtract + startOffset,
            neuronsInGrid,
            nTimesteps);
    }
}


/** Backward pass for IAF or LIF neuron dynaimcs to calculate alpha gradients.
 *
 * Corresponding backward function for alpha gradients of lifForward. Will backpropagate
 * a gradient wrt. outputSpikes to a gradient wrt. alpha. Works for arbitrary choices of
 * surrogate gradients.
 *
 * Parallelize over neurons/batches by using lifBackwardAlphaKernel.
 *
 * @param alphaGrad 2D-tensor (nNeurons x nTimesteps) to which the computed
 *                  alpha gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x nTimesteps) that holds the given output gradients
 * @param vmemPost 2D-tensor (nNeurons x nTimesteps) with membrane potentials after reset
 * @param vmemPostInitial 1D-tensor (nNeurons) with initial membrane potentials (after reset)
 * @param surr 2D-tensor (nNeurons x nTimesteps) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x nTimesteps) indicating for each time step whether the
 *                   membrane potential has been clipped to a constant, which will
 *                   result in 0 gradients at this point.
 * @param alhpa 1D-tensor with decay factor of the neuron states (exp(-dt/tau)).
 *              For IAF neurons set to 1.
 * @param membrSubtract 1D-tensor of value that is subtracted from the membrane potential
 *        when spiking
 * @param nNeurons Number of neurons/batches
 * @param nTimesteps Number of timesteps
 */
template <class scalarType>
void lifBackwardAlphaCuda(
    scalarType* alphaGrad,
    const scalarType* outputGrad,
    const scalarType* vmemPost,
    const scalarType* vmemPostInitial,
    const scalarType* surr,
    const scalarType* notClipped,
    const scalarType* alpha,
    const scalarType* membrSubtract,
    const unsigned nNeurons,
    const unsigned nTimesteps)
{


    unsigned thread = 256;
    unsigned block  = ceil(1.0f * nNeurons / thread);

    lifBackwardAlphaKernel<scalarType><<< block, thread >>>(
            alphaGrad,
            outputGrad,
	    vmemPost,
            vmemPostInitial,
            surr,
            notClipped,
            alpha,
            membrSubtract,
            nNeurons,
            nTimesteps);
}


#endif // LIFKERNELS_H_INCLUDED
