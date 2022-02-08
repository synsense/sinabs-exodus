/*
 * Author: Felix Bauer, based on slayerPytorch library by Sumit Bam Shrestha
 * Contains routines that convert membrane potential of neurons into spikes
 * and their corresponding gradient (backward) functions.
 */
#ifndef SPIKEKERNELS_H_INCLUDED
#define SPIKEKERNELS_H_INCLUDED

#include <stdio.h>
/**
 * Forward evolution for a single IAF or LIF neuron, over time. Including state decay,
 * spike generation and subtract mechanism. No synaptic dynamics.
 * vmem_t = alpha * (vmem_{t-1} - spikes_{t-1}) + input_t
 * spikes_t = (vmem_t // theta) * (vmem_t > 0)
 *
 * @param outputSpikes 2D-tensor (nNeurons x Ns) to which the computed output spikes
 					   are to be written
 * @param vmem 2D-tensor (nNeurons x Ns) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x Ns) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param activationsPrev 1D-tensor (nNeurons) with the spikes of the preceding time step
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param theta Firing threshold
 * @param thetaLow Lower bound to vmem
 * @param applyThetaLow Flag whether vmem is lower bounded
 * @param maxNumSpikes Maximum number of spikes a neuron can emit per time step
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
**/
template <class T>
__global__ void lifForwardKernel(
	T* __restrict__ outputSpikes,
	T* __restrict__ vmemAll,
	T* __restrict__ input,
	T* __restrict__ vmemInitial,
	T* __restrict__ activationsPrev,
	float membrSubtract,
	float alpha,
	float theta,
	float thetaLow,
	bool applyThetaLow,
	unsigned maxNumSpikes,
	unsigned nNeurons,
	unsigned Ns)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	T vmemCurr = vmemInitial[neuronID];
	unsigned activation = unsigned(activationsPrev[neuronID]);

	for(unsigned t=0; t<Ns; ++t){
		// Decay state
		vmemCurr *= alpha;

		// ID of neuron and current timestep
		unsigned linearID = t + neuronID * Ns;

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

		// Subtract spikes
		vmemCurr -= activation * membrSubtract;

		// Write current vmemCurr into tensor
		vmemAll[linearID] = vmemCurr;
	}

}


/**
 * Forward evolution for (leaky) integrator dynamic, over time.
 * vmem_t = alpha * vmem_{t-1} + input_t
 *
 * @param vmem 2D-tensor (nNeurons x Ns) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x Ns) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
**/
template <class T>
__global__ void leakyForwardKernel(
	T* __restrict__ vmemAll,
	const T* __restrict__ input,
	const T* __restrict__ vmemInitial,
	float alpha,
	unsigned nNeurons,
	unsigned Ns)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	T vmemCurr = vmemInitial[neuronID];

	for(unsigned t=0; t<Ns; ++t){

		// Decay state
		vmemCurr *= alpha;

		// ID of neuron and current timestep
		unsigned linearID = t + neuronID * Ns;

		// Add current input to vmemCurr
		vmemCurr += input[linearID];

		// Write current vmemCurr into tensor
		vmemAll[linearID] = vmemCurr;
	}

}


/**
 * WIP
 * Backward pass for (leaky) integrator dynamic.
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
 * @param vmem 2D-tensor (nNeurons x Ns) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x Ns) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
**/
template <class T>
__global__ void leakyBackwardKernel(
	T* __restrict__ gradInput,
	const T* __restrict__ gradOutput,
	float alpha,
	unsigned nNeurons,
	unsigned Ns)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	T grad_curr = 0;

	for(unsigned t=Ns-1; t<Ns; --t){

		// ID of neuron and current timestep
		unsigned linearID = t + neuronID * Ns;

		// Add corresponding element of gradOutput and multiply by alpha
		grad_curr = grad_curr * alpha + gradOutput[linearID];

		// Write current grad into gradInput
		gradInput[linearID] = grad_curr;
	}

}
template <class T>
__global__ void getSpikesKernel(
	T* __restrict__ d_s,
	T* __restrict__ d_u,
	float membrSubtract,
	unsigned nNeurons,
	unsigned Ns,
	float theta,
	float theta_low,
	bool applyThetaLow,
	unsigned maxNumSpikes)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	T vmemCurr = vmemInitial[neuronID];
	unsigned activation;

	for(unsigned i=0; i<Ns; ++i)
	{
		// ID of neuron and current timestep
		unsigned linearID = i + neuronID * Ns;

		// Spike mechanism
		if(d_u[linearID] >= theta)
		{
            unsigned activation = min(unsigned(d_u[linearID] / theta), maxNumSpikes);
			d_s[linearID] = static_cast<float>(activation);

			// Reset mechanism
			for(unsigned j=1; j<Ns; ++j)
			{
				if(i + j < Ns)	d_u[linearID + j] -= membrSubtract * activation;
			}

		// Lower bound
		} else if(applyThetaLow && (d_u[linearID] < theta_low))
		{
			float difference = theta_low - d_u[linearID];
			for(unsigned j=1; j<Ns; ++j)
			{
				if(i + j < Ns) d_u[linearID + j] += difference;
			}
		}
	}

}


/**
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
 * @param inputGrad 2D-tensor (nNeurons x Ns) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x Ns) that holds the given output gradients
 * @param jaco 3D-tensor (nNeurons x Ns x Ns) to temporarily store elements of the Jacobian
 * 			   matrix for each neuron.
 * @param surr 2D-tensor (nNeurons x Ns) that holds the surrogate gradients at different points in time
 * @param refr 1D-tensor (refrSize) Refractory response over time
 * @param nNeurons Number of neurons/batches
 * @param refrSize Number of timesteps of the refractory response
 * @param Ns Number of timesteps
 */
template <class T>
__global__ void spikeGradsRefrKernel(
	T* __restrict__ inputGrad,
	const T* __restrict__ outputGrad,
	T* __restrict__ jaco,
	const T* __restrict__ surr,
	const T* __restrict__ refr,
	unsigned nNeurons, unsigned refrSize, unsigned Ns)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	// First ID in current row of surr
	unsigned linearSurrRowID = neuronID * Ns;
	// First ID in current transposed jacobian matrix
	unsigned linearJacoRowID = neuronID * Ns;

	// iterate over rows of transposed jacobian (i.e. 'denominator' of derivative)
	for(unsigned i=0; i<Ns; ++i)
	{
		// diagonal entry (j=i) equal to i-th surrogate gradient
		jaco[i + linearJacoRowID] = surr[i + linearSurrRowID];

		inputGrad[i + linearSurrRowID] += surr[i + linearSurrRowID] * outputGrad[i + linearSurrRowID];

		// above diagonal entries, iterate over coloumns (i.e. 'numerator' of derivative)
		for(unsigned j=i+1; j<Ns; ++j)
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
 * Assuming a function that calculates the output spikes of a SRM-neuron (step-function
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
 * 				 * alpha_j * \sum_{k=i}^{j - 1} beta_k * dOutput_k / dV_i
 * where alpha_i = surr_i * notClipped_i
 * and beta_i = (1 - membrSubtract_i * surr_i) * notClipped_i - 1
 *
 * @param inputGrad 2D-tensor (nNeurons x Ns) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x Ns) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
 */
template <class T>
__global__ void spikeGradsKernel(
	T* __restrict__ inputGrad,
	const T* __restrict__ outputGrad,
	const T* __restrict__ surr,
	const T* __restrict__ notClipped,
	float membrSubtract, unsigned nNeurons, unsigned Ns)
{
	// Identifier corresponding to the element of the input gradient that is
	// computed as well as the denominator in the derivatives
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= Ns)	return;

	// Identifier for the current neuron and/or batch
	unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;
	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (i.e. for current neuron)
	unsigned linearRowID = neuronID * Ns;

	// Index at which input-gradient is to be calculated
	unsigned inputGradID = i + linearRowID;

	// ID for current surrogate gradient and output gradient
	unsigned linearID = i + linearRowID;

	// First summand of input gradient is surrogate gradient * output gradient
	float delta = surr[linearID] * notClipped[linearID];
	inputGrad[inputGradID] = delta * outputGrad[inputGradID];

	// Integrate over past gradients, implicitly implementing the sum over
	// k from the formula given in the function description
	float gradSum = 0.0f;

	float gamma;

	// Iterate through sum, over different derivative enumerators.
	for(unsigned j=i + 1; j<Ns; ++j)
	{
		// New intermediate grad still uses surr and notClipped at previous index (j-1)
		gamma = (1.0f - membrSubtract * surr[linearID]) * notClipped[linearID] - 1.0f;
		gradSum += gradSum * gamma;
		// New gradient (da_j/dV_i) is delta * gradSum, at current index (j)
		linearID = j + linearRowID;
		delta = surr[linearID] * notClipped[linearID];
		// Add product of output gradient and new gradient to input gradient
		inputGrad[inputGradID] += gradSum * delta * outputGrad[linearID];
	}
}


/**
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
 * @param inputGrad 2D-tensor (nNeurons x Ns) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x Ns) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
 */
template <class T>
__global__ void fullGradsKernel(
	T* __restrict__ inputGrad,
	const T* __restrict__ outputGrad,
	const T* __restrict__ surr,
	const T* __restrict__ notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned Ns)
{
	// Identifier corresponding to the element of the input gradient that is
	// computed as well as the denominator in the derivatives
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= Ns)	return;

	// Identifier for the current neuron and/or batch
	unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;
	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (i.e. for current neuron)
	unsigned linearRowID = neuronID * Ns;
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
	for(unsigned j=i + 1; (j<Ns and accGrad != 0.0f); ++j)
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


/**
 * WIP
 * Equivalent to fullGradsKernel but with a recursive formula:
 * inputGrad_i = notClipped_i * (surr_i * ouputGrad_i + (alpha - surr_i * membrSubtract))
 * inputGrad_{N_s-1} = surr_{N_s-1} * notClipped_{N_s-1} * ouputGrad_{N_s-1}
 *
 * Parallelize across neurons and batches
 */
template <class T>
__global__ void fullGradsKernelRecursive(
	T* __restrict__ inputGrad,
	const T* __restrict__ outputGrad,
	const T* __restrict__ surr,
	const T* __restrict__ notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned Ns)
{
	// Identifier for the current neuron and/or batch
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (i.e. for current neuron)
	unsigned linearRowID = neuronID * Ns;

	T new_summand;  // notClipped_i * ouputGrad_i
	T new_factor;  // (alpha - surr_i * membrSubtract)
	T grad = 1.0;  // Current gradient.

	// ID corresponding to current gradient component
	unsigned linearID;

	for(unsigned j=Ns-1; j-- > 0; )
	{
		linearID = linearRowID + j;

		new_summand = outputGrad[linearID] * surr[linearID];
		new_factor = (alpha - surr[linearID]);
		grad = (new_summand + new_factor * grad) * notClipped[linearID];
		inputGrad[linearID] = grad;
	}
}


template <class T>
__global__ void evalRhoKernel(T* d_rho, const T* d_u, float theta, float tau, unsigned nNeurons, unsigned Ns, float scale)
{
	unsigned timeID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned nID    = blockIdx.y * blockDim.y + threadIdx.y;

	if(timeID >= Ns || nID >= nNeurons)	return;

	unsigned linearID = timeID + nID * Ns;

	d_rho[linearID] = scale/tau * exp(-fabs(theta - d_u[linearID])/tau);
}

template <class T>
void getSpikes(
	T* d_s,
	T* d_u,
	float membrSubtract,
	unsigned nNeurons,
	unsigned Ns,
	float theta,
	float theta_low,
	bool applyThetaLow,
	unsigned maxNumSpikes)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	getSpikesKernel<T><<< block, thread >>>(
		d_s, d_u, membrSubtract, nNeurons, Ns, theta, theta_low, applyThetaLow, maxNumSpikes);
}


/**
 * Gradients for getSpikes and getSpikesLowBound functions
 */
template <class T>
void spikeGradsRefr(T* inputGrad, const T* outputGrad, T* jaco, const T* surr, const T* refr, unsigned nNeurons, unsigned refrSize, unsigned Ns)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	spikeGradsRefrKernel<T><<< block, thread >>>(inputGrad, outputGrad, jaco, surr, refr, nNeurons, refrSize, Ns);
}


/**
 * Forward evolution for IAF or LIF neurons, including state decay, spike generation
 * and subtract mechanism. No synaptic dynamics.
 * vmem_t = alpha * (vmem_{t-1} - spikes_{t-1}) + input_t
 * spikes_t = (vmem_t // theta) * (vmem_t > 0)
 *
 * For IAF dynamics set alpha = 1, for LIF 0 < alpha < 1
 *
 * Parallelize over neurons/batches
 *
 * @param outputSpikes 2D-tensor (nNeurons x Ns) to which the computed output spikes
 					   are to be written
 * @param vmem 2D-tensor (nNeurons x Ns) to which the computed membrane potentials
 * 			   are to be written
 * @param input 2D-tensor (nNeurons x Ns) with the input
 * @param vmemInitial 1D-tensor (nNeurons) with the initial membrane potentials
 * @param activationsPrev 1D-tensor (nNeurons) with the spikes of the preceding time step
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param theta Firing threshold
 * @param thetaLow Lower bound to vmem
 * @param applyThetaLow Flag whether vmem is lower bounded
 * @param multipleSpikes Flag whether multiple spikes can be emitted in a single time step
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
 */
template <class T>
void lifForward(
	T* outputSpikes,
	T* vmem,
	T* const input,
	T* const vmemInitial,
	T* const activationsPrev,
	const float membrSubtract,
	const float alpha,
	const float theta,
	const float thetaLow,
	const bool applyThetaLow,
	const unsigned maxNumSpikes,
	const unsigned nNeurons,
	const unsigned Ns)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	lifForwardKernel<T><<< block, thread >>>(
			outputSpikes,
			vmem,
			input,
			vmemInitial,
			activationsPrev,
			membrSubtract, alpha, theta, thetaLow, applyThetaLow, maxNumSpikes, nNeurons, Ns);
}


/**
 * Assuming a function that calculates the output spikes of an IAF or LIF neuron (step-function
 * or exponential for input spike response and step-function for refractory response, arbitrary
 * surrogate gradients) for a given (synaptic) input, use the fullGradsKernel kernel to compute
 * the input gradient.
 * It amounts to the product of the transposed Jacobian (derivative of output spikes wrt.
 * synaptic input, i.e. over the whole spike-response, resetting and spiking process)
 * and the output gradient.
 *
 * Parallelize over neurons/batches (thread.y) and elements of the input gradient (thread.x)
 *
 * The call to fullGradsKernel can be replaced with spikeGradsKernel, which will
 * give the derivatives wrt. to the synaptic inputs after they have been convolved with
 * an input-spike response kernel, so only for the spiking/resetting mechanism, allowing for
 * arbitrary spike response kernels.
 *
 * Neuron-grid logic is taken from conv/corr functions in convKernels.h and ensures that
 * maximum block sizes are not exceeded, even for large number of parallel units.
 *
 * @param inputGrad 2D-tensor (nNeurons x Ns) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x Ns) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x Ns) indicating for each time step whether the
 * 					 membrane potential has been clipped to a constant, which will
 * 					 result in 0 gradients at this point.
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param alhpa Decay factor of the neuron state (exp(-dt/tau)). For IAF neurons set to 1.
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
 */
template <class T>
void spikeGradsFull(
	T* inputGrad,
	const T* outputGrad,
	const T* surr,
	const T* notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned Ns)
{
	dim3 thread(128, 8, 1);

	int nGrid = ceil(1.0f * nNeurons / thread.y / 65535);
	int neuronsPerGrid = ceil(1.0f * nNeurons / nGrid);

	for(auto i=0; i<nGrid; ++i)
	{
		int startOffset = i * neuronsPerGrid;
		int neuronsInGrid = (startOffset + neuronsPerGrid <= nNeurons) ? neuronsPerGrid : nNeurons - startOffset;

		if(neuronsInGrid < 0)	break;

		dim3 block( ceil( 1.0f * Ns    / thread.x ),
					ceil( 1.0f * neuronsInGrid / thread.y ),
					1 );

		// these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

		fullGradsKernel<T><<< block, thread >>>( inputGrad + startOffset * Ns,
													outputGrad  + startOffset * Ns,
													surr + startOffset * Ns,
													notClipped + startOffset * Ns,
													membrSubtract, alpha, neuronsInGrid, Ns);
	}
}

/**
 * Forward pass for exponential leak
 * v_t = alpha * v_{t-1} + I_t
 * Parallelize across neurons/batches
 */
template <class T>
void leakyForward(
	T* vmemAll,
	const T* input,
	const T* vmemInitial,
	float alpha, unsigned nNeurons, unsigned Ns)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyForwardKernel<T><<< block, thread >>>(
			vmemAll,
			input,
			vmemInitial,
			alpha, nNeurons, Ns);
}

/**
 * Backward pass for exponential leak
 * v_t = alpha * v_{t-1} + I_t
 * Parallelize across neurons/batches
 */
template <class T>
void leakyBackward(
	T* inputGrad,
	const T* outputGrad,
	float alpha, unsigned nNeurons, unsigned Ns)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	leakyBackwardKernel<T><<< block, thread >>>(
			inputGrad,
			outputGrad,
			alpha, nNeurons, Ns);
}


/**
 * WIP
 * Like spikeGradsFull, but using a different computation method
 */
template <class T>
void spikeGradsFullRecursive(
	T* inputGrad,
	const T* outputGrad,
	const T* surr,
	const T* notClipped,
	float membrSubtract, float alpha, unsigned nNeurons, unsigned Ns)
{

	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);

	fullGradsKernelRecursive<T><<< block, thread >>>(
			inputGrad,
			outputGrad,
			surr,
			notClipped,
			membrSubtract, alpha, nNeurons, Ns);
}


/**
 * In short: gradients for getSpikes function, but with constant refractory response.
 *
 * In Detail:
 * Assuming a function that calculates the output spikes of an LIF neuron (step-function
 * for refractory response, arbitrary surrogate gradients) for a given (synaptic) input,
 * that has already been convolved with the spike respones kernel, use the
 * spikeGradsKernel kernel to compute the input gradient.
 * It amounts to the product of the transposed Jacobian (derivative of output spikes wrt.
 * convolved synaptic input and the output gradient.
 *
 * Parallelize over neurons/batches (thread.y) and elements of the input gradient (thread.x)
 *
 * Neuron-grid logic is taken from conv/corr functions in convKernels.h and ensures that
 * maximum block sizes are not exceeded, even for large number of parallel units.
 *
 * @param inputGrad 2D-tensor (nNeurons x Ns) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x Ns) that holds the given output gradients
 * @param surr 2D-tensor (nNeurons x Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param notClipped 2D-tensor (nNeurons x Ns) indicating for each time step whether the
 * 					 membrane potential has been clipped to a constant, which will
 * 					 result in 0 gradients at this point.
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
 */
template <class T>
void spikeGrads(
	T* inputGrad,
	const T* outputGrad,
	const T* surr,
	const T* notClipped,
	float membrSubtract, unsigned nNeurons, unsigned Ns)
{
	dim3 thread(128, 8, 1);

	int nGrid = ceil(1.0f * nNeurons / thread.y / 65535);
	int neuronsPerGrid = ceil(1.0f * nNeurons / nGrid);

	for(auto i=0; i<nGrid; ++i)
	{
		int startOffset = i * neuronsPerGrid;
		int neuronsInGrid = (startOffset + neuronsPerGrid <= nNeurons) ? neuronsPerGrid : nNeurons - startOffset;

		if(neuronsInGrid < 0)	break;

		dim3 block( ceil( 1.0f * Ns    / thread.x ),
					ceil( 1.0f * neuronsInGrid / thread.y ),
					1 );

		// these should never be trigerred
		if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded.");
		if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded.");

		spikeGradsKernelLB<T><<< block, thread >>>( inputGrad + startOffset * Ns,
													outputGrad  + startOffset * Ns,
													surr + startOffset * Ns,
													notClipped + startOffset * Ns,
													membrSubtract, neuronsInGrid, Ns);
	}
}

template <class T>
void evalRho(T* d_rho, const T* d_u, float theta, float tauRho, float scaleRho, unsigned nNeurons, unsigned Ns)
{
	dim3 thread, block;
	thread.x = 128;
	thread.y = 8;
	block.x = ceil(1.0f * Ns/thread.x);
	block.y = ceil(1.0f * nNeurons/thread.y);
	if(block.y >= 65535)	AT_ERROR("maximum blockDim.y exceeded");
	if(block.z >= 65535)	AT_ERROR("maximum blockDim.z exceeded");

	// slayerio::cout << "scaleRho = " << scaleRho << ", tauRho = " << tauRho << std::endl;

	// evalRhoKernel<<< block, thread >>>(rho, u, theta, tau, info.nNeurons, Ns);
	// evalRhoKernel<<< block, thread >>>(rho, u, theta, tau, info.nNeurons, Ns, 1.0/10);
	evalRhoKernel<<< block, thread >>>(d_rho, d_u, theta, tauRho * theta, nNeurons, Ns, scaleRho);
}

#endif // SPIKEKERNELS_H_INCLUDED
