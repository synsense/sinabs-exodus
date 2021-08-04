/*
 * Author: Sumit Bam Shrestha
 * 09/05/2019 4:00 PM
 * Contains routines that converts membrane potential of neuron into spikes
 */
#ifndef SPIKEKERNELS_H_INCLUDED
#define SPIKEKERNELS_H_INCLUDED

template <class T>
__global__ void getSpikesKernel(
	T* __restrict__ d_s,
	T* __restrict__ d_u,
	const T* __restrict__ d_nu,
	unsigned nNeurons, unsigned nuSize, unsigned Ns, float theta, float Ts)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
	const T spike = 1.0f/Ts;

	if(neuronID >= nNeurons)	return;

	for(unsigned i=0; i<Ns; ++i)
	{
		unsigned linearID = i + neuronID * Ns;
		if(d_u[linearID] >= theta)
		{
            int num_spikes = d_u[linearID] / theta;
			d_s[linearID] += spike * num_spikes;
			for(unsigned j=1; j<nuSize; ++j)
			{
				if(i + j < Ns)	d_u[linearID + j] += d_nu[j] * num_spikes;
			}
		}
	}

}

template <class T>
__global__ void getSpikesKernelLowBound(
	T* __restrict__ d_s,
	T* __restrict__ d_u,
	const T* __restrict__ d_nu,
	unsigned nNeurons,
	unsigned nuSize,
	unsigned Ns,
	float theta,
	float theta_low,
	float Ts
)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;
	const T spike = 1.0f/Ts;

	if(neuronID >= nNeurons)	return;

	for(unsigned i=0; i<Ns; ++i)
	{
		unsigned linearID = i + neuronID * Ns;
		if(d_u[linearID] >= theta)
		{
            int num_spikes = d_u[linearID] / theta;
			d_s[linearID] += spike * num_spikes;
			for(unsigned j=1; j<nuSize; ++j)
			{
				if(i + j < Ns)	d_u[linearID + j] += d_nu[j] * num_spikes;
			}
		} else if(d_u[linearID] < theta_low)
		{
			float difference = theta_low - d_u[linearID];
			for(unsigned j=1; j<Ns; ++j)
			{
				if(i + j < Ns) d_u[linearID + j] += difference;
			}
		}
	}

}

template <class T>
__global__ void spikeGradsKernel(
	T* __restrict__ inGrad,
	const T* __restrict__ outGrad,
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

		inGrad[i + linearSurrRowID] += surr[i + linearSurrRowID] * outGrad[i + linearSurrRowID];

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
			inGrad[i + linearSurrRowID] += jaco[linearJacoID] * outGrad[linearSurrID];

			// printf("j: %d, i: %d, a: %f, out: %f, in:%f\n", j, i, jaco[linearJacoID], outGrad[linearSurrID], inGrad[i + linearSurrRowID]);
		}
	}
}

template <class T>
__global__ void spikeGradsKernel1D(
	T* __restrict__ inGrad,
	const T* __restrict__ outGrad,
	const T* __restrict__ surr,
	float refr, unsigned nNeurons, unsigned Ns)
{
	// identifier corresponding to denominator in derivative (like 'i' in spikeGradsKernel)
	// and to current row of transposed jacobian matrix
	unsigned tID = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;

	//printf("block.x: %d, thread.x: %d, block.y: %d, thread.y: %d\n", blockIdx.x, threadIdx.x, blockIdx.y, threadIdx.y);

	if(neuronID >= nNeurons)	return;
	if(tID >= Ns)	return;

	// ID of first element in current row of 2D tensors (e.g. for current neuron)
	unsigned linearSurrRowID = neuronID * Ns;
	// ID of time point at which input-gradient is to be calculated
	unsigned inGradID = tID + linearSurrRowID;

	// First summand of input gradient is surrogate gradient * output gradient
	float newGrad = surr[tID + linearSurrRowID];
	inGrad[inGradID] = newGrad * outGrad[inGradID];
	// Integrate over past gradients
	float gradSum = 0;

	// above diagonal entries, iterate over coloumns (i.e. 'numerator' of derivative)
	for(unsigned j=tID + 1; j<Ns; ++j)
	{
		// ID for current surrogate gradient and output gradient
		unsigned linearSurrID = j + linearSurrRowID;
		// Add previous gradient to gradient sum
		gradSum += newGrad;
		// New gradient (da_j/dV_t) is current surrogate gradient * refractory constant * grad sum
		newGrad = surr[linearSurrID] * refr * gradSum;
		// Add product of output gradient and new gradient to input gradient
		inGrad[inGradID] += newGrad * outGrad[linearSurrID];
	}
}


/**
 * This kernel computes the i-th element (corresponding to the i-th time step) each
 * of the the input gradient for one neuron and/or batch.
 * It amounts to the scalar product of the output gradient with the derivative of
 * the spike output wrt. the input at the i-th timestep.
 *
 * inputGrad_i = surr_i * outputGrad_i + \sum_{j=i}^{N_s - 1} outputGrad_j * surr_j *
 * 				 * \prod_{k=i}^{j-1} (1 + surr_k * membrSubtract)
 * @param inputGrad 2D-tensor (nNeurons x Ns) to which the computed
 * 					input gradients are to be written
 * @param outputGrad 2D-tensor (nNeurons x Ns) that holds the given output gradients
 * @param surr 1D-tensor (Ns) with the given surrogate gradients ds_t/dV_t for each t
 * @param membrSubtract Value that is subtracted from the membrane potential when spiking
 * @param nNeurons Number of neurons/batches
 * @param Ns Number of timesteps
 */
template <class T>
__global__ void fullGradsKernel1D(
	T* __restrict__ inputGrad,
	const T* __restrict__ outputGrad,
	const T* __restrict__ surr,
	float membrSubtract, unsigned nNeurons, unsigned Ns)
{
	// Identifier corresponding to the element of the input gradient that is
	// computed as well as the denominator in the derivatives
	unsigned i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i >= Ns)	return;

	// Identifier for the current neuron and/or batch
	unsigned neuronID = blockIdx.y * blockDim.y + threadIdx.y;
	if(neuronID >= nNeurons)	return;

	// Index of first element in current row of 2D tensors (e.g. for current neuron)
	unsigned linearSurrRowID = neuronID * Ns;
	// Index at which input-gradient is to be calculated
	unsigned inputGradID = i + linearSurrRowID;

	// First summand of input gradient is surrogate gradient * output gradient
	inputGrad[inputGradID] = surr[inputGradID] * outputGrad[inputGradID];

	// Accumulate product of past (1 + surr * membrSubtract) terms
	float accGrad = 1.0f;
	float newFactor;
	unsigned linearSurrID;

	// Iterate through sum. Stop early when accumulated product is 0
	for(unsigned j=i + 1; (j<Ns and accGrad != 0.0f); ++j)
	{
		// ID for current surrogate gradient and output gradient
		linearSurrID = j + linearSurrRowID;
		// New factor to be accumulated
		newFactor = 1.0f - membrSubtract * surr[linearSurrID - 1];
		accGrad *= newFactor;
		// Add new term to current gradient
		inputGrad[inputGradID] += accGrad * surr[linearSurrID] * outputGrad[linearSurrID];
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
void getSpikes(T* d_s, T* d_u, const T* d_nu, unsigned nNeurons, unsigned nuSize, unsigned Ns, float theta, float Ts)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	getSpikesKernel<T><<< block, thread >>>(d_s, d_u, d_nu, nNeurons, nuSize, Ns, theta, Ts);
}

template <class T>
void getSpikesLowBound(T* d_s, T* d_u, const T* d_nu, unsigned nNeurons, unsigned nuSize, unsigned Ns, float theta, float theta_low, float Ts)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	getSpikesKernelLowBound<T><<< block, thread >>>(d_s, d_u, d_nu, nNeurons, nuSize, Ns, theta, theta_low, Ts);
}

template <class T>
void spikeGrads(T* inGrad, const T* outGrad, T* jaco, const T* surr, const T* refr, unsigned nNeurons, unsigned refrSize, unsigned Ns)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	spikeGradsKernel<T><<< block, thread >>>(inGrad, outGrad, jaco, surr, refr, nNeurons, refrSize, Ns);
}

template <class T>
void spikeGradsFast(T* inGrad, const T* outGrad, const T* surr, float refr, unsigned nNeurons, unsigned Ns)
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

		spikeGradsKernel1D<T><<< block, thread >>>( inGrad + startOffset * Ns,
													outGrad  + startOffset * Ns,
													surr + startOffset * Ns,
													refr, neuronsInGrid, Ns);
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
