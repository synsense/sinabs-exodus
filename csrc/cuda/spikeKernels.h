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
			// dynamic parallelism seems to be slower because of race condition!!!
			// ahpKernel<<< block, thread >>>(d_u + linearID, d_nu, nuSize);
			// cudaDeviceSynchronize();
			for(unsigned j=1; j<nuSize; ++j)
			{
				if(i + j < Ns)	d_u[linearID + j] += d_nu[j] * num_spikes;
			}
		}
		//else	++i;
	}

}

template <class T>
__global__ void spikeGradsKernel(
	T* __restrict__ jaco,
	const T* __restrict__ surr,
	const T* __restrict__ refr,
	unsigned nNeurons, unsigned refrSize, unsigned Ns)
{
	unsigned neuronID = blockIdx.x * blockDim.x + threadIdx.x;

	if(neuronID >= nNeurons)	return;

	// First ID in current row of surr
	unsigned linearSurrRowID = neuronID * Ns;
	// First ID in current transoised jacobian matrix
	unsigned linearJacoOuterID = neuronID * Ns * Ns;

	// iterate over rows of transposed jacobian (i.e. 'denominator' of derivative)
	for(unsigned i=0; i<Ns; ++i)
	{
		// First ID in current row of transposed jacobian matrix
		unsigned linearJacoInnerID = i * Ns + linearJacoOuterID;

		// diagonal entry (j=i) equal to i-th surrogate gradient
		jaco[i + linearJacoInnerID] = surr[i + linearSurrRowID];

		// above diagonal entries, iterate over coloumns (i.e. 'numerator' of derivative)
		for(unsigned j=i+1; j<Ns; ++j)
		{
			unsigned linearSurrID = j + linearSurrRowID;
			unsigned linearJacoID = j + linearJacoInnerID;

			float inner_sum = 0;
			for(unsigned k=i; k<j; ++k)
			{
				inner_sum += jaco[k + linearJacoInnerID] * refr[j - k];
			}

			jaco[linearJacoID] = surr[linearSurrID] * inner_sum;

		}
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
void spikeGrads(T* jaco, const T* surr, const T* refr, unsigned nNeurons, unsigned refrSize, unsigned Ns)
{
	unsigned thread = 256;
	unsigned block  = ceil(1.0f * nNeurons / thread);
	spikeGradsKernel<T><<< block, thread >>>(jaco, surr, refr, nNeurons, refrSize, Ns);
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
