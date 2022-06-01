#include <torch/extension.h>
#include <vector>
#include "lif_kernels.h"
#include "leaky_kernels.h"
#include "experimental_kernels.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// LIF dynamics

void lifForward(
	const torch::Tensor& outputSpikes,
	const torch::Tensor& vmem,
	const torch::Tensor& input,
	const torch::Tensor& vmemInitial,
	const torch::Tensor& activationsPrev,
    const torch::Tensor& alpha,
	float membrSubtract,
	float theta,
	float thetaLow,
	bool applyThetaLow,
	int maxNumSpikes)
{
	CHECK_INPUT(input);
	CHECK_INPUT(outputSpikes);
	CHECK_INPUT(vmem);
	CHECK_INPUT(vmemInitial);
	CHECK_INPUT(activationsPrev);
	CHECK_INPUT(alpha);

	// check if tensors are on same device
	CHECK_DEVICE(input, vmem);
	CHECK_DEVICE(input, outputSpikes);
	CHECK_DEVICE(input, vmemInitial);
	CHECK_DEVICE(input, activationsPrev);
	CHECK_DEVICE(input, alpha);

	// set the current cuda device to wherever the tensor input resides
	cudaSetDevice(input.device().index());

	unsigned nTimesteps = input.size(-1);
	unsigned nNeurons = input.size(0);

	// convert maxNumSpikes to usnigned (-1 will become max)
	unsigned maxNumSpikesU = maxNumSpikes;

	// // output spikes
	// auto outputSpikes = torch::empty_like(input);
	// // membrane potential
	// auto vmem = torch::empty_like(input);

	lifForwardCuda<float>(
		outputSpikes.data_ptr<float>(),
		vmem.data_ptr<float>(),
		input.data_ptr<float>(),
		vmemInitial.data_ptr<float>(),
		activationsPrev.data_ptr<float>(),
		alpha.data_ptr<float>(),
		membrSubtract, theta, thetaLow, applyThetaLow, maxNumSpikesU, nNeurons, nTimesteps);

	return;
}

torch::Tensor lifBackward(
	const torch::Tensor& surr,
	const torch::Tensor& outGrad,
	const torch::Tensor& notClipped,
	const torch::Tensor& alpha,
	float refr)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outGrad);
	CHECK_INPUT(notClipped);
	CHECK_INPUT(alpha);

	// check if tensor are in same device
	CHECK_DEVICE(surr, outGrad);
	CHECK_DEVICE(surr, notClipped);
	CHECK_DEVICE(surr, alpha);

	// set the current cuda device to wherever the tensor surr resides
	cudaSetDevice(surr.device().index());

	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inGrad = torch::empty_like(surr);

	lifBackwardCuda<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		alpha.data_ptr<float>(),
		refr, nNeurons, nTimesteps);

	return inGrad;
}

torch::Tensor lifBackwardAlpha(
	const torch::Tensor& surr,
	const torch::Tensor& outGrad,
	const torch::Tensor& vmem,
	const torch::Tensor& notClipped,
	const torch::Tensor& alpha,
	float refr)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outGrad);
	CHECK_INPUT(vmem);
	CHECK_INPUT(notClipped);
	CHECK_INPUT(alpha);

	// check if tensor are in same device
	CHECK_DEVICE(surr, outGrad);
	CHECK_DEVICE(surr, vmem);
	CHECK_DEVICE(surr, notClipped);
	CHECK_DEVICE(surr, alpha);

	// set the current cuda device to wherever the tensor surr resides
	cudaSetDevice(surr.device().index());

	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto alphaGrad = torch::empty_like(surr);

	lifBackwardCuda<float>(
		alphaGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		vmem.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		alpha.data_ptr<float>(),
		refr, nNeurons, nTimesteps);

	return alphaGrad;
}


// Leaky integrators

torch::Tensor leakyForward(
	const torch::Tensor& input,
	const torch::Tensor& vmemInitial,
    float alpha)
{
	CHECK_INPUT(input);
	CHECK_INPUT(vmemInitial);

	// check if tensor are in same device
	CHECK_DEVICE(input, vmemInitial);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(vmemInitial.device().index());

	unsigned nTimesteps = input.size(-1);
	unsigned nNeurons = input.size(0);

	// Tensor to store membrane potential
	auto vmemFull = torch::empty_like(input);

	leakyForwardCuda<float>(
		vmemFull.data_ptr<float>(),
		input.data_ptr<float>(),
		vmemInitial.data_ptr<float>(),
		alpha, nNeurons, nTimesteps);

	return vmemFull;
}

torch::Tensor leakyBackward(
	const torch::Tensor& gradOutput,
    float alpha)
{
	CHECK_INPUT(gradOutput);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(gradOutput.device().index());

	unsigned nTimesteps = gradOutput.size(-1);
	unsigned nNeurons = gradOutput.size(0);

	// Tensor to store input gradient
	auto gradInput = torch::empty_like(gradOutput);

	leakyBackwardCuda<float>(
		gradInput.data_ptr<float>(),
		gradOutput.data_ptr<float>(),
		alpha, nNeurons, nTimesteps);

	return gradInput;
}


// Experimental functions

torch::Tensor spikeForward(
	torch::Tensor d_u,
	const float alpha,
	const float membrSubtract,
	const float theta,
	const float theta_low,
	const bool applyThetaLow,
	int maxNumSpikes)
{
	CHECK_INPUT(d_u);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(d_u.device().index());

	// Tensor to collect output spikes
	auto d_s = torch::zeros_like(d_u);

	// convert maxNumSpikes to usnigned (-1 will become max)
	unsigned maxNumSpikesU = maxNumSpikes;

	unsigned nTimesteps = d_u.size(-1);
	unsigned nNeurons = d_u.size(0);
	spikeForwardCuda<float>(
		d_s.data_ptr<float>(),
		d_u.data_ptr<float>(),
		alpha, membrSubtract, nNeurons, nTimesteps, theta, theta_low, applyThetaLow, maxNumSpikesU);

	return d_s;
}

torch::Tensor spikeBackwardRefrCuda(
	const torch::Tensor& surr, const torch::Tensor& outGrad, const torch::Tensor& refr)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outGrad);
	CHECK_INPUT(refr);

	// check if tensor are in same device
	CHECK_DEVICE(surr, outGrad);
	CHECK_DEVICE(surr, refr);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(surr.device().index());

	unsigned refrSize = refr.size(-1);
	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// jacobian
	auto jaco = torch::zeros({nNeurons, nTimesteps, nTimesteps}, torch::dtype(torch::kFloat32).device(surr.device()));

	// input gradients
	auto inGrad = torch::zeros_like(surr);

	spikeBackwardRefr<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		jaco.data_ptr<float>(),
		surr.data_ptr<float>(),
		refr.data_ptr<float>(),
		nNeurons, refrSize, nTimesteps);

	return inGrad;
}

torch::Tensor spikeBackward(
	const torch::Tensor& surr,
	const torch::Tensor& outGrad,
	const torch::Tensor& notClipped,
	float alpha,
	float membrSubtract)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outGrad);
	CHECK_INPUT(notClipped);

	// check if tensor are in same device
	CHECK_DEVICE(surr, outGrad);
	CHECK_DEVICE(surr, notClipped);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(surr.device().index());

	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inGrad = torch::empty_like(surr);

	spikeBackwardCuda<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		alpha, membrSubtract, nNeurons, nTimesteps);

	return inGrad;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("spikeForward"     ,  &spikeForward     , "Spike generation forward pass");
	m.def("spikeBackward"    ,  &spikeBackward    ,	"Spike generation backward pass");
	m.def("spikeBackwardRefr",  &spikeBackwardRefrCuda,	"Spike generation backward pass for arbitrary refractory response");
	m.def("lifBackward"      ,  &lifBackward      , "LIF backward pass");
	m.def("lifBackwardAlpha" ,  &lifBackwardAlpha , "LIF backward pass for alphas");
	m.def("lifForward" 	     ,  &lifForward       , "LIF forward dynamics");
	m.def("leakyForward"     ,  &leakyForward     ,	"Forward pass of leaky integrator");
	m.def("leakyBackward"    ,  &leakyBackward    ,	"Backward pass of leaky integrator");
}
