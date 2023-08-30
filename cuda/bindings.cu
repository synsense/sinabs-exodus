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
	const torch::Tensor& vmemPostInitial,
    const torch::Tensor& alpha,
	const torch::Tensor& membrSubtract,
	const torch::Tensor& theta,
	const torch::Tensor& thetaLow,
	const bool applyThetaLow,
	const int maxNumSpikes)
{
	CHECK_INPUT(input);
	CHECK_INPUT(outputSpikes);
	CHECK_INPUT(vmem);
	CHECK_INPUT(vmemPostInitial);
	CHECK_INPUT(alpha);
	CHECK_INPUT(membrSubtract);
	CHECK_INPUT(theta);
	CHECK_INPUT(thetaLow);

	// check if tensors are on same device
	CHECK_DEVICE(input, vmem);
	CHECK_DEVICE(input, outputSpikes);
	CHECK_DEVICE(input, vmemPostInitial);
	CHECK_DEVICE(input, alpha);
	CHECK_DEVICE(input, membrSubtract);
	CHECK_DEVICE(input, theta);
	CHECK_DEVICE(input, thetaLow);

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
		vmemPostInitial.data_ptr<float>(),
		alpha.data_ptr<float>(),
		membrSubtract.data_ptr<float>(),
		theta.data_ptr<float>(),
		thetaLow.data_ptr<float>(),
		applyThetaLow, maxNumSpikesU, nNeurons, nTimesteps);

	return;
}

torch::Tensor lifBackward(
	const torch::Tensor& surr,
	const torch::Tensor& outputGrad,
	const torch::Tensor& notClipped,
	const torch::Tensor& alpha,
	const torch::Tensor& membrSubtract)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outputGrad);
	CHECK_INPUT(notClipped);
	CHECK_INPUT(alpha);
	CHECK_INPUT(membrSubtract);

	// check if tensors are on same device
	CHECK_DEVICE(surr, outputGrad);
	CHECK_DEVICE(surr, notClipped);
	CHECK_DEVICE(surr, alpha);
	CHECK_DEVICE(surr, membrSubtract);

	// set the current cuda device to wherever the tensor surr resides
	cudaSetDevice(surr.device().index());

	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inputGrad = torch::empty_like(surr);

	lifBackwardCuda<float>(
		inputGrad.data_ptr<float>(),
		outputGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		alpha.data_ptr<float>(),
		membrSubtract.data_ptr<float>(),
		nNeurons, nTimesteps);

	return inputGrad;
}

torch::Tensor lifBackwardAlpha(
	const torch::Tensor& surr,
	const torch::Tensor& outputGrad,
	const torch::Tensor& vmemPost,
	const torch::Tensor& vmemPostInitial,
	const torch::Tensor& notClipped,
	const torch::Tensor& alpha,
	const torch::Tensor& membrSubtract)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outputGrad);
	CHECK_INPUT(vmemPost);
	CHECK_INPUT(vmemPostInitial);
	CHECK_INPUT(notClipped);
	CHECK_INPUT(alpha);
	CHECK_INPUT(membrSubtract);

	// check if tensors are on same device
	CHECK_DEVICE(surr, outputGrad);
	CHECK_DEVICE(surr, vmemPost);
	CHECK_DEVICE(surr, vmemPostInitial);
	CHECK_DEVICE(surr, notClipped);
	CHECK_DEVICE(surr, alpha);
	CHECK_DEVICE(surr, membrSubtract);

	// set the current cuda device to wherever the tensor surr resides
	cudaSetDevice(surr.device().index());

	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto alphaGrad = torch::empty_like(alpha);

	lifBackwardAlphaCuda<float>(
		alphaGrad.data_ptr<float>(),
		outputGrad.data_ptr<float>(),
		vmemPost.data_ptr<float>(),
		vmemPostInitial.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		alpha.data_ptr<float>(),
		membrSubtract.data_ptr<float>(),
		nNeurons, nTimesteps);

	return alphaGrad;
}


// Leaky integrators

torch::Tensor leakyForward(
	const torch::Tensor& input,
	const torch::Tensor& vmemInitial,
    const torch::Tensor& alpha)
{
	CHECK_INPUT(input);
	CHECK_INPUT(vmemInitial);
	CHECK_INPUT(alpha);

	// check if tensors are on same device
	CHECK_DEVICE(input, vmemInitial);
	CHECK_DEVICE(input, alpha);

	// set the current cuda device to wherever the tensor vmemInitial resides
	cudaSetDevice(vmemInitial.device().index());

	unsigned nTimesteps = input.size(-1);
	unsigned nNeurons = input.size(0);

	// Tensor to store membrane potential
	auto vmemFull = torch::empty_like(input);

	leakyForwardCuda<float>(
		vmemFull.data_ptr<float>(),
		input.data_ptr<float>(),
		vmemInitial.data_ptr<float>(),
		alpha.data_ptr<float>(),
		nNeurons, nTimesteps);

	return vmemFull;
}

torch::Tensor leakyBackward(
	const torch::Tensor& outputGrad,
    const torch::Tensor& alpha)
{
	CHECK_INPUT(outputGrad);
	CHECK_INPUT(alpha);

	// check if tensors are on same device
	CHECK_DEVICE(outputGrad, alpha);

	// set the current cuda device to wherever the tensor outputGrad resides
	cudaSetDevice(outputGrad.device().index());

	unsigned nTimesteps = outputGrad.size(-1);
	unsigned nNeurons = outputGrad.size(0);

	// Tensor to store input gradient
	auto inputGrad = torch::empty_like(outputGrad);

	leakyBackwardCuda<float>(
		inputGrad.data_ptr<float>(),
		outputGrad.data_ptr<float>(),
		alpha.data_ptr<float>(),
		nNeurons, nTimesteps);

	return inputGrad;
}

torch::Tensor leakyBackwardAlpha(
	const torch::Tensor& outputGrad,
	const torch::Tensor& output,
	const torch::Tensor& vmemInitial,
	const torch::Tensor& alpha)
{
	CHECK_INPUT(outputGrad);
	CHECK_INPUT(output);
	CHECK_INPUT(vmemInitial);
	CHECK_INPUT(alpha);

	// check if tensors are on same device
	CHECK_DEVICE(outputGrad, output);
	CHECK_DEVICE(outputGrad, vmemInitial);
	CHECK_DEVICE(outputGrad, alpha);

	// set the current cuda device to wherever the tensor outputGrad resides
	cudaSetDevice(outputGrad.device().index());

	unsigned nTimesteps = outputGrad.size(-1);
	unsigned nNeurons = outputGrad.size(0);

	// Tensor to store alpha gradient
	auto alphaGrad = torch::zeros_like(alpha);

	leakyBackwardAlphaCuda<float>(
		alphaGrad.data_ptr<float>(),
		outputGrad.data_ptr<float>(),
		output.data_ptr<float>(),
		vmemInitial.data_ptr<float>(),
		alpha.data_ptr<float>(),
		nNeurons, nTimesteps);

	return alphaGrad;
}


// Experimental functions

torch::Tensor spikeForward(
	torch::Tensor d_u,
	const float alpha,
	const float membrSubtract,
	const float theta,
	const float theta_low,
	const bool applyThetaLow,
	const int maxNumSpikes)
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
	const torch::Tensor& surr, const torch::Tensor& outputGrad, const torch::Tensor& refr)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outputGrad);
	CHECK_INPUT(refr);

	// check if tensors are on same device
	CHECK_DEVICE(surr, outputGrad);
	CHECK_DEVICE(surr, refr);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(surr.device().index());

	unsigned refrSize = refr.size(-1);
	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// jacobian
	auto jaco = torch::zeros({nNeurons, nTimesteps, nTimesteps}, torch::dtype(torch::kFloat32).device(surr.device()));

	// input gradients
	auto inputGrad = torch::zeros_like(surr);

	spikeBackwardRefr<float>(
		inputGrad.data_ptr<float>(),
		outputGrad.data_ptr<float>(),
		jaco.data_ptr<float>(),
		surr.data_ptr<float>(),
		refr.data_ptr<float>(),
		nNeurons, refrSize, nTimesteps);

	return inputGrad;
}

torch::Tensor spikeBackward(
	const torch::Tensor& surr,
	const torch::Tensor& outputGrad,
	const torch::Tensor& notClipped,
	const float alpha,
	const float membrSubtract)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outputGrad);
	CHECK_INPUT(notClipped);

	// check if tensors are on same device
	CHECK_DEVICE(surr, outputGrad);
	CHECK_DEVICE(surr, notClipped);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(surr.device().index());

	unsigned nTimesteps = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inputGrad = torch::empty_like(surr);

	spikeBackwardCuda<float>(
		inputGrad.data_ptr<float>(),
		outputGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		alpha, membrSubtract, nNeurons, nTimesteps);

	return inputGrad;
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
	m.def("leakyBackwardAlpha", &leakyBackwardAlpha,"Backward pass of leaky integrator for alphas");
}
