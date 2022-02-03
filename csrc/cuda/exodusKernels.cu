#include <torch/extension.h>
#include <vector>
#include "spikeKernels.h"
#include "convKernels.h"
#include "shiftKernels.h"

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_DEVICE(x, y) AT_ASSERTM(x.device().index() == y.device().index(), #x " and " #y " must be in same CUDA device")

// C++ Python interface

torch::Tensor getSpikesCuda(
	torch::Tensor d_u,
	const float membrSubtract,
	const float theta)
{
	CHECK_INPUT(d_u);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(d_u.device().index());

	// Tensor to collect output spikes
	auto d_s = torch::zeros_like(d_u);

	unsigned Ns = d_u.size(-1);
	unsigned nNeurons = d_u.size(0);
	getSpikes<float>(
		d_s.data_ptr<float>(),
		d_u.data_ptr<float>(),
		membrSubtract, nNeurons, Ns, theta);

	return d_s;
}

torch::Tensor getSpikesCudaLB(
	torch::Tensor d_u,
	const float membrSubtract,
	const float theta,
	const float theta_low)
{
	CHECK_INPUT(d_u);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(d_u.device().index());

	// Tensor to collect output spikes
	auto d_s = torch::zeros_like(d_u);

	unsigned Ns = d_u.size(-1);
	unsigned nNeurons = d_u.size(0);
	getSpikesLowBound<float>(
		d_s.data_ptr<float>(),
		d_u.data_ptr<float>(),
		membrSubtract, nNeurons, Ns, theta, theta_low);

	return d_s;
}

torch::Tensor spikeGradsRefrCuda(
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
	unsigned Ns = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// jacobian
	auto jaco = torch::zeros({nNeurons, Ns, Ns}, torch::dtype(torch::kFloat32).device(surr.device()));

	// input gradients
	auto inGrad = torch::zeros_like(surr);

	spikeGradsRefr<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		jaco.data_ptr<float>(),
		surr.data_ptr<float>(),
		refr.data_ptr<float>(),
		nNeurons, refrSize, Ns);

	return inGrad;
}

torch::Tensor spikeGradsCuda(
	const torch::Tensor& surr, const torch::Tensor& outGrad, float membrSubtract)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outGrad);

	// check if tensor are in same device
	CHECK_DEVICE(surr, outGrad);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(surr.device().index());

	unsigned Ns = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inGrad = torch::empty_like(surr);

	spikeGrads<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		membrSubtract, nNeurons, Ns);

	return inGrad;
}

torch::Tensor spikeGradsCudaLB(
	const torch::Tensor& surr,
	const torch::Tensor& outGrad,
	const torch::Tensor& notClipped,
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

	unsigned Ns = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inGrad = torch::empty_like(surr);

	spikeGradsLB<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		membrSubtract, nNeurons, Ns);

	return inGrad;
}

torch::Tensor spikeGradsFullCuda(
	const torch::Tensor& surr,
	const torch::Tensor& outGrad,
	const torch::Tensor& notClipped,
	float refr,
    float alpha)
{
	CHECK_INPUT(surr);
	CHECK_INPUT(outGrad);
	CHECK_INPUT(notClipped);

	// check if tensor are in same device
	CHECK_DEVICE(surr, outGrad);
	CHECK_DEVICE(surr, notClipped);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(surr.device().index());

	unsigned Ns = surr.size(-1);
	unsigned nNeurons = surr.size(0);

	// input gradients
	auto inGrad = torch::empty_like(surr);

	spikeGradsFull<float>(
		inGrad.data_ptr<float>(),
		outGrad.data_ptr<float>(),
		surr.data_ptr<float>(),
		notClipped.data_ptr<float>(),
		refr, alpha, nNeurons, Ns);

	return inGrad;
}

void lifForwardCuda(
	const torch::Tensor& outputSpikes,
	const torch::Tensor& vmem,
	const torch::Tensor& input,
	const torch::Tensor& vmemInitial,
	const torch::Tensor& activationsPrev,
	float membrSubtract,
    float alpha,
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

	// check if tensors are on same device
	CHECK_DEVICE(input, vmem);
	CHECK_DEVICE(input, outputSpikes);
	CHECK_DEVICE(input, vmemInitial);
	CHECK_DEVICE(input, activationsPrev);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(input.device().index());

	unsigned Ns = input.size(-1);
	unsigned nNeurons = input.size(0);

	// convert maxNumSpikes to usnigned (-1 will become max)
	unsigned maxNumSpikesU = maxNumSpikes;

	// // output spikes
	// auto outputSpikes = torch::empty_like(input);
	// // membrane potential
	// auto vmem = torch::empty_like(input);

	lifForward<float>(
		outputSpikes.data_ptr<float>(),
		vmem.data_ptr<float>(),
		input.data_ptr<float>(),
		vmemInitial.data_ptr<float>(),
		activationsPrev.data_ptr<float>(),
		membrSubtract, alpha, theta, thetaLow, applyThetaLow, maxNumSpikesU, nNeurons, Ns);

	return;
}   

torch::Tensor leakyForwardCuda(
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

	unsigned Ns = input.size(-1);
	unsigned nNeurons = input.size(0);

	// Tensor to store membrane potential
	auto vmemFull = torch::empty_like(input);

	leakyForward<float>(
		vmemFull.data_ptr<float>(),
		input.data_ptr<float>(),
		vmemInitial.data_ptr<float>(),
		alpha, nNeurons, Ns);

	return vmemFull;
}

torch::Tensor leakyBackwardCuda(
	const torch::Tensor& gradOutput,
    float alpha)
{
	CHECK_INPUT(gradOutput);

	// set the current cuda device to wherever the tensor d_u resides
	cudaSetDevice(gradOutput.device().index());

	unsigned Ns = gradOutput.size(-1);
	unsigned nNeurons = gradOutput.size(0);

	// Tensor to store input gradient 
	auto gradInput = torch::empty_like(gradOutput);

	leakyBackward<float>(
		gradInput.data_ptr<float>(),
		gradOutput.data_ptr<float>(),
		alpha, nNeurons, Ns);

	return gradInput;
}

torch::Tensor convCuda(torch::Tensor input, torch::Tensor filter, float Ts)
{
	CHECK_INPUT(input);
	CHECK_INPUT(filter);
	CHECK_DEVICE(input, filter);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);

	unsigned signalSize = input.size(-1);
	unsigned filterSize = filter.numel();
	unsigned nNeurons   = input.numel()/input.size(-1);
	conv<float>(output.data_ptr<float>(), input.data_ptr<float>(), filter.data_ptr<float>(), signalSize, filterSize, nNeurons, Ts);

	return output;
}

torch::Tensor corrCuda(torch::Tensor input, torch::Tensor filter, float Ts)
{
	CHECK_INPUT(input);
	CHECK_INPUT(filter);
	CHECK_DEVICE(input, filter);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);

	unsigned signalSize = input.size(-1);
	unsigned filterSize = filter.numel();
	unsigned nNeurons   = input.numel()/input.size(-1);
	corr<float>(output.data_ptr<float>(), input.data_ptr<float>(), filter.data_ptr<float>(), signalSize, filterSize, nNeurons, Ts);

	return output;
}

torch::Tensor shiftCuda(torch::Tensor input, torch::Tensor shiftLUT, float Ts)
{
	CHECK_INPUT(input);
	CHECK_INPUT(shiftLUT);
	CHECK_DEVICE(input, shiftLUT);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);

	if(shiftLUT.numel() == 1)
	{
		unsigned signalSize = input.size(-1);
		unsigned nNeurons   = input.numel()/signalSize;

		float shiftValue = shiftLUT.item<float>();

		shift<float>(output.data_ptr<float>(), input.data_ptr<float>(), shiftValue, signalSize, nNeurons, Ts);
	}
	else
	{
		unsigned signalSize = input.size(-1);
		unsigned nBatch     = input.size(0);
		unsigned nNeurons   = input.numel()/signalSize/nBatch;

		AT_ASSERTM(shiftLUT.numel() == nNeurons, "shift and number of neurons must be same");

		shift<float>(output.data_ptr<float>(), input.data_ptr<float>(), shiftLUT.data_ptr<float>(), signalSize, nNeurons, nBatch, Ts);
	}

	return output;
}

torch::Tensor shift1Cuda(torch::Tensor input, torch::Tensor shiftLUT)
{
	return shiftCuda(input, shiftLUT, 1.0f);
}

torch::Tensor shiftFlCuda(torch::Tensor input, float shiftLUT, float Ts)
{
	CHECK_INPUT(input);

	cudaSetDevice(input.device().index());

	auto output = torch::empty_like(input);

	unsigned signalSize = input.size(-1);
	unsigned nNeurons   = input.numel()/signalSize;

	shift<float>(output.data_ptr<float>(), input.data_ptr<float>(), shiftLUT, signalSize, nNeurons, Ts);

	return output;
}

torch::Tensor shiftFl1Cuda(torch::Tensor input, float shiftLUT)
{
	return shiftFlCuda(input, shiftLUT, 1.0f);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
	m.def("getSpikes" 	  ,  &getSpikesCuda      , 	"Get spikes (CUDA)");
	m.def("getSpikesLB"   ,  &getSpikesCudaLB    ,  "Get spikes with lower bound on neuron state(CUDA)");
	m.def("spikeGradsRefr",  &spikeGradsRefrCuda ,	"Get spike gradients with arbitrary refractory response(CUDA)");
	m.def("spikeGrads"    ,  &spikeGradsCuda     ,	"Get spike gradients (CUDA)");
	m.def("spikeGradsLB"  ,  &spikeGradsCudaLB   ,	"Get spike gradients with lower bounded vmem (CUDA)");
	m.def("spikeGradsFull",  &spikeGradsFullCuda ,	"Get spike gradients from input to output spikes (CUDA)");
	m.def("lifForward" 	  ,  &lifForwardCuda     , 	"LIF forward dynamics (CUDA)");
	m.def("leakyForward"  ,  &leakyForwardCuda   ,	"Forward pass of leaky integrator");
	m.def("leakyBackward" ,  &leakyBackwardCuda  ,	"Backward pass of leaky integrator");
	m.def("conv"     	  ,  &convCuda           , 	"Convolution in time (CUDA)");
	m.def("corr"     	  ,  &corrCuda           , 	"Correlation in time (CUDA)");
	m.def("shift"    	  ,  &shiftCuda          , 	"Element shift in time (CUDA)");
	m.def("shift"    	  ,  &shift1Cuda         , 	"Element shift in time (CUDA)");
	m.def("shift"    	  ,  &shiftFlCuda        , 	"Element shift in time (CUDA)");
	m.def("shift"    	  ,  &shiftFl1Cuda       , 	"Element shift in time (CUDA)");
}