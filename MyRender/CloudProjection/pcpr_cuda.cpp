#include <torch/extension.h>
#include <vector>
#include "point_render.cuh"
#include <iostream>


// CUDA forward declarations

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor ,
    int tar_width, int tar_height, int block_size
    );

// C++ interface

#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_FLOAT(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Float, #x " must be a float tensor")
#define CHECK_Int(x) AT_ASSERTM(x.type().scalarType()==torch::ScalarType::Int, #x " must be a Int tensor")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x);

std::vector<torch::Tensor> pcpr_cuda_forward(
    torch::Tensor in_points, //(num_points,3)
    torch::Tensor total_m,
    int tar_width, int tar_height, int block_size
    ) 
{
  in_points = in_points.to(torch::kCUDA);
  total_m = total_m.to(torch::kCUDA);

  CHECK_INPUT(in_points); CHECK_FLOAT(in_points);
  CHECK_INPUT(total_m); CHECK_FLOAT(total_m);
  AT_ASSERTM(total_m.sizes().size()==3, "batch_size check");

  return GPU_PCPR(in_points,	total_m, tar_width, tar_height, block_size);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &pcpr_cuda_forward, "PCPR forward (CUDA)");
}

