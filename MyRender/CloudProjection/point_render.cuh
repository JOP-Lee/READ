#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <torch/extension.h>

std::vector<torch::Tensor> GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor total_m,
	int tar_width, int tar_height, int block_size); // (tar_heigh ,tar_width)
