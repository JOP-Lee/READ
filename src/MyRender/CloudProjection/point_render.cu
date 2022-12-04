#include "point_render.cuh"
#include <stdio.h>

#include "helper_math.h"

#include "cooperative_groups.h"

struct Matrix4x4
{
public:
	float4 col[4];
	__device__ __forceinline__
		Matrix4x4()
	{
		col[0] = col[1] = col[2] = col[3] = make_float4(0, 0, 0, 0);
	}
	__device__ __forceinline__
		Matrix4x4(float4 a, float4 b, float4 c, float4 d)
	{
		col[0].x = a.x;
		col[0].y = a.y;
		col[0].z = a.z;
		col[0].w = a.w;

		col[1].x = b.x;
		col[1].y = b.y;
		col[1].z = b.z;
		col[1].w = b.w;

		col[2].x = c.x;
		col[2].y = c.y;
		col[2].z = c.z;
		col[2].w = c.w;

		col[3].x = d.x;
		col[3].y = d.y;
		col[3].z = d.z;
		col[3].w = d.w;
	}

	__device__ __forceinline__
		Matrix4x4 transpose() const
	{
		Matrix4x4 res;

		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = col[3].x;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = col[3].y;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = col[3].z;

		res.col[3].x = col[0].w;
		res.col[3].y = col[1].w;
		res.col[3].z = col[2].w;
		res.col[3].w = col[3].w;
		return res;

	}
	__device__ __forceinline__
		Matrix4x4 inv() const
	{
		Matrix4x4 res;
		res.col[0].x = col[0].x;
		res.col[0].y = col[1].x;
		res.col[0].z = col[2].x;
		res.col[0].w = 0;

		res.col[1].x = col[0].y;
		res.col[1].y = col[1].y;
		res.col[1].z = col[2].y;
		res.col[1].w = 0;

		res.col[2].x = col[0].z;
		res.col[2].y = col[1].z;
		res.col[2].z = col[2].z;
		res.col[2].w = 0;

		res.col[3].x = -dot(col[0], col[3]);
		res.col[3].y = -dot(col[1], col[3]);
		res.col[3].z = -dot(col[2], col[3]);
		res.col[3].w = 1;
		return res;
	}
};


typedef struct CamMatrix
{
	float4 m[4];
	__device__ __forceinline__
		Matrix4x4 getRT() const
	{
		return Matrix4x4(m[0],m[1],m[2],m[3]);
	}

};

namespace math
{
	__device__ __forceinline__
	float4 MatrixMul(const Matrix4x4& mat, float4& p)
	{
		Matrix4x4 res = mat;
		float4 ans;
		ans.x = dot(res.col[0], p);
		ans.y = dot(res.col[1], p);
		ans.z = dot(res.col[2], p);
		ans.w = dot(res.col[3], p);

		ans = ans / ans.w;
		return ans;
	}
}


__global__ 
void DepthProject(float3 * point_clouds, int num_points, int batch_id,
	CamMatrix* total_m, int tar_width, int tar_height,
	int* mutex_map, float* out_depth, float* out_index)
{
	// int ids = blockDim.x * blockIdx.x + threadIdx.x; //  index of point
	cooperative_groups::grid_group grid = cooperative_groups::this_grid();
	for(int ids = grid.thread_rank(); ids<num_points; ids+= grid.size())
	{
		if (ids > num_points)	return;
		float4 p = make_float4(point_clouds[ids], 1.0);

		float4 camp =  math::MatrixMul(total_m->getRT(), p);

		if(camp.x<-1 || camp.x>1 || camp.y<-1 || camp.y>1 || camp.z<-1 || camp.z>1) return;

		float u = tar_width*(camp.x+1)*0.5;
		float v = tar_height*(1-camp.y)*0.5;
		float tdepth = ((camp.z+1)*0.5);

		int xx = int(u);
		int yy = int(v);
		if (xx < 0 || xx >= tar_width || yy < 0 || yy >= tar_height)	return;
		int ind = batch_id * tar_width * tar_height + yy * tar_width + xx ;
		bool isSet = false;
		do
		{
			if ((isSet = atomicCAS(mutex_map + ind, 0, 1)) == false)
			{
				// critical section goes here
				if (out_depth[ind] > tdepth || out_depth[ind]==0)
				{
					out_depth[ind] = tdepth;
					out_index[ind] = (float)ids ; // 0 denote empty
				}
			}
			if (isSet)
			{
				mutex_map[ind] = 0;
			}
		} while (!isSet);
	}
}

std::vector<torch::Tensor> GPU_PCPR(
	torch::Tensor in_points, //(num_points,3)
	torch::Tensor total_m, int tar_width, int tar_height, int block_size)
{
	const auto num_points = in_points.size(0);
	const auto batch_size = total_m.size(0);

	torch::Tensor out_index = torch::zeros({batch_size, tar_height, tar_width}).to(torch::kCUDA);
	torch::Tensor out_depth = torch::zeros({batch_size, tar_height, tar_width}).to(torch::kCUDA);
	
	int default_block_size = block_size;
	int block_num = (num_points+default_block_size-1) / default_block_size;

	int *mutex_map;
	cudaMalloc(&mutex_map, sizeof(int) * batch_size * tar_width * tar_height);
	cudaMemset(mutex_map, 0, sizeof(int) * batch_size * tar_width * tar_height);

	for(int b=0; b<batch_size; ++b){
		DepthProject << <block_num, default_block_size >> > (
			(float3*)in_points.data<float>(), num_points, b,
			(CamMatrix*)total_m[b].data<float>(),
			tar_width, tar_height, mutex_map, 
			out_depth.data<float>(), out_index.data<float>());
	}
	cudaFree(mutex_map);
	cudaDeviceSynchronize();

	out_index = out_index.to(torch::kCPU);
	out_depth = out_depth.to(torch::kCPU);
	
	return {out_index, out_depth};
}

