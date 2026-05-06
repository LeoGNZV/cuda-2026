#include "gelu_cuda.h"

#define PI_COEFF 0.79788456f
#define CONST_044 0.044715f

__global__ void gelu_kernel_vec(const float4* __restrict__ in, float4* __restrict__ out, int n_vec) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (i < n_vec) {
        float4 val = in[i];
	      float4 res;
        float p_x = PI_COEFF * fmaf(CONST_044, val.x * val.x * val.x, val.x);
        res.x = val.x * (1.0f - 1.0f / (1.0f + __expf(2.0f * p_x)));

        float p_y = PI_COEFF * fmaf(CONST_044, val.y * val.y * val.y, val.y);
        res.y = val.y * (1.0f - 1.0f / (1.0f + __expf(2.0f * p_y)));

        float p_z = PI_COEFF * fmaf(CONST_044, val.z * val.z * val.z, val.z);
        res.z = val.z * (1.0f - 1.0f / (1.0f + __expf(2.0f * p_z)));

        float p_w = PI_COEFF * fmaf(CONST_044, val.w * val.w * val.w, val.w);
        res.w = val.w * (1.0f - 1.0f / (1.0f + __expf(2.0f * p_w)));

        out[i] = res;
    }
}

std::vector<float> GeluCUDA(const std::vector<float>& input) {
    size_t n = input.size();
    size_t bytes = n * sizeof(float);
    
    float *d_in, *d_out;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_in, input.data(), bytes, cudaMemcpyHostToDevice);

    int n_vec = n / 4;
    const int block_size = 256;
    int num_blocks = (n_vec + block_size - 1) / block_size;

    gelu_kernel_vec<<<num_blocks, block_size>>>(
        (const float4*)d_in, 
        (float4*)d_out, 
        n_vec
    );

    std::vector<float> result(n);
    cudaMemcpy(result.data(), d_out, bytes, cudaMemcpyDeviceToHost);
    
    cudaFree(d_in);
    cudaFree(d_out);
    return result;
}