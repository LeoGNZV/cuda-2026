#include <cuda_runtime.h>
#include <cufft.h>
#include "fft_cufft.h"

__global__ void rescale_kernel(cufftComplex* buffer, int count, float scale_factor) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) 
    {
        buffer[i].x *= scale_factor;
        buffer[i].y *= scale_factor;
    }
}

std::vector<float> FffCUFFT(const std::vector<float>& input, int batch) 
{
    const size_t total_elements = input.size();
    const size_t data_size_bytes = total_elements * sizeof(float);
    
    const int complex_len = static_cast<int>(total_elements / 2);
    const int signal_len = complex_len / batch;
    const float norm_coeff = 1.0f / static_cast<float>(signal_len);

    cufftComplex* gpu_buffer = nullptr;
    cudaMalloc(&gpu_buffer, data_size_bytes);
    cudaMemcpy(gpu_buffer, input.data(), data_size_bytes, cudaMemcpyHostToDevice);

    cufftHandle fft_plan;
    cufftPlan1d(&fft_plan, signal_len, CUFFT_C2C, batch);

    cufftExecC2C(fft_plan, gpu_buffer, gpu_buffer, CUFFT_FORWARD);
    cufftExecC2C(fft_plan, gpu_buffer, gpu_buffer, CUFFT_INVERSE);

    const int tpb = 256; 
    const int bpg = (complex_len + tpb - 1) / tpb; 
    rescale_kernel<<<bpg, tpb>>>(gpu_buffer, complex_len, norm_coeff);

    std::vector<float> output_vec(total_elements);
    cudaMemcpy(output_vec.data(), gpu_buffer, data_size_bytes, cudaMemcpyDeviceToHost);

    cufftDestroy(fft_plan);
    cudaFree(gpu_buffer);

    return output_vec;
}