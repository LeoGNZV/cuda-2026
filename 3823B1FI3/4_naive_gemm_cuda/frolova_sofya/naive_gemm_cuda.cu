#include "naive_gemm_cuda.h"
#include <cuda_runtime.h>

#define TILE_SIZE 4    
#define BLOCK_SIZE 256 


__global__ void NaiveGemmKernel(const float* __restrict__ A,
                                const float* __restrict__ B,
                                float* __restrict__ C,
                                int n) {
    int row = blockIdx.x;                       
    int col_tile = blockIdx.y * blockDim.x + threadIdx.x; 
    int col_base = col_tile * TILE_SIZE;       

    if (row >= n || col_base >= n) return;

    float a_val;
    float4 b_val;
    float4 c_val = make_float4(0.0f, 0.0f, 0.0f, 0.0f);


    #pragma unroll 4
    for (int k = 0; k < n; ++k) {
        a_val = A[row * n + k];
        b_val = *reinterpret_cast<const float4*>(&B[k * n + col_base]);
        c_val.x += a_val * b_val.x;
        c_val.y += a_val * b_val.y;
        c_val.z += a_val * b_val.z;
        c_val.w += a_val * b_val.w;
    }


    *reinterpret_cast<float4*>(&C[row * n + col_base]) = c_val;
}

__global__ void NaiveGemmSmallKernel(const float* A,
                                     const float* B,
                                     float* C,
                                     int n) {
    int row = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;

    if (row < n && col < n) {
        float sum = 0.0f;
        for (int k = 0; k < n; ++k) {
            sum += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = sum;
    }
}

std::vector<float> NaiveGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n) {
    if (n == 0) return {};

    std::vector<float> c(n * n, 0.0f);
    size_t bytes = n * n * sizeof(float);

    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaMemcpyAsync(d_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    if (n >= 4 && n % 4 == 0) {
        dim3 block(BLOCK_SIZE);
        dim3 grid(n, (n + BLOCK_SIZE * TILE_SIZE - 1) / (BLOCK_SIZE * TILE_SIZE));
        NaiveGemmKernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    } else {
        dim3 block(n < BLOCK_SIZE ? n : BLOCK_SIZE);
        dim3 grid(n, (n + block.x - 1) / block.x);
        NaiveGemmSmallKernel<<<grid, block, 0, stream>>>(d_a, d_b, d_c, n);
    }

    cudaMemcpyAsync(c.data(), d_c, bytes, cudaMemcpyDeviceToHost, stream);

    
    cudaStreamSynchronize(stream);

    cudaStreamDestroy(stream);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return c;
}