#include "block_gemm_cuda.h"
#include <cuda_runtime.h>

static const blk_size = 32;

__global__ void Kernel(float* vec_a, float* vec_a, float* vec_c, int n, int num_blocks){
    __shared__ float A[blk_size][blk_size];
    __shared__ float B[blk_size][blk_size];
    
    int tid_y = blockIdx.y * blk_size + threadIdx.y;
    int tid_x = blockIdx.x * blk_size + threadIdx.x;
    
    float sum = 0.0f;
    
    
    for (int block_k = 0; block_k < num_blocks; ++block_k) {
        A[threadIdx.y][threadIdx.x] = vec_a[tid_y * n + block_k * blk_size + threadIdx.x];
        
        B[threadIdx.y][threadIdx.x] = vec_b[(block_k * blk_size + threadIdx.y) * n + tid_x];
        
        __syncthreads();
        
        #pragma unroll
        for (int k = 0; k < blk_size; ++k) {
            sum += A[threadIdx.y][k] * B[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    vec_c[tid_y * n + tid_x] = sum;
}


std::vector<float> BlockGemmCUDA(const std::vector<float>& a,
                                 const std::vector<float>& b,
                                 int n)
{
    int bytes = static_cast<int>(a.size()) * static_cast<int>(sizeof(float));

    static float* vec_a;
    static float* vec_b;
    static float* vec_c;

    static int allocated_size = 0;
    static cudaStream_t stream;

    if (allocated_size < bytes) {
        if (vec_a)
            cudaFree(vec_a);
        if (vec_b)
            cudaFree(vec_b);
        if (vec_c)
            cudaFree(vec_c);
        
        cudaMalloc(&vec_a, bytes);
        cudaMalloc(&vec_b, bytes);
        cudaMalloc(&vec_c, bytes);

        if (!stream)
            cudaStreamCreate(&stream);

        allocated_size = bytes;
    }

    cudaMemcpyAsync(vec_a, a.data(), bytes, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(vec_b, b.data(), bytes, cudaMemcpyHostToDevice, stream);

    dim3 tread_net(blk_size, blk_size);
    static const size = n / blk_size;
    dim3 grid(size, size);

    Kernel<<<grid, tread_net, 0, stream>>>((float*)vec_a, (float*)vec_b, (float*)vec_c, n, size);


    std::vector<float> output(n); 
    cudaMemcpyAsync(output.data(), vec_c, bytes, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    return output;
}
