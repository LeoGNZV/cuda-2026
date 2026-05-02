#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>

std::vector<float> GemmCUBLAS(const std::vector<float>& a, const std::vector<float>& b, int n) 
{
    float* a_gpu;
    float* b_gpu;
    float* c_gpu;
    int bytes = n*n*sizeof(float);

    cudaMalloc(&a_gpu, bytes);
    cudaMalloc(&b_gpu, bytes);
    cudaMalloc(&c_gpu, bytes);

    cublasSetMatrix(n, n, sizeof(float), a.data(), n, a_gpu, n);
    cublasSetMatrix(n, n, sizeof(float), b.data(), n, b_gpu, n);

    cublasHandle_t handle;
    cublasCreate(&handle);

    float alpha = 1.0F;
    float betta = 0.0F;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, b_gpu, n, a_gpu, n, &betta, c_gpu, n);

    std::vector<float> c;
    c.resize(n*n);

    cublasGetMatrix(n, n, sizeof(float), c_gpu, n, c.data(), n);

    cublasDestroy(handle);
    cudaFree(a_gpu);
    cudaFree(b_gpu);
    cudaFree(c_gpu);

    return c;
}