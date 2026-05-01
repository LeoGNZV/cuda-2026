#include "gemm_cublas.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <vector>

// Статические переменные для переиспользования памяти
static float *d_A = nullptr;
static float *d_B = nullptr;
static float *d_C = nullptr;
static size_t prev_size = 0;

std::vector<float> GemmCUBLAS(const std::vector<float>& a,
                              const std::vector<float>& b,
                              int n) {
    size_t bytes = n * n * sizeof(float);
    
    // Переиспользование памяти - выделяем только если размер изменился
    if (bytes > prev_size || d_A == nullptr) {
        if (d_A) cudaFree(d_A);
        if (d_B) cudaFree(d_B);
        if (d_C) cudaFree(d_C);
        
        cudaMalloc(&d_A, bytes);
        cudaMalloc(&d_B, bytes);
        cudaMalloc(&d_C, bytes);
        prev_size = bytes;
    }
    
    // Копирование на GPU
    cudaMemcpy(d_A, a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, b.data(), bytes, cudaMemcpyHostToDevice);
    
    // cuBLAS умножение
    cublasHandle_t handle;
    cublasCreate(&handle);
    
    const float alpha = 1.0f;
    const float beta = 0.0f;
    
    // Для row-major матриц переставляем порядок
    cublasSgemm(handle, 
                CUBLAS_OP_N, CUBLAS_OP_N, 
                n, n, n, 
                &alpha, 
                d_B, n,    
                d_A, n,
                &beta, 
                d_C, n);
    
    // Копирование результата обратно
    std::vector<float> c(n * n);
    cudaMemcpy(c.data(), d_C, bytes, cudaMemcpyDeviceToHost);
    
    cublasDestroy(handle);
    
    return c;
}