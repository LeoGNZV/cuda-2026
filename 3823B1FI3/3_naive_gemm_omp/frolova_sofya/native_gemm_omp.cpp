#include "naive_gemm_omp.h"
#include <omp.h>

std::vector<float> NaiveGemmOMP(const std::vector<float>& a,
                                const std::vector<float>& b,
                                int n) {
    std::vector<float> c(n * n, 0.0f);
    const int block = 64; 

    // collapse(2) объединяет два внешних цикла (i0, j0) для лучшей балансировки
    #pragma omp parallel for collapse(2) schedule(static, 1)
    for (int i0 = 0; i0 < n; i0 += block) {
        for (int j0 = 0; j0 < n; j0 += block) {
            for (int k0 = 0; k0 < n; k0 += block) {
                // Внутри тайла — классический i‑k‑j
                for (int i = i0; i < i0 + block; ++i) {
                    float* crow = &c[i * n];
                    for (int k = k0; k < k0 + block; ++k) {
                        const float aik = a[i * n + k];
                        const float* bk = &b[k * n];
                        for (int j = j0; j < j0 + block; ++j) {
                            crow[j] += aik * bk[j];
                        }
                    }
                }
            }
        }
    }

    return c;
}