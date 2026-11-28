// Замер времени: наивное ядро, ядро с тайлами и cuBLAS.
// Мерим только работу на GPU (cudaEvent). Выделение памяти и копирования
// CPU <-> GPU в время НЕ входят: данные один раз копируются до замеров.
// Для каждого варианта: первые 3 запуска (прогрев) печатаем отдельно,
// затем 20 замеряемых запусков, по ним считаем медиану и GFLOP/s.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "matmul.h"

namespace {

constexpr int warmup_runs = 3;
constexpr int timed_runs = 20;
// Размеры блока и тайла для эксперимента.
constexpr int tile_sizes[] = {8, 16, 32};

struct Size {
  int m;
  int k;
  int n;
};

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}

void CheckBlas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuBLAS error, status %d\n", static_cast<int>(status));
    std::exit(1);
  }
}

// Данные на GPU, общие для всех вариантов.
struct DeviceData {
  float* a = nullptr;
  float* b = nullptr;
  float* c = nullptr;
};

DeviceData AllocateData(const Size& size) {
  const size_t count_a = static_cast<size_t>(size.m) * size.k;
  const size_t count_b = static_cast<size_t>(size.k) * size.n;
  const size_t count_c = static_cast<size_t>(size.m) * size.n;
  std::vector<float> host_a(count_a, 1.0f);
  std::vector<float> host_b(count_b, 2.0f);

  DeviceData data;
  Check(cudaMalloc(&data.a, count_a * sizeof(float)));
  Check(cudaMalloc(&data.b, count_b * sizeof(float)));
  Check(cudaMalloc(&data.c, count_c * sizeof(float)));
  Check(cudaMemcpy(data.a, host_a.data(), count_a * sizeof(float),
                   cudaMemcpyHostToDevice));
  Check(cudaMemcpy(data.b, host_b.data(), count_b * sizeof(float),
                   cudaMemcpyHostToDevice));
  return data;
}

void FreeData(const DeviceData& data) {
  Check(cudaFree(data.a));
  Check(cudaFree(data.b));
  Check(cudaFree(data.c));
}

// Один запуск cuBLAS (с теми же событиями, что и у наших ядер).
// Порядок аргументов: см. комментарий к ReferenceGemm в tests/verify.cu.
float TimeCublas(cublasHandle_t handle, const DeviceData& data,
                 const Size& size) {
  const float alpha = 1.0f;
  const float beta = 0.0f;
  cudaEvent_t start;
  cudaEvent_t stop;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&stop));

  Check(cudaEventRecord(start));
  CheckBlas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size.n, size.m,
                        size.k, &alpha, data.b, size.n, data.a, size.k, &beta,
                        data.c, size.n));
  Check(cudaEventRecord(stop));
  Check(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  Check(cudaEventElapsedTime(&milliseconds, start, stop));
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(stop));
  return milliseconds;
}

// Запускает вариант (0 - naive, 1 - tiled, 2 - cuBLAS) один раз.
// tile - размер блока и тайла для наших ядер (для cuBLAS не используется).
float RunVariant(int variant, int tile, cublasHandle_t handle,
                 const DeviceData& data, const Size& size) {
  if (variant == 2) {
    return TimeCublas(handle, data, size);
  }
  return TimeMatMulKernel(data.a, data.b, data.c, size.m, size.k, size.n,
                          variant == 1, tile);
}

double Median(std::vector<float> values) {
  std::sort(values.begin(), values.end());
  return values[values.size() / 2];
}

void BenchVariant(int variant, int tile, cublasHandle_t handle,
                  const DeviceData& data, const Size& size) {
  const char* names[] = {"naive", "tiled", "cublas"};

  float warmup[warmup_runs];
  for (int i = 0; i < warmup_runs; ++i) {
    warmup[i] = RunVariant(variant, tile, handle, data, size);
  }
  std::vector<float> times;
  for (int i = 0; i < timed_runs; ++i) {
    times.push_back(RunVariant(variant, tile, handle, data, size));
  }

  const double median_ms = Median(times);
  const double flops = 2.0 * size.m * size.n * size.k;
  const double gflops = flops / (median_ms * 1e-3) / 1e9;
  char name[32];
  std::snprintf(name, sizeof(name), "%dx%dx%d", size.m, size.k, size.n);
  char tile_name[8];
  if (variant == 2) {
    std::snprintf(tile_name, sizeof(tile_name), "-");
  } else {
    std::snprintf(tile_name, sizeof(tile_name), "%d", tile);
  }
  std::printf("%-14s %-7s %-5s %9.4f %9.4f %9.4f %11.4f %10.1f\n", name,
              names[variant], tile_name, warmup[0], warmup[1], warmup[2],
              median_ms, gflops);
}

}  // namespace

int main() {
  cublasHandle_t handle;
  CheckBlas(cublasCreate(&handle));
  CheckBlas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

  const Size sizes[] = {{512, 512, 512},
                        {1024, 1024, 1024},
                        {2048, 2048, 2048},
                        {1000, 700, 1300}};

  std::printf(
      "Time is kernel only (cudaEvent); no allocation, no CPU<->GPU copy\n");
  std::printf("Warmup runs: %d (printed), timed runs: %d (median)\n\n",
              warmup_runs, timed_runs);
  std::printf("%-14s %-7s %-5s %9s %9s %9s %11s %10s\n", "size", "variant",
              "tile", "run1_ms", "run2_ms", "run3_ms", "median_ms", "GFLOP/s");
  for (const Size& size : sizes) {
    DeviceData data = AllocateData(size);
    for (int variant = 0; variant < 2; ++variant) {
      for (int tile : tile_sizes) {
        BenchVariant(variant, tile, handle, data, size);
      }
    }
    BenchVariant(2, 0, handle, data, size);
    FreeData(data);
  }
  CheckBlas(cublasDestroy(handle));
  return 0;
}