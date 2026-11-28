#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "matmul.h"

namespace {

// Размер блока и тайла по умолчанию: 16 x 16. Для экспериментов поддерживаются
// ещё 8 и 32 (см. LaunchKernel).
constexpr int kDefaultTile = 16;

// Проверка кода ошибки CUDA: при ошибке печатаем сообщение и выходим с кодом 1.
void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}

// Наивное ядро: один поток считает один элемент C[row][col].
// Все числа читаются из глобальной памяти.
__global__ void NaiveMatMulKernel(const float* a, const float* b, float* c,
                                  int m, int k, int n) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < m && col < n) {
    float sum = 0.0f;
    for (int t = 0; t < k; ++t) {
      sum += a[row * k + t] * b[t * n + col];
    }
    c[row * n + col] = sum;
  }
}

// Ядро с тайлами. Блок считает тайл матрицы C размером kTile x kTile. Вдоль
// общего измерения k идём кусками по kTile: каждый поток кладёт в shared memory
// по одному числу из A и из B, потом все считают часть суммы из shared memory.
template <int kTile>
__global__ void TiledMatMulKernel(const float* a, const float* b, float* c,
                                  int m, int k, int n) {
  __shared__ float tile_a[kTile][kTile];
  __shared__ float tile_b[kTile][kTile];

  int row = blockIdx.y * kTile + threadIdx.y;
  int col = blockIdx.x * kTile + threadIdx.x;
  int num_tiles = (k + kTile - 1) / kTile;

  float sum = 0.0f;
  for (int tile_index = 0; tile_index < num_tiles; ++tile_index) {
    // Загрузка: числа за границей матрицы заменяем нулями.
    int a_col = tile_index * kTile + threadIdx.x;
    int b_row = tile_index * kTile + threadIdx.y;
    tile_a[threadIdx.y][threadIdx.x] =
        (row < m && a_col < k) ? a[row * k + a_col] : 0.0f;
    tile_b[threadIdx.y][threadIdx.x] =
        (b_row < k && col < n) ? b[b_row * n + col] : 0.0f;
    __syncthreads();  // Все загрузили, можно читать.

    for (int t = 0; t < kTile; ++t) {
      sum += tile_a[threadIdx.y][t] * tile_b[t][threadIdx.x];
    }
    __syncthreads();  // Все досчитали, можно загружать следующий тайл.
  }

  // Потоки за границей C только помогали грузить тайлы, записывать им нечего.
  if (row < m && col < n) {
    c[row * n + col] = sum;
  }
}

// Запускает ядро с блоком kTile x kTile потоков.
template <int kTile>
void LaunchWithTile(bool tiled, const float* a, const float* b, float* c, int m,
                    int k, int n) {
  dim3 block(kTile, kTile);
  dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
  if (tiled) {
    TiledMatMulKernel<kTile><<<grid, block>>>(a, b, c, m, k, n);
  } else {
    NaiveMatMulKernel<<<grid, block>>>(a, b, c, m, k, n);
  }
  Check(cudaGetLastError());
}

// Выбирает размер тайла (он должен быть известен при компиляции ядра).
void LaunchKernel(int tile, bool tiled, const float* a, const float* b,
                  float* c, int m, int k, int n) {
  if (tile == 8) {
    LaunchWithTile<8>(tiled, a, b, c, m, k, n);
  } else if (tile == 16) {
    LaunchWithTile<16>(tiled, a, b, c, m, k, n);
  } else if (tile == 32) {
    LaunchWithTile<32>(tiled, a, b, c, m, k, n);
  } else {
    std::fprintf(stderr, "Unsupported tile size: %d\n", tile);
    std::exit(1);
  }
}

}  // namespace

void MatMulOnGpuWithTile(const std::vector<float>& a,
                         const std::vector<float>& b, int m, int k, int n,
                         bool tiled, int tile, std::vector<float>* c) {
  const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);
  c->resize(static_cast<size_t>(m) * n);

  // Память на GPU.
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  Check(cudaMalloc(&d_a, bytes_a));
  Check(cudaMalloc(&d_b, bytes_b));
  Check(cudaMalloc(&d_c, bytes_c));

  // CPU -> GPU.
  Check(cudaMemcpy(d_a, a.data(), bytes_a, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_b, b.data(), bytes_b, cudaMemcpyHostToDevice));

  // Запуск ядра: блоков по каждой оси столько, чтобы покрыть всю матрицу C.
  LaunchKernel(tile, tiled, d_a, d_b, d_c, m, k, n);

  // GPU -> CPU. Здесь же ждём окончания ядра.
  Check(cudaMemcpy(c->data(), d_c, bytes_c, cudaMemcpyDeviceToHost));
  Check(cudaDeviceSynchronize());

  Check(cudaFree(d_a));
  Check(cudaFree(d_b));
  Check(cudaFree(d_c));
}

void MatMulOnGpu(const std::vector<float>& a, const std::vector<float>& b,
                 int m, int k, int n, bool tiled, std::vector<float>* c) {
  MatMulOnGpuWithTile(a, b, m, k, n, tiled, kDefaultTile, c);
}

float TimeMatMulKernel(const float* d_a, const float* d_b, float* d_c, int m,
                       int k, int n, bool tiled, int tile) {
  cudaEvent_t start;
  cudaEvent_t stop;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&stop));

  Check(cudaEventRecord(start));
  LaunchKernel(tile, tiled, d_a, d_b, d_c, m, k, n);
  Check(cudaEventRecord(stop));
  Check(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  Check(cudaEventElapsedTime(&milliseconds, start, stop));
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(stop));
  return milliseconds;
}