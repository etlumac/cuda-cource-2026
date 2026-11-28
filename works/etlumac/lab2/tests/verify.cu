// Проверка корректности: наши ядра (naive и tiled) сравниваются с cuBLAS.
// Эталон: cublasSgemm в режиме чистого FP32 (CUBLAS_PEDANTIC_MATH, без TF32).
// Допуск для элемента C[i][j] берётся из задания: gamma(2K) * S[i][j],
// где S[i][j] = сумма модулей произведений A[i][t] * B[t][j].
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "matmul.h"

namespace {

// u = 2^-24, единица округления для float.
// Размеры блока и тайла, которые проверяем.
constexpr int kTiles[] = {8, 16, 32};

constexpr double kUnitRoundoff = 5.9604644775390625e-08;

struct Size {
  int m;
  int k;
  int n;
};

// Результат сравнения: максимальная ошибка, максимум отношения ошибки к допуску
// и итог (ok = true, если все элементы в допуске и нет NaN/бесконечностей).
struct Result {
  double max_abs_err = 0.0;
  double max_ratio = 0.0;
  bool ok = true;
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

// Заполняет массив числами одного из трёх видов:
// 0 - случайные от -10 до 10, 1 - целые от -4 до 4 (результат считается точно),
// 2 - примерно половина нулей, остальные случайные.
void FillData(int kind, unsigned seed, std::vector<float>* values) {
  std::mt19937 gen(seed);
  std::uniform_real_distribution<float> real_dist(-10.0f, 10.0f);
  std::uniform_int_distribution<int> int_dist(-4, 4);
  for (float& value : *values) {
    if (kind == 0) {
      value = real_dist(gen);
    } else if (kind == 1) {
      value = static_cast<float>(int_dist(gen));
    } else {
      value = (int_dist(gen) > 0) ? real_dist(gen) : 0.0f;
    }
  }
}

std::vector<float> Abs(const std::vector<float>& values) {
  std::vector<float> result = values;
  for (float& value : result) {
    value = std::fabs(value);
  }
  return result;
}

// Эталонное умножение через cuBLAS. Наши матрицы лежат по строкам, а cuBLAS
// читает массивы по столбцам. Массив по строкам, прочитанный по столбцам, - это
// транспонированная матрица, поэтому считаем C^T = B^T * A^T: матрицы передаём
// в обратном порядке (сначала B, потом A), результат ляжет по строкам.
void ReferenceGemm(cublasHandle_t handle, const std::vector<float>& a,
                   const std::vector<float>& b, int m, int k, int n,
                   std::vector<float>* c) {
  const size_t bytes_a = static_cast<size_t>(m) * k * sizeof(float);
  const size_t bytes_b = static_cast<size_t>(k) * n * sizeof(float);
  const size_t bytes_c = static_cast<size_t>(m) * n * sizeof(float);
  c->resize(static_cast<size_t>(m) * n);

  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  Check(cudaMalloc(&d_a, bytes_a));
  Check(cudaMalloc(&d_b, bytes_b));
  Check(cudaMalloc(&d_c, bytes_c));
  Check(cudaMemcpy(d_a, a.data(), bytes_a, cudaMemcpyHostToDevice));
  Check(cudaMemcpy(d_b, b.data(), bytes_b, cudaMemcpyHostToDevice));

  const float alpha = 1.0f;
  const float beta = 0.0f;
  CheckBlas(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, d_b,
                        n, d_a, k, &beta, d_c, n));
  Check(cudaMemcpy(c->data(), d_c, bytes_c, cudaMemcpyDeviceToHost));

  Check(cudaFree(d_a));
  Check(cudaFree(d_b));
  Check(cudaFree(d_c));
}

// gamma(q) = q * u / (1 - q * u).
double Gamma(double q) { return q * kUnitRoundoff / (1.0 - q * kUnitRoundoff); }

// Сравнивает наш результат с эталоном. sums - массив S (сумма модулей
// произведений), посчитанный тем же cuBLAS на матрицах из модулей.
Result Compare(const std::vector<float>& mine, const std::vector<float>& ref,
               const std::vector<float>& sums, int k) {
  Result result;
  const double gamma = Gamma(2.0 * k);
  for (size_t i = 0; i < mine.size(); ++i) {
    if (!std::isfinite(mine[i])) {
      result.ok = false;
      continue;
    }
    double err = std::fabs(static_cast<double>(mine[i]) - ref[i]);
    // Запас 0.1 %: S тоже посчитан в float и сам чуть неточен.
    double tolerance = gamma * sums[i] * 1.001;
    result.max_abs_err = std::max(result.max_abs_err, err);
    if (tolerance > 0.0) {
      result.max_ratio = std::max(result.max_ratio, err / tolerance);
    }
    if (err > tolerance) {
      result.ok = false;
    }
  }
  return result;
}

Result Merge(const Result& x, const Result& y) {
  Result result;
  result.max_abs_err = std::max(x.max_abs_err, y.max_abs_err);
  result.max_ratio = std::max(x.max_ratio, y.max_ratio);
  result.ok = x.ok && y.ok;
  return result;
}

// Один тест: генерирует A и B, считает эталон, допуск и наш результат.
Result RunOne(cublasHandle_t handle, const Size& size, int kind, bool tiled,
              int tile) {
  std::vector<float> a(static_cast<size_t>(size.m) * size.k);
  std::vector<float> b(static_cast<size_t>(size.k) * size.n);
  unsigned seed = static_cast<unsigned>(size.m * 1000003 + size.k * 10007 +
                                        size.n * 101 + kind);
  FillData(kind, seed, &a);
  FillData(kind, seed + 1, &b);

  std::vector<float> ref;
  std::vector<float> sums;
  std::vector<float> mine;
  ReferenceGemm(handle, a, b, size.m, size.k, size.n, &ref);
  ReferenceGemm(handle, Abs(a), Abs(b), size.m, size.k, size.n, &sums);
  MatMulOnGpuWithTile(a, b, size.m, size.k, size.n, tiled, tile, &mine);
  return Compare(mine, ref, sums, size.k);
}

void PrintRow(const Size& size, bool tiled, const Result& result) {
  char name[32];
  std::snprintf(name, sizeof(name), "%dx%dx%d", size.m, size.k, size.n);
  std::printf("%-14s %-7s %-14.3g %-14.3g %s\n", name,
              tiled ? "tiled" : "naive", result.max_abs_err, result.max_ratio,
              result.ok ? "OK" : "FAIL");
}

}  // namespace

int main() {
  cublasHandle_t handle;
  CheckBlas(cublasCreate(&handle));
  CheckBlas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));

  const Size sizes[] = {{1, 1, 1},      {2, 3, 2},       {1, 100, 1},
                        {16, 16, 16},   {15, 17, 33},    {17, 33, 15},
                        {31, 32, 33},   {127, 65, 257},  {100, 100, 100},
                        {64, 4000, 64}, {256, 256, 256}, {300, 500, 200},
                        {512, 512, 512}};
  bool all_ok = true;

  std::printf("Reference: cublasSgemm, FP32, CUBLAS_PEDANTIC_MATH (no TF32)\n");
  std::printf("Tolerance: gamma(2K) * sum_t |A[i][t] * B[t][j]|\n");
  std::printf("Sizes are M x K x N; max_ratio = max(error / tolerance)\n");
  std::printf("Every row covers block/tile sizes 8, 16 and 32 together\n\n");
  std::printf("%-14s %-7s %-14s %-14s %s\n", "size", "mode", "max_abs_err",
              "max_ratio", "result");
  for (const Size& size : sizes) {
    for (int mode = 0; mode < 2; ++mode) {
      Result total;
      for (int tile : kTiles) {
        for (int kind = 0; kind < 3; ++kind) {
          total = Merge(total, RunOne(handle, size, kind, mode == 1, tile));
        }
      }
      PrintRow(size, mode == 1, total);
      all_ok = all_ok && total.ok;
    }
  }
  CheckBlas(cublasDestroy(handle));
  std::printf(all_ok ? "ALL OK\n" : "SOME TESTS FAILED\n");
  return all_ok ? 0 : 1;
}