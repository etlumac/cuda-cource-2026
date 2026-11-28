// Эксперимент с банками shared memory.
// Shared memory поделена на 32 банка: слово номер w лежит в банке w % 32.
// Если потоки одного warp читают разные слова из одного банка, обращения
// выполняются по очереди (bank conflict).
// Ядро много раз читает tile из shared memory. Три варианта:
//   1. tile[32][32], чтение по строке:  потоки warp берут соседние слова.
//   2. tile[32][32], чтение по столбцу: слова лежат через 32, все в одном
//   банке.
//   3. tile[32][33], чтение по столбцу: слова лежат через 33, банки разные.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int tile_width = 32;
constexpr int block_count = 1024;
constexpr int iterations_per_thread = 2000;
constexpr int warmup_runs = 3;
constexpr int timed_runs = 20;

void Check(cudaError_t err) {
  if (err != cudaSuccess) {
    std::fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(err));
    std::exit(1);
  }
}

// kPad - сколько лишних столбцов добавлено к строке tile (0 или 1).
// kByColumn - читаем по столбцу (true) или по строке (false).
// Блок 32 x 32 потоков, поэтому один warp - это потоки с одним threadIdx.y и
// всеми threadIdx.x от 0 до 31. Индекс (x + i) меняется на каждом шаге, чтобы
// компилятор не вынес чтение из цикла.
template <int kPad, bool kByColumn>
__global__ void SharedAccessKernel(float* out, int iterations) {
  __shared__ float tile[tile_width][tile_width + kPad];

  int x = threadIdx.x;
  int y = threadIdx.y;
  tile[y][x] = static_cast<float>(y * tile_width + x);
  __syncthreads();

  float sum = 0.0f;
  for (int i = 0; i < iterations; ++i) {
    int shifted = (x + i) & (tile_width - 1);
    sum += kByColumn ? tile[shifted][y] : tile[y][shifted];
  }
  out[(blockIdx.x * tile_width + y) * tile_width + x] = sum;
}

// Один запуск ядра, возвращает время в миллисекундах.
template <int kPad, bool kByColumn>
float TimeOnce(float* d_out) {
  cudaEvent_t start;
  cudaEvent_t stop;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&stop));

  Check(cudaEventRecord(start));
  SharedAccessKernel<kPad, kByColumn>
      <<<block_count, dim3(tile_width, tile_width)>>>(d_out,
                                                      iterations_per_thread);
  Check(cudaGetLastError());
  Check(cudaEventRecord(stop));
  Check(cudaEventSynchronize(stop));

  float milliseconds = 0.0f;
  Check(cudaEventElapsedTime(&milliseconds, start, stop));
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(stop));
  return milliseconds;
}

// 3 прогревочных запуска печатаем, по следующим 20 считаем медиану.
// Результат ядра копируем в *result, чтобы потом сравнить варианты.
template <int kPad, bool kByColumn>
double Measure(const char* name, float* d_out, std::vector<float>* result) {
  float warmup[warmup_runs];
  for (int i = 0; i < warmup_runs; ++i) {
    warmup[i] = TimeOnce<kPad, kByColumn>(d_out);
  }
  std::vector<float> times;
  for (int i = 0; i < timed_runs; ++i) {
    times.push_back(TimeOnce<kPad, kByColumn>(d_out));
  }
  std::sort(times.begin(), times.end());
  const double median_ms = times[times.size() / 2];

  result->resize(static_cast<size_t>(block_count) * tile_width * tile_width);
  Check(cudaMemcpy(result->data(), d_out, result->size() * sizeof(float),
                   cudaMemcpyDeviceToHost));
  std::printf("%-26s %-7d %9.4f %9.4f %9.4f %11.4f\n", name, tile_width + kPad,
              warmup[0], warmup[1], warmup[2], median_ms);
  return median_ms;
}

}  // namespace

int main() {
  float* d_out = nullptr;
  Check(cudaMalloc(&d_out, static_cast<size_t>(block_count) * tile_width *
                               tile_width * sizeof(float)));

  std::printf("Blocks: %d (32x32 threads), iterations per thread: %d\n",
              block_count, iterations_per_thread);
  std::printf("%-26s %-7s %9s %9s %9s %11s\n", "variant", "stride", "run1_ms",
              "run2_ms", "run3_ms", "median_ms");

  std::vector<float> row_result;
  std::vector<float> column_result;
  std::vector<float> padded_result;
  const double row_ms =
      Measure<0, false>("[32][32] by row", d_out, &row_result);
  const double column_ms =
      Measure<0, true>("[32][32] by column", d_out, &column_result);
  const double padded_ms =
      Measure<1, true>("[32][33] by column", d_out, &padded_result);

  std::printf("\nby column [32][32] / by row:      %.2f x\n",
              column_ms / row_ms);
  std::printf("by column [32][33] / by row:      %.2f x\n", padded_ms / row_ms);
  std::printf("by column [32][32] / [32][33]:    %.2f x\n",
              column_ms / padded_ms);
  std::printf("Same result for [32][32] and [32][33] by column: %s\n",
              column_result == padded_result ? "yes" : "NO");

  Check(cudaFree(d_out));
  return column_result == padded_result ? 0 : 1;
}
