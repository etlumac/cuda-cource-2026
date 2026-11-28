#ifndef LAB2_INCLUDE_MATMUL_H_
#define LAB2_INCLUDE_MATMUL_H_

#include <vector>

// Умножает матрицы на GPU: C = A * B.
// tiled = false: наивное ядро (все числа читаются из глобальной памяти).
// tiled = true: ядро с тайлами в shared memory.
// A имеет размер m x k, B имеет размер k x n, C получится размером m x n.
// Все матрицы лежат в массивах по строкам: элемент (i, j) матрицы с шириной w
// находится под номером i * w + j.
// Результат записывается в *c (функция сама задаёт ему нужную длину).
void MatMulOnGpu(const std::vector<float>& a, const std::vector<float>& b,
                 int m, int k, int n, bool tiled, std::vector<float>* c);

// То же, но с явным размером блока и тайла: tile = 8, 16 или 32.
// MatMulOnGpu использует tile = 16.
void MatMulOnGpuWithTile(const std::vector<float>& a,
                         const std::vector<float>& b, int m, int k, int n,
                         bool tiled, int tile, std::vector<float>* c);

// Один запуск ядра на данных, которые уже лежат на GPU (d_a, d_b, d_c).
// Возвращает время работы только ядра в миллисекундах (без выделения памяти и
// копирований). Нужна для замеров в tests/bench.cu.
float TimeMatMulKernel(const float* d_a, const float* d_b, float* d_c, int m,
                       int k, int n, bool tiled, int tile);

#endif  // LAB2_INCLUDE_MATMUL_H_