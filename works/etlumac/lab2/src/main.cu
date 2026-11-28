#include <cstdio>
#include <string>
#include <vector>

#include "matmul.h"

namespace {

// Параметры командной строки.
struct Options {
  std::string input_path;
  std::string output_path;
  std::string mode = "tiled";
};

enum class ParseResult { kOk, kHelp, kError };

void PrintHelp() {
  std::printf(
      "Lab 2: matrix multiplication C = A * B on GPU\n"
      "\n"
      "Usage:\n"
      "  submission --help\n"
      "  submission --input <file> --output <file> [--mode <mode>]\n"
      "\n"
      "Arguments:\n"
      "  --input <file>   input: 'M K N', then A (M x K), then B (K x N), by "
      "rows\n"
      "  --output <file>  output: 'M N', then C (M x N) by rows\n"
      "  --mode <mode>    naive or tiled (default: tiled)\n"
      "\n"
      "Exit codes: 0 - success, 1 - runtime or write error,\n"
      "2 - bad arguments or bad input.\n");
}

// Разбор аргументов командной строки.
ParseResult ParseArgs(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg == "--help") {
      return ParseResult::kHelp;
    } else if (arg == "--input" && i + 1 < argc) {
      options->input_path = argv[++i];
    } else if (arg == "--output" && i + 1 < argc) {
      options->output_path = argv[++i];
    } else if (arg == "--mode" && i + 1 < argc) {
      options->mode = argv[++i];
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n", arg.c_str());
      return ParseResult::kError;
    }
  }
  if (options->input_path.empty() || options->output_path.empty()) {
    std::fprintf(stderr,
                 "Both --input and --output are required (see --help)\n");
    return ParseResult::kError;
  }
  if (options->mode != "naive" && options->mode != "tiled") {
    std::fprintf(stderr, "Unknown mode: %s (use naive or tiled)\n",
                 options->mode.c_str());
    return ParseResult::kError;
  }
  return ParseResult::kOk;
}

// Читает из файла count чисел и добавляет их в конец вектора.
bool ReadNumbers(FILE* file, long long count, std::vector<float>* numbers) {
  for (long long i = 0; i < count; ++i) {
    float value = 0.0f;
    if (std::fscanf(file, "%f", &value) != 1) {
      std::fprintf(stderr, "Bad input: expected %lld numbers, got %lld\n",
                   count, i);
      return false;
    }
    numbers->push_back(value);
  }
  return true;
}

// Читает файл: "M K N", затем матрица A (M x K), затем матрица B (K x N).
bool ReadInput(const std::string& path, int* m, int* k, int* n,
               std::vector<float>* a, std::vector<float>* b) {
  FILE* file = std::fopen(path.c_str(), "r");
  if (file == nullptr) {
    std::fprintf(stderr, "Cannot open input file: %s\n", path.c_str());
    return false;
  }
  bool ok = std::fscanf(file, "%d %d %d", m, k, n) == 3;
  if (!ok || *m <= 0 || *k <= 0 || *n <= 0) {
    std::fprintf(stderr,
                 "Bad input: the first line must be 'M K N' (positive)\n");
    std::fclose(file);
    return false;
  }
  ok = ReadNumbers(file, static_cast<long long>(*m) * *k, a) &&
       ReadNumbers(file, static_cast<long long>(*k) * *n, b);
  std::fclose(file);
  return ok;
}

// Пишет результат: строка "M N", затем матрица C по строкам.
bool WriteOutput(const std::string& path, int m, int n,
                 const std::vector<float>& c) {
  FILE* file = std::fopen(path.c_str(), "w");
  if (file == nullptr) {
    std::fprintf(stderr, "Cannot open output file: %s\n", path.c_str());
    return false;
  }
  std::fprintf(file, "%d %d\n", m, n);
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      std::fprintf(file, j + 1 < n ? "%.9g " : "%.9g\n", c[i * n + j]);
    }
  }
  if (std::fclose(file) != 0) {
    std::fprintf(stderr, "Cannot write output file: %s\n", path.c_str());
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  ParseResult parsed = ParseArgs(argc, argv, &options);
  if (parsed == ParseResult::kHelp) {
    PrintHelp();
    return 0;
  }
  if (parsed == ParseResult::kError) {
    return 2;
  }

  int m = 0;
  int k = 0;
  int n = 0;
  std::vector<float> a;
  std::vector<float> b;
  if (!ReadInput(options.input_path, &m, &k, &n, &a, &b)) {
    return 2;
  }

  std::vector<float> c;
  MatMulOnGpu(a, b, m, k, n, options.mode == "tiled", &c);

  if (!WriteOutput(options.output_path, m, n, c)) {
    return 1;
  }
  std::printf("Done: M=%d K=%d N=%d, mode=%s\n", m, k, n, options.mode.c_str());
  return 0;
}