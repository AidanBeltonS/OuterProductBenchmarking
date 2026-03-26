// MFMA outer product benchmark without shared writeback

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

constexpr size_t iterations_before_sync = 1;

#define HIP_CHECK(fn)                                                          \
  {                                                                            \
    hipError_t res = fn;                                                       \
    if (res != hipSuccess) {                                                   \
      std::cerr << "HIP error code:" << hipGetErrorName(res) << std::endl;    \
      assert(false);                                                           \
    }                                                                          \
  }

int ceil_div(int x, int y) { return (x + y - 1) / y; }

size_t align_up(size_t value, size_t alignment) {
  assert(alignment != 0 && (alignment & (alignment - 1)) == 0);
  return (value + alignment - 1) & ~(alignment - 1);
}

// Calculate throughput in GiB/s from bytes transferred and time in nanoseconds
double calculate_throughput_gibps(size_t bytes, size_t time_ns) {
  return (static_cast<double>(bytes) * 1e9) /
         (time_ns * 1024.0 * 1024.0 * 1024.0);
}

// Calculate TFLOPS given total floating point operations and time in nanoseconds
double calculate_tflops(size_t flops, size_t time_ns) {
  // FLOP/s = flops * 1e9 / time_ns, TFLOP/s = FLOP/s / 1e12
  // => TFLOP/s = flops / (time_ns * 1e3)
  return static_cast<double>(flops) / (static_cast<double>(time_ns) * 1e3);
}

// Helper to convert type T to float for verification
template <typename T>
inline float to_float(T val) {
  return static_cast<float>(val);
}

template <>
inline float to_float(half val) { return __half2float(val); }

// Helper to convert float to type T for initialization
template <typename T>
inline T from_float(float val) {
  return static_cast<T>(val);
}

template <>
inline half from_float(float val) { return __float2half(val); }

// Verify outer product result: C[i][j] should equal a[i] * b[j]
template <typename T>
bool verify_outer_product(const T *C_host, const T *a_host, const T *b_host,
                          int n) {
  constexpr float tolerance = 1e-3f;
  int errors = 0;
  const int max_errors_to_print = 10;

  for (int i = 0; i < n && errors < max_errors_to_print; ++i) {
    for (int j = 0; j < n && errors < max_errors_to_print; ++j) {
      float expected = to_float(a_host[i]) * to_float(b_host[j]);
      float actual = to_float(C_host[i * n + j]);
      float diff = std::fabs(expected - actual);
      float rel_err = (expected != 0.0f) ? diff / std::fabs(expected) : diff;

      if (rel_err > tolerance && diff > tolerance) {
        if (errors < max_errors_to_print) {
          std::cerr << "  Mismatch at C[" << i << "][" << j << "]: "
                    << "expected " << expected << ", got " << actual
                    << " (diff=" << diff << ", rel_err=" << rel_err << ")"
                    << std::endl;
        }
        ++errors;
      }
    }
  }

  if (errors > 0) {
    std::cerr << "  Total errors: " << errors << " (showed first "
              << std::min(errors, max_errors_to_print) << ")" << std::endl;
    return false;
  }
  return true;
}

constexpr int n_compute_units = 256;

// Use Row major layout
// Outer product kernel: C[i][j] = a[i] * b[j]
// C is n x n, a is n elements, b is n elements
template <typename T, size_t N>
using my_vec = __attribute__((__vector_size__(N * sizeof(T)))) T;

// Each thread writes 64 values
constexpr uint32_t block_size = 256;
constexpr uint32_t wave_size = 64;
constexpr uint32_t tile_size_x = 128;
constexpr uint32_t tile_size_y = wave_size;
template <typename T>
__global__ void __launch_bounds__(4 * 64)
outer_product_kernel(T *__restrict__ C,
                     const T *__restrict__ a,
                     const T *__restrict__ b,
                     size_t size) {
  constexpr uint32_t n_waves = block_size / wave_size;
  static_assert(tile_size_y == wave_size, "tile_size_y must match wave_size");
  static_assert(tile_size_x == n_waves * 32,
                "tile_size_x must match MFMA wave column coverage");

  const uint32_t lane_32 = threadIdx.x & (32 - 1);
  const uint32_t lane_64 = threadIdx.x & (wave_size - 1);
  const uint32_t is_second_lane = (threadIdx.x & (wave_size - 1)) >> 5; // (threadIdx.x % 64) / 32
  const uint32_t wave_id = threadIdx.x >> 6;

  // 2D launch: each block computes one 64x128 tile.
  const uint32_t block_x = blockIdx.x;
  const uint32_t block_y = blockIdx.y;

  // This kernel computes one 64x128 tile:
  // - rows come from lane_64 (same for all waves)
  // - columns are split across waves in 32-column groups
  int row_id = lane_64 + block_y * tile_size_y;
  int col_id = lane_32 + wave_id * 32 + block_x * tile_size_x;

  T a_val = (row_id < size) ? a[row_id] : 0;
  T b_val = (col_id < size) ? b[col_id] : 0;

  my_vec<T, 32> acc = {0};
  acc = __builtin_amdgcn_mfma_f32_32x32x1f32(a_val, b_val, acc, 0, 0, 0);

  // Direct writeback: each thread stores its own MFMA fragment straight to C.
  for (int tile_half = 0; tile_half < 2; ++tile_half) {
    for (int j = 0; j < 4; ++j) {
      for (int i = 0; i < 4; ++i) {
        const uint32_t row = i + is_second_lane * 4 + j * 8 + tile_half * 32;
        const uint32_t col = lane_32 + wave_id * 32;
        const uint32_t global_row = block_y * tile_size_y + row;
        const uint32_t global_col = block_x * tile_size_x + col;
        if (global_row < size && global_col < size) {
          C[global_row * size + global_col] = acc[i + j * 4 + tile_half * 16];
        }
      }
    }
  }
}

template <typename T>
class run_iteration {
public:
  run_iteration(T *C_, T *a_, T *b_, int n_, hipStream_t stream_)
      : C(C_), a(a_), b(b_), n(n_), stream(stream_) {};

  void operator()() {
    for (int i = 0; i < iterations_before_sync; ++i) {
      dim3 block(block_size);
      dim3 grid(ceil_div(n, static_cast<int>(tile_size_x)),
                ceil_div(n, static_cast<int>(tile_size_y)));

      outer_product_kernel<T><<<grid, block, 0, stream>>>(
          C,
          a,
          b,
          static_cast<size_t>(n));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }

private:
  T *C;
  T *a;
  T *b;
  int n;
  hipStream_t stream;
};

template <typename T>
size_t run_benchmark(int n, int warmups, int iterations,
                     bool *verification_passed) {
  HIP_CHECK(hipSetDevice(0));

  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));

  // Allocate vectors a (n), b (n) and output matrix C (n x n)
  const size_t vector_bytes = sizeof(T) * n;
  const size_t matrix_bytes = sizeof(T) * n * n;

  T *a, *b, *C;
  T *a_host, *b_host, *C_host;

  // Allocate host memory
  a_host = static_cast<T *>(malloc(vector_bytes));
  b_host = static_cast<T *>(malloc(vector_bytes));
  C_host = static_cast<T *>(malloc(matrix_bytes));

  // Initialize vectors with known values: a[i] = (i % 16) + 1, b[j] = (j % 16) + 1
  for (int i = 0; i < n; ++i) {
    a_host[i] = from_float<T>(static_cast<float>((i % 16) + 1));
    b_host[i] = from_float<T>(static_cast<float>((i % 16) + 1 + 3));
  }

  HIP_CHECK(hipMalloc(&a, vector_bytes));
  HIP_CHECK(hipMalloc(&b, vector_bytes));
  HIP_CHECK(hipMalloc(&C, matrix_bytes));

  // Copy initialized vectors to device
  HIP_CHECK(hipMemcpy(a, a_host, vector_bytes, hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(b, b_host, vector_bytes, hipMemcpyHostToDevice));

  run_iteration<T> run_functor(C, a, b, n, stream);

  for (int i = 0; i < warmups; ++i) {
    run_functor();
  }

  unsigned long long total_ns = 0;

  for (int i = 0; i < iterations / iterations_before_sync; ++i) {
    auto start = std::chrono::high_resolution_clock::now();
    run_functor();
    size_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::high_resolution_clock::now() - start)
                    .count();
    total_ns += ns;
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify the result after all timed iterations
  HIP_CHECK(hipMemcpy(C_host, C, matrix_bytes, hipMemcpyDeviceToHost));
  *verification_passed = verify_outer_product(C_host, a_host, b_host, n);

  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(C));
  free(a_host);
  free(b_host);
  free(C_host);
  HIP_CHECK(hipStreamDestroy(stream));
  return total_ns / iterations;
}

template <typename T>
void run_benchmark_suite(const std::string &type_name,
                         const std::vector<int> &sizes, int warmups,
                         int iterations) {
  std::cout << "Outer Product Kernel Benchmark (mfma, " << type_name << ")"
            << std::endl;
  std::cout << "n, Matrix Size (n x n), Time (ns), Throughput (GiB/s), TFLOPS, "
               "Verified"
            << std::endl;
  for (const auto &n : sizes) {
    bool verified = false;
    auto time = run_benchmark<T>(n, warmups, iterations, &verified);
    // Total bytes: read a (n), read b (n), write C (n*n)
    const size_t bytes = sizeof(T) * (2 * n + (size_t)n * n);
    // FLOPs: one multiply per output element (n x n)
    const size_t flops = static_cast<size_t>(n) * static_cast<size_t>(n);
    double throughput = calculate_throughput_gibps(bytes, time);
    double tflops = calculate_tflops(flops, time);
    std::cout << n << "," << static_cast<size_t>(n) * static_cast<size_t>(n) << "," << time << "," << throughput << ","
              << tflops << "," << (verified ? "PASS" : "FAIL") << std::endl;
  }

  std::cout << std::endl;
}

int main() {
  int warmups = 1000;
  int iterations = 3000;
  // n is the vector size; output matrix is n x n
  std::vector<int> sizes{64,   128,  256,      512,      1024,     2048,
                         4096, 8192, 2 * 8192, 4 * 8192, 8 * 8192};

  run_benchmark_suite<float>("float", sizes, warmups, iterations);

  return 0;
}

