// Combined shared-memory + vectorized outer product benchmark

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

constexpr size_t iterations_before_sync = 1;

#define HIP_CHECK(fn)                                                          \
  {                                                                            \
    hipError_t res = fn;                                                       \
    if (res != hipSuccess) {                                                   \
      std::cerr << "HIP error code:" << hipGetErrorName(res) << std::endl;     \
      assert(false);                                                           \
    }                                                                          \
  }

int ceil_div(int x, int y) { return (x + y - 1) / y; }

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

constexpr int n_warps = 8;
constexpr int warp_size = 64;
constexpr int max_block_size = n_warps * warp_size;

// Use row-major layout
// Outer product kernel: C[i][j] = a[i] * b[j]
// C is n x n, a is n elements, b is n elements

template <typename T, size_t N>
using my_vec = __attribute__((__vector_size__(N * sizeof(T)))) T;

template <typename T, size_t N>
__global__ void __launch_bounds__(max_block_size)
outer_product_kernel(my_vec<T, N> *__restrict__ C,
                     const my_vec<T, N> *__restrict__ a,
                     const my_vec<T, N> *__restrict__ b,
                     size_t size,
                     size_t n_vec,
                     size_t n_passes_x,
                     size_t n_passes_y,
                     size_t stride_x,
                     size_t stride_y) {
  int tid_x = threadIdx.x + blockIdx.x * blockDim.x;
  int tid_y = threadIdx.y + blockIdx.y * blockDim.y;

  extern __shared__ my_vec<T, N> shared_mem[];
  my_vec<T, N> *b_shared = shared_mem;
  my_vec<T, N> *a_shared = b_shared + blockDim.y * n_passes_y;

  // Load B tile into shared memory (b has n_vec elements when viewed as vector array)
  const size_t local_y = threadIdx.y;
  const size_t shared_stride_y = blockDim.y;

  for (int j = 0; j < n_passes_y; ++j) {
    const size_t global_y = tid_y + j * stride_y;
    if (global_y < n_vec) {
      b_shared[local_y + j * shared_stride_y] = b[global_y];
    }
  }

  for (int i = 0; i < n_passes_x; ++i) {
    const size_t block_x_vec = tid_x + i * stride_x;
    const size_t block_x = block_x_vec * N;
    if (threadIdx.y == 0) {
      my_vec<T, N> zero{};
      a_shared[threadIdx.x] = (block_x_vec < n_vec) ? a[block_x_vec] : zero;
    }
    __syncthreads();
    const my_vec<T, N> x_val = a_shared[threadIdx.x];

    for (int j = 0; j < n_passes_y; ++j) {
      const size_t block_y = tid_y + j * stride_y;
      if (block_x < size && block_y < n_vec) {
        const my_vec<T, N> y_val = b_shared[local_y + j * shared_stride_y];

#pragma unroll
        for (size_t p = 0; p < N; ++p) {
          // Row (block_x+p), columns 2*block_y and 2*block_y+1 -> vector index (block_x+p)*n_vec + block_y
          C[(block_x + p) * n_vec + block_y] = x_val[p] * y_val;
        }
      }
    }
    __syncthreads();
  }
}

template <typename T>
class run_iteration {
public:
  run_iteration(T *C_, T *a_, T *b_, int n_, hipStream_t stream_)
      : C(C_), a(a_), b(b_), n(n_), stream(stream_) {};

  void operator()() {
    for (int i = 0; i < iterations_before_sync; ++i) {
      constexpr size_t vec_size = 4;
      const size_t n_vec = n / vec_size;

      dim3 block(16, 16);
      size_t grid_dim = std::min(n_compute_units, ceil_div(static_cast<int>(n_vec), static_cast<int>(block.x)));
      dim3 grid(grid_dim, grid_dim);

      size_t total_n_threads_x = block.x * grid.x;
      size_t total_n_threads_y = block.y * grid.y;
      size_t n_passes_x = ceil_div(static_cast<int>(n_vec), static_cast<int>(total_n_threads_x));
      size_t n_passes_y = ceil_div(static_cast<int>(n_vec), static_cast<int>(total_n_threads_y));

      // Strides equal the total number of threads in each dimension
      size_t stride_x = total_n_threads_x;
      size_t stride_y = total_n_threads_y;

      const size_t shared_size =
          sizeof(my_vec<T, vec_size>) * (block.y * n_passes_y + block.x);

      outer_product_kernel<T, vec_size><<<grid, block, shared_size, stream>>>(
          reinterpret_cast<my_vec<T, vec_size> *>(C),
          reinterpret_cast<const my_vec<T, vec_size> *>(a),
          reinterpret_cast<const my_vec<T, vec_size> *>(b),
          static_cast<size_t>(n),
          n_vec,
          n_passes_x,
          n_passes_y,
          stride_x,
          stride_y);
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
    b_host[i] = from_float<T>(static_cast<float>((i % 16) + 1));
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
  std::cout << "Outer Product Kernel Benchmark (shared+vec, " << type_name << ")"
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

