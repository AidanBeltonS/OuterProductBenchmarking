#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <rocblas/rocblas.h>

constexpr size_t iterations_before_sync = 1;

#define HIP_CHECK(fn)                                                          \
  {                                                                            \
    hipError_t res = fn;                                                       \
    if (res != hipSuccess) {                                                   \
      std::cerr << "HIP error code:" << hipGetErrorName(res) << std::endl;    \
      assert(false);                                                           \
    }                                                                          \
  }

#define ROCBLAS_CHECK(fn)                                                      \
  {                                                                            \
    rocblas_status res = fn;                                                   \
    if (res != rocblas_status_success) {                                       \
      std::cerr << "rocBLAS error code: " << res << std::endl;                 \
      assert(false);                                                           \
    }                                                                          \
  }

int ceil_div(int x, int y) { return (x + y - 1) / y; }

// Calculate throughput in GiB/s from bytes transferred and time in nanoseconds
double calculate_throughput_gibps(size_t bytes, size_t time_ns) {
  return (static_cast<double>(bytes) * 1e9) / (time_ns * 1024.0 * 1024.0 * 1024.0);
}

// Calculate TFLOPS given total floating point operations and time in nanoseconds
double calculate_tflops(size_t flops, size_t time_ns) {
  // FLOP/s = flops * 1e9 / time_ns, TFLOP/s = FLOP/s / 1e12
  // => TFLOP/s = flops / (time_ns * 1e3)
  return static_cast<double>(flops) / (static_cast<double>(time_ns) * 1e3);
}

// Helper to convert type T to float for verification
template <typename T>
inline float to_float(T val) { return static_cast<float>(val); }

template <>
inline float to_float(half val) { return __half2float(val); }

// Helper to convert float to type T for initialization
template <typename T>
inline T from_float(float val) { return static_cast<T>(val); }

template <>
inline half from_float(float val) { return __float2half(val); }

// Verify outer product result: C[i][j] should equal a[i] * b[j]
// rocBLAS uses column-major: C(i,j) is stored at index i + j*lda = i + j*n
template <typename T>
bool verify_outer_product_colmajor(const T *C_host, const T *a_host, const T *b_host, int n) {
  constexpr float tolerance = 1e-3f;
  int errors = 0;
  const int max_errors_to_print = 10;
  
  for (int i = 0; i < n && errors < max_errors_to_print; ++i) {
    for (int j = 0; j < n && errors < max_errors_to_print; ++j) {
      float expected = to_float(a_host[i]) * to_float(b_host[j]);
      // Column-major layout: (i,j) at index i + j*n
      float actual = to_float(C_host[i + j * n]);
      float diff = std::fabs(expected - actual);
      float rel_err = (expected != 0.0f) ? diff / std::fabs(expected) : diff;
      
      if (rel_err > tolerance && diff > tolerance) {
        if (errors < max_errors_to_print) {
          std::cerr << "  Mismatch at C[" << i << "][" << j << "]: "
                    << "expected " << expected << ", got " << actual 
                    << " (diff=" << diff << ", rel_err=" << rel_err << ")" << std::endl;
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

// rocBLAS gemm: C = alpha*A*B + beta*C (column-major)
// Outer product C = a*b^T: A = a (n x 1), B = b^T (1 x n), so C = A*B with m=n, n=n, k=1.
// A (n x 1) stored col-major lda=n, B (1 x n) stored col-major ldb=1.

template <typename T>
struct GemmTraits;

template <>
struct GemmTraits<float> {
  static rocblas_status call(rocblas_handle handle,
                             rocblas_int m, rocblas_int n, rocblas_int k,
                             const float *alpha, const float *A, rocblas_int lda,
                             const float *B, rocblas_int ldb, const float *beta,
                             float *C, rocblas_int ldc) {
    return rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                         m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

template <>
struct GemmTraits<double> {
  static rocblas_status call(rocblas_handle handle,
                             rocblas_int m, rocblas_int n, rocblas_int k,
                             const double *alpha, const double *A, rocblas_int lda,
                             const double *B, rocblas_int ldb, const double *beta,
                             double *C, rocblas_int ldc) {
    return rocblas_dgemm(handle, rocblas_operation_none, rocblas_operation_none,
                         m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
  }
};

template <typename T>
class run_iteration {
public:
  run_iteration(rocblas_handle handle_, T *C_, T *a_, T *b_, int n_, hipStream_t stream_)
      : handle(handle_), C(C_), a(a_), b(b_), n(n_), stream(stream_) {
  };

  void operator()() {
    for (int i = 0; i < iterations_before_sync; ++i) {
      T alpha = from_float<T>(1.0f);
      T beta = from_float<T>(0.0f);
      // C = alpha * a * b^T: a is (n x 1), b^T is (1 x n)
      ROCBLAS_CHECK(GemmTraits<T>::call(handle, n, n, 1, &alpha, a, n, b, 1, &beta, C, n));
    }
    HIP_CHECK(hipStreamSynchronize(stream));
  }

private:
  rocblas_handle handle;
  T *C;
  T *a;
  T *b;
  int n;
  hipStream_t stream;
};

template <typename T>
size_t run_benchmark(int n, int warmups, int iterations, bool *verification_passed) {
  HIP_CHECK(hipSetDevice(0));
  
  rocblas_handle handle;
  ROCBLAS_CHECK(rocblas_create_handle(&handle));
  
  hipStream_t stream;
  HIP_CHECK(hipStreamCreate(&stream));
  ROCBLAS_CHECK(rocblas_set_stream(handle, stream));

  // Allocate vectors a (n), b (n) and output matrix C (n x n) in column-major
  const size_t vector_bytes = sizeof(T) * n;
  const size_t matrix_bytes = sizeof(T) * n * n;
  
  T *a, *b, *C;
  T *a_host, *b_host, *C_host;

  // Allocate host memory
  a_host = (T*)malloc(vector_bytes);
  b_host = (T*)malloc(vector_bytes);
  C_host = (T*)malloc(matrix_bytes);

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

  run_iteration<T> run_functor(handle, C, a, b, n, stream);

  for (int i = 0; i < warmups; ++i) {
    run_functor();
  }

  unsigned long long total_ns = 0;

  for (int i = 0; i < iterations/iterations_before_sync; ++i) {
    
    auto start = std::chrono::high_resolution_clock::now();
    run_functor();
    size_t ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                 std::chrono::high_resolution_clock::now() - start)
                 .count();
    total_ns += ns;
  }
  HIP_CHECK(hipStreamSynchronize(stream));

  // Verify the result after all timed iterations (column-major)
  HIP_CHECK(hipMemcpy(C_host, C, matrix_bytes, hipMemcpyDeviceToHost));
  *verification_passed = verify_outer_product_colmajor(C_host, a_host, b_host, n);

  HIP_CHECK(hipFree(a));
  HIP_CHECK(hipFree(b));
  HIP_CHECK(hipFree(C));
  free(a_host);
  free(b_host);
  free(C_host);
  ROCBLAS_CHECK(rocblas_destroy_handle(handle));
  HIP_CHECK(hipStreamDestroy(stream));
  return total_ns / iterations;
}

template <typename T>
void run_benchmark_suite(const std::string& type_name, const std::vector<int>& sizes, 
                         int warmups, int iterations) {
  std::cout << "Outer Product (rocBLAS gemm) Benchmark (" << type_name << ")" << std::endl;
  std::cout << "n, Matrix Size (n x n), Time (ns), Throughput (GiB/s), TFLOPS, Verified" << std::endl;
  for (const auto &n : sizes) {
    bool verified = false;
    auto time = run_benchmark<T>(n, warmups, iterations, &verified);
    // Total bytes: read a (n), read b (n), read+write C (n*n)
    const size_t bytes = sizeof(T) * (2 * n + (size_t)n * n);
    // FLOPs: one multiply per output element (n x n)
    const size_t flops = static_cast<size_t>(n) * static_cast<size_t>(n);
    double throughput = calculate_throughput_gibps(bytes, time);
    double tflops = calculate_tflops(flops, time);
    std::cout << n << "," << static_cast<size_t>(n) * static_cast<size_t>(n) << "," << time << "," << throughput << "," << tflops
              << "," << (verified ? "PASS" : "FAIL") << std::endl;
  }
}

int main() {
  int warmups = 1000;
  int iterations = 3000;
  // n is the vector size; output matrix is n x n
  std::vector<int> sizes{64, 128, 256, 512, 1024, 2048, 4096, 8192,
                         2*8192, 4*8192, 8*8192};

  run_benchmark_suite<float>("float", sizes, warmups, iterations);

  return 0;
}
