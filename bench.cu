#include <cstdio>
#include <cstdlib>
#include <ctime>

#include <cuda_runtime.h>

#include "ampere/optimized.cuh"
#include "cuda_utils.cuh"
#include "utils.cuh"

#define TEST_DIM_M 4096
#define TEST_DIM_K 4096
#define TEST_DIM_N 4096
#define WARMUP_ITERS 50
#define BENCH_ITERS 500

const char *kernels[] = {"0. cublas", "1. optimized"};

void run_kernel(int num, const float *A, const float *B, float *C);

int main(int argc, char **argv) {
  constexpr int kernel_choices = sizeof(kernels) / sizeof(const char *);
  int kernel_num = -1;
  if (argc == 2)
    kernel_num = atoi(argv[1]);
  if (argc != 2 || kernel_num < 0 || kernel_num >= kernel_choices) {
    printf("Please enter a number selecting one of the following kernels:\n");
    for (int i = 0; i < kernel_choices; i++)
      printf("\t%s\n", kernels[i]);
    return EXIT_FAILURE;
  }

  int deviceIdx = 0;
  cudaCheck(cudaSetDevice(deviceIdx));

  cublasCheck(cublasCreate(&cublas_handle));
  bool use_tensor = true;
  cublasMath_t cublas_math_mode =
      use_tensor ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH;
  cublasCheck(cublasSetMathMode(cublas_handle, cublas_math_mode));

  srand(time(NULL));

  float *h_A = (float *)malloc(TEST_DIM_M * TEST_DIM_K * sizeof(float));
  float *h_B = (float *)malloc(TEST_DIM_K * TEST_DIM_N * sizeof(float));
  float *h_C = (float *)malloc(TEST_DIM_M * TEST_DIM_N * sizeof(float));
  float *h_C_ref = (float *)malloc(TEST_DIM_M * TEST_DIM_N * sizeof(float));
  init_random_matrix(h_A, TEST_DIM_M * TEST_DIM_K);
  init_random_matrix(h_B, TEST_DIM_K * TEST_DIM_N);

  float *d_A;
  float *d_B;
  float *d_C;
  float *d_C_ref;
  cudaCheck(cudaMalloc(&d_A, TEST_DIM_M * TEST_DIM_K * sizeof(float)));
  cudaCheck(cudaMalloc(&d_B, TEST_DIM_K * TEST_DIM_N * sizeof(float)));
  cudaCheck(cudaMalloc(&d_C, TEST_DIM_M * TEST_DIM_N * sizeof(float)));
  cudaCheck(cudaMalloc(&d_C_ref, TEST_DIM_M * TEST_DIM_N * sizeof(float)));

  cudaCheck(cudaMemcpy(d_A, h_A, TEST_DIM_M * TEST_DIM_K * sizeof(float),
                       cudaMemcpyHostToDevice));
  cudaCheck(cudaMemcpy(d_B, h_B, TEST_DIM_K * TEST_DIM_N * sizeof(float),
                       cudaMemcpyHostToDevice));

  // A @ B.T
  run_kernel(0, d_A, d_B, d_C_ref);
  run_kernel(kernel_num, d_A, d_B, d_C);
  cudaCheck(cudaMemcpy(h_C_ref, d_C_ref,
                       TEST_DIM_M * TEST_DIM_N * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaMemcpy(h_C, d_C, TEST_DIM_M * TEST_DIM_N * sizeof(float),
                       cudaMemcpyDeviceToHost));
  cudaCheck(cudaDeviceSynchronize());
  if (!verify_matrix(h_C_ref, h_C, TEST_DIM_M * TEST_DIM_N)) {
    printf(
        "Kernel output did not match with cuBLAS reference implementation.\n");
    return EXIT_FAILURE;
  }

  for (int i = 0; i < WARMUP_ITERS; i++)
    run_kernel(kernel_num, d_A, d_B, d_C);
  cudaCheck(cudaDeviceSynchronize());

  cudaEvent_t start, end;
  cudaCheck(cudaEventCreate(&start));
  cudaCheck(cudaEventCreate(&end));
  float time;
  cudaCheck(cudaEventRecord(start));
  for (int i = 0; i < BENCH_ITERS; i++)
    run_kernel(kernel_num, d_A, d_B, d_C);
  cudaCheck(cudaEventRecord(end));
  cudaCheck(cudaEventSynchronize(start));
  cudaCheck(cudaEventSynchronize(end));
  cudaCheck(cudaEventElapsedTime(&time, start, end));

  printf("Average time of %7.6f ms with %7.3f GFLOPS throughput.",
         time / BENCH_ITERS,
         (BENCH_ITERS * 2ll * TEST_DIM_M * TEST_DIM_K * TEST_DIM_N * 1e-9) /
             (time * 0.001));

  cudaCheck(cudaFree(d_A));
  cudaCheck(cudaFree(d_B));
  cudaCheck(cudaFree(d_C));
  cudaCheck(cudaFree(d_C_ref));

  free(h_A);
  free(h_B);
  free(h_C);
  free(h_C_ref);

  return EXIT_SUCCESS;
}

void run_kernel(int num, const float *A, const float *B, float *C) {
  constexpr static float alpha = 1.0f;
  constexpr static float beta = 0.0f;
  switch (num) {
  case 0:
    // C = A @ B.T
    cublasSgemm(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, TEST_DIM_N, TEST_DIM_M,
                TEST_DIM_K, &alpha, B, TEST_DIM_K, A, TEST_DIM_K, &beta, C,
                TEST_DIM_N);
    break;
  case 1:
    matmul_optimized(A, B, C, TEST_DIM_M, TEST_DIM_K, TEST_DIM_N);
    break;
  }
}
