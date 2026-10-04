#ifndef __BASELINE_CUH__
#define __BASELINE_CUH__

#include <cuda_runtime.h>
#include <mma.h>

#include "cuda_utils.cuh"

#define FRAG_MN 16
#define FRAG_K 8
#define WARP_SIZE 32

using namespace nvcuda;

__global__ void matmul_baseline_kernel(const float *A, const float *B, float *C,
                                       int M, int K, int N) {
  const int col_block = blockIdx.x * FRAG_MN;
  const int row_block = blockIdx.y * FRAG_MN;

  __shared__ float a_tile[FRAG_MN * FRAG_K];
  __shared__ float b_tile[FRAG_MN * FRAG_K];

  wmma::fragment<wmma::matrix_a, FRAG_MN, FRAG_MN, FRAG_K,
                 wmma::precision::tf32, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, FRAG_MN, FRAG_MN, FRAG_K,
                 wmma::precision::tf32, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, FRAG_MN, FRAG_MN, FRAG_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int i = 0; i < CEIL_DIV(K, FRAG_K); i++) {
    for (int load_idx = threadIdx.x; load_idx < FRAG_MN * FRAG_K;
         load_idx += WARP_SIZE) {
      int row_idx_a = row_block + load_idx / FRAG_K;
      int row_idx_b = col_block + load_idx / FRAG_K;
      int col_idx = i * FRAG_K + load_idx % FRAG_K;

      a_tile[load_idx] =
          (row_idx_a < M && col_idx < K) ? A[row_idx_a * K + col_idx] : 0;
      b_tile[load_idx] =
          (row_idx_b < N && col_idx < K) ? B[row_idx_b * K + col_idx] : 0;
    }
    __syncthreads();

    wmma::load_matrix_sync(a_frag, a_tile, FRAG_K);
    wmma::load_matrix_sync(b_frag, b_tile, FRAG_K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
  }

  __shared__ float c_tile[FRAG_MN * FRAG_MN];
  wmma::store_matrix_sync(c_tile, c_frag, FRAG_MN, wmma::mem_row_major);

  for (int store_idx = threadIdx.x; store_idx < FRAG_MN * FRAG_MN;
       store_idx += WARP_SIZE) {
    int row_idx = row_block + store_idx / FRAG_MN;
    int col_idx = col_block + store_idx % FRAG_MN;

    if (row_idx < M && col_idx < N)
      C[row_idx * N + col_idx] = c_tile[store_idx];
  }
}

void matmul_baseline(const float *A, const float *B, float *C, int M, int K,
                     int N) {
  dim3 blockDim(WARP_SIZE);
  dim3 gridDim(CEIL_DIV(N, FRAG_MN), CEIL_DIV(M, FRAG_MN));

  matmul_baseline_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
}

#endif // __BASELINE_CUH__
