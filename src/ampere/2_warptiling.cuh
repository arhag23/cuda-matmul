#ifndef __WARPTILING_CUH__
#define __WARPTILING_CUH__

#include <cuda_runtime.h>
#include <mma.h>

#include "cuda_utils.cuh"

#define FRAG_MN 16
#define FRAG_K 8
#define WARP_SIZE 32

#define BLOCK_NWARPS_X 2
#define BLOCK_NWARPS_Y 2

using namespace nvcuda;

__global__ void matmul_warptiling_kernel(const float *A, const float *B,
                                         float *C, int M, int K, int N) {
  const int col_block = blockIdx.x * FRAG_MN * BLOCK_NWARPS_X;
  const int row_block = blockIdx.y * FRAG_MN * BLOCK_NWARPS_Y;
  const int col_warp = threadIdx.x / WARP_SIZE % BLOCK_NWARPS_X;
  const int row_warp = threadIdx.x / WARP_SIZE / BLOCK_NWARPS_X;

  __shared__ float a_tile[FRAG_MN * FRAG_K * BLOCK_NWARPS_Y];
  __shared__ float b_tile[FRAG_MN * FRAG_K * BLOCK_NWARPS_X];

  wmma::fragment<wmma::matrix_a, FRAG_MN, FRAG_MN, FRAG_K,
                 wmma::precision::tf32, wmma::row_major>
      a_frag;
  wmma::fragment<wmma::matrix_b, FRAG_MN, FRAG_MN, FRAG_K,
                 wmma::precision::tf32, wmma::col_major>
      b_frag;
  wmma::fragment<wmma::accumulator, FRAG_MN, FRAG_MN, FRAG_K, float> c_frag;

  wmma::fill_fragment(c_frag, 0.0f);

  for (int i = 0; i < CEIL_DIV(K, FRAG_K); i++) {
    for (int load_idx = threadIdx.x;
         load_idx < FRAG_MN * FRAG_K * BLOCK_NWARPS_Y;
         load_idx += WARP_SIZE * BLOCK_NWARPS_X * BLOCK_NWARPS_Y) {
      int row_idx = row_block + load_idx / FRAG_K;
      int col_idx = i * FRAG_K + load_idx % FRAG_K;

      a_tile[load_idx] =
          (row_idx < M && col_idx < K) ? A[row_idx * K + col_idx] : 0;
    }

    for (int load_idx = threadIdx.x;
         load_idx < FRAG_MN * FRAG_K * BLOCK_NWARPS_X;
         load_idx += WARP_SIZE * BLOCK_NWARPS_X * BLOCK_NWARPS_Y) {
      int row_idx = col_block + load_idx / FRAG_K;
      int col_idx = i * FRAG_K + load_idx % FRAG_K;

      b_tile[load_idx] =
          (row_idx < N && col_idx < K) ? B[row_idx * K + col_idx] : 0;
    }

    __syncthreads();

    wmma::load_matrix_sync(a_frag, a_tile + FRAG_MN * FRAG_K * row_warp,
                           FRAG_K);
    wmma::load_matrix_sync(b_frag, b_tile + FRAG_MN * FRAG_K * col_warp,
                           FRAG_K);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    __syncthreads();
  }

  __shared__ float c_tile[FRAG_MN * FRAG_MN * BLOCK_NWARPS_X * BLOCK_NWARPS_Y];
  wmma::store_matrix_sync(
      c_tile + row_warp * FRAG_MN * FRAG_MN * BLOCK_NWARPS_X +
          col_warp * FRAG_MN,
      c_frag, FRAG_MN * BLOCK_NWARPS_X, wmma::mem_row_major);

  __syncthreads();

  for (int store_idx = threadIdx.x;
       store_idx < FRAG_MN * FRAG_MN * BLOCK_NWARPS_X * BLOCK_NWARPS_Y;
       store_idx += WARP_SIZE * BLOCK_NWARPS_X * BLOCK_NWARPS_Y) {
    int row_idx = row_block + store_idx / (FRAG_MN * BLOCK_NWARPS_X);
    int col_idx = col_block + store_idx % (FRAG_MN * BLOCK_NWARPS_X);

    if (row_idx < M && col_idx < N)
      C[row_idx * N + col_idx] = c_tile[store_idx];
  }
}

void matmul_warptiling(const float *A, const float *B, float *C, int M, int K,
                       int N) {
  dim3 blockDim(WARP_SIZE * BLOCK_NWARPS_X * BLOCK_NWARPS_Y);
  dim3 gridDim(CEIL_DIV(N, FRAG_MN * BLOCK_NWARPS_X),
               CEIL_DIV(M, FRAG_MN * BLOCK_NWARPS_Y));

  matmul_warptiling_kernel<<<gridDim, blockDim>>>(A, B, C, M, K, N);
}

#undef BLOCK_NWARPS_X
#undef BLOCK_NWARPS_Y

#endif // __WARPTILING_CUH__
