#ifndef __MATMUL_KERNEL_CUH__
#define __MATMUL_KERNEL_CUH__

#include <cuda_runtime.h>

#include "cuda_utils.cuh"

#define FRAG_MN 16
#define FRAG_K 8
#define WARP_SIZE 32
#define WARP_MMA_ACC_REGS 8
#define WARP_MMA_FRAG_REGS 4

#define MMF_NUM_BUFFERS 3 // minimum of 2
#define REG_BUFFERS 2

template <int WARP_COARSEN_X_, int WARP_COARSEN_Y_, int WARP_COARSEN_K_,
          int BLOCK_NWARPS_X_, int BLOCK_NWARPS_Y_, int MIN_OCCUPANCY_>
struct GemmConfig {
  static constexpr int WARP_COARSEN_X = WARP_COARSEN_X_;
  static constexpr int WARP_COARSEN_Y = WARP_COARSEN_Y_;
  static constexpr int WARP_COARSEN_K = WARP_COARSEN_K_;
  static constexpr int BLOCK_NWARPS_X = BLOCK_NWARPS_X_;
  static constexpr int BLOCK_NWARPS_Y = BLOCK_NWARPS_Y_;
  static constexpr int MIN_OCCUPANCY = MIN_OCCUPANCY_;

  static constexpr int WARP_WIDTH = (FRAG_MN * WARP_COARSEN_X);
  static constexpr int WARP_HEIGHT = (FRAG_MN * WARP_COARSEN_Y);
  static constexpr int TILE_WIDTH = (FRAG_K * WARP_COARSEN_K);
  static constexpr int SWIZZLE_WIDTH = (TILE_WIDTH / 4);
  static constexpr int SWIZZLE_ROWS = 8 / SWIZZLE_WIDTH;
  static constexpr int BLOCK_WIDTH = (WARP_WIDTH * BLOCK_NWARPS_X);
  static constexpr int BLOCK_HEIGHT = (WARP_HEIGHT * BLOCK_NWARPS_Y);
  static constexpr int NUM_THREADS =
      (WARP_SIZE * BLOCK_NWARPS_X * BLOCK_NWARPS_Y);
  static constexpr int BLOCK_TILE_WIDTH = 512 / BLOCK_WIDTH;
};

template <typename Config, bool A_TILE> struct GlobalToSharedLoader {
  static constexpr int TILE_HEIGHT =
      A_TILE ? Config::BLOCK_HEIGHT : Config::BLOCK_WIDTH;
  static constexpr int LOAD_STEPS =
      CEIL_DIV(TILE_HEIGHT * Config::TILE_WIDTH, Config::NUM_THREADS * 4);
  const float *input;
  const int stride;
  const int max_row;
  const int tx;

  __device__ __forceinline__ GlobalToSharedLoader(const float *gmem,
                                                  const int stride,
                                                  const int max_row,
                                                  const int tx)
      : input(gmem + (tx / Config::SWIZZLE_WIDTH) * stride +
              (tx % Config::SWIZZLE_WIDTH) * 4),
        stride((Config::NUM_THREADS / Config::SWIZZLE_WIDTH) * stride),
        max_row(max_row), tx(tx) {}

  __device__ __forceinline__ void load(uint32_t buf_base) {
    const float *load = input;
    uint32_t store = buf_base + (tx / 8 * 32) * 4;
    int c = tx % 8;
    int r = tx / 8;

    int r_v = tx / Config::SWIZZLE_WIDTH;
#pragma unroll
    for (int l = 0; l < LOAD_STEPS; l++) {
      int c_swiz = (c ^ (r % Config::SWIZZLE_WIDTH)) * 4;
      uint32_t store_iter = store + c_swiz * 4;

      int valid = (r_v) < max_row ? 16 : 0;

      asm("cp.async.cg.shared.global.L2::128B [%0], [%1], 16, %2;" ::"r"(
              store_iter),
          "l"(load), "r"(valid));

      load += stride;
      store += Config::NUM_THREADS * 4 * 4;
      r += Config::NUM_THREADS / 8;
      r_v += Config::NUM_THREADS / Config::SWIZZLE_WIDTH;
    }
    input = input + Config::TILE_WIDTH;
  }
};

template <typename Config> struct SwizzleTable {
  int offsets[Config::WARP_COARSEN_K][WARP_SIZE];

  __device__ constexpr SwizzleTable() : offsets{} {
    for (int k = 0; k < Config::WARP_COARSEN_K; k++)
      for (int i = 0; i < WARP_SIZE; i++) {
        int r_frag = i % FRAG_MN;
        int r_swiz = (r_frag / Config::SWIZZLE_ROWS) % Config::SWIZZLE_WIDTH;
        int c = (((r_frag % Config::SWIZZLE_ROWS) * Config::SWIZZLE_WIDTH +
                  k * (FRAG_K / 4) + i / FRAG_MN) ^
                 (r_swiz)) *
                4;
        offsets[k][i] = c * 4;
      }
  }
};

template <typename Config>
__device__ __constant__ const SwizzleTable<Config> SWIZ_TBL{};

template <typename Config, bool A_TILE> struct RegisterLoader {
  static constexpr int WARP_COARSEN =
      A_TILE ? Config::WARP_COARSEN_Y : Config::WARP_COARSEN_X;
  static constexpr int WARP_LENGTH =
      A_TILE ? Config::WARP_HEIGHT : Config::WARP_WIDTH;
  static constexpr int TILE_HEIGHT =
      A_TILE ? Config::BLOCK_HEIGHT : Config::BLOCK_WIDTH;
  const uint32_t base_offset;
  uint32_t frag[REG_BUFFERS][WARP_COARSEN][WARP_MMA_FRAG_REGS];
  const int lane_idx;

  __device__ __forceinline__ RegisterLoader(const int warp_idx,
                                            const int lane_idx)
      : base_offset((warp_idx * WARP_LENGTH * Config::TILE_WIDTH +
                     (lane_idx % FRAG_MN) / Config::SWIZZLE_ROWS * 32) *
                    4),
        lane_idx(lane_idx) {}

  __device__ __forceinline__ void load(const uint32_t buf_base, const int k) {
    uint32_t load =
        buf_base + base_offset + SWIZ_TBL<Config>.offsets[k][lane_idx];
#pragma unroll
    for (int i = 0; i < WARP_COARSEN; i++) {
      asm("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
          : "=r"(frag[k % REG_BUFFERS][i][0]),
            "=r"(frag[k % REG_BUFFERS][i][1]),
            "=r"(frag[k % REG_BUFFERS][i][2]), "=r"(frag[k % REG_BUFFERS][i][3])
          : "r"(load));
      load += FRAG_MN * Config::TILE_WIDTH * 4;
    }
  }
};

template <typename Config> struct Matmul {
  float acc[Config::WARP_COARSEN_Y][Config::WARP_COARSEN_X][WARP_MMA_ACC_REGS];

  __device__ __forceinline__ Matmul() {

    for (unsigned int i = 0; i < Config::WARP_COARSEN_Y; i++)
      for (unsigned int j = 0; j < Config::WARP_COARSEN_X; j++)
        for (unsigned int k = 0; k < WARP_MMA_ACC_REGS; k++)
          acc[i][j][k] = 0.0;
  }
  __device__ __forceinline__ void compute(
      const uint32_t (&a_frag)[Config::WARP_COARSEN_Y][WARP_MMA_FRAG_REGS],
      const uint32_t (&b_frag)[Config::WARP_COARSEN_X][WARP_MMA_FRAG_REGS]) {
#pragma unroll
    for (int j = 0; j < Config::WARP_COARSEN_X; j++)
#pragma unroll
      for (int i = 0; i < Config::WARP_COARSEN_Y; i++) {
        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(acc[i][j][0]), "+f"(acc[i][j][1]), "+f"(acc[i][j][2]),
              "+f"(acc[i][j][3])
            : "r"(a_frag[i][0]), "r"(a_frag[i][1]), "r"(a_frag[i][2]),
              "r"(a_frag[i][3]), "r"(b_frag[j][0]), "r"(b_frag[j][2]));

        asm("mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
            "{%0, %1, %2, %3}, "
            "{%4, %5, %6, %7}, "
            "{%8, %9}, "
            "{%0, %1, %2, %3};"
            : "+f"(acc[i][j][4]), "+f"(acc[i][j][5]), "+f"(acc[i][j][6]),
              "+f"(acc[i][j][7])
            : "r"(a_frag[i][0]), "r"(a_frag[i][1]), "r"(a_frag[i][2]),
              "r"(a_frag[i][3]), "r"(b_frag[j][1]), "r"(b_frag[j][3]));
      }
  }
};

template <typename Config>
__global__ __launch_bounds__(
    Config::NUM_THREADS,
    Config::MIN_OCCUPANCY) void matmul_optimized_kernel(const float *A,
                                                        const float *B,
                                                        float *C, int M, int K,
                                                        int N) {
  const int tx = threadIdx.x;
  __builtin_assume(tx >= 0 && tx < Config::NUM_THREADS);
  // thread block rasterization
  const int grid_height = (M + Config::BLOCK_HEIGHT - 1) / Config::BLOCK_HEIGHT;
  const unsigned int grid_width =
      ((N + Config::BLOCK_WIDTH - 1) / Config::BLOCK_WIDTH);
  const int super_col_size = Config::BLOCK_TILE_WIDTH * grid_height;
  const int super_col = blockIdx.x / super_col_size;
  const int local_idx = blockIdx.x % super_col_size;
  const int super_col_width =
      ((super_col + 1) * Config::BLOCK_TILE_WIDTH - 1 < grid_width)
          ? Config::BLOCK_TILE_WIDTH
          : grid_width % Config::BLOCK_TILE_WIDTH;
  const int blk_idx_y = local_idx / super_col_width;
  const int blk_idx_x =
      super_col * Config::BLOCK_TILE_WIDTH + local_idx % super_col_width;

  // tiling indices
  const int warpIdx_x = (tx / WARP_SIZE) % Config::BLOCK_NWARPS_X;
  const int warpIdx_y = (tx / WARP_SIZE) / Config::BLOCK_NWARPS_X;
  const int row_block = blk_idx_y * Config::BLOCK_HEIGHT;
  const int col_block = blk_idx_x * Config::BLOCK_WIDTH;
  const int row = row_block + warpIdx_y * Config::WARP_HEIGHT;
  const int col = col_block + warpIdx_x * Config::WARP_WIDTH;
  const int lane_idx = tx % WARP_SIZE;

  extern __shared__ __align__(128) float smem_block[];
  float (*a_tile)[Config::BLOCK_HEIGHT * Config::TILE_WIDTH] =
      (float (*)[Config::BLOCK_HEIGHT * Config::TILE_WIDTH]) smem_block;
  float (*b_tile)[Config::BLOCK_WIDTH * Config::TILE_WIDTH] =
      (float (*)[Config::BLOCK_WIDTH * Config::TILE_WIDTH])(
          smem_block +
          MMF_NUM_BUFFERS * Config::BLOCK_HEIGHT * Config::TILE_WIDTH);

  const uint32_t a_addr_base = __cvta_generic_to_shared(a_tile);
  const uint32_t b_addr_base = __cvta_generic_to_shared(b_tile);

  const uint32_t a_stride = Config::BLOCK_HEIGHT * Config::TILE_WIDTH * 4;
  const uint32_t b_stride = Config::BLOCK_WIDTH * Config::TILE_WIDTH * 4;

  // const uint32_t a_max_offset = a_addr_base + MMF_NUM_BUFFERS * a_stride;

  // uint32_t a_addr_fetch = a_addr_base;
  // uint32_t b_addr_fetch = b_addr_base;
  //
  // uint32_t a_addr_compute = a_addr_base;
  // uint32_t b_addr_compute = b_addr_base;

  GlobalToSharedLoader<Config, true> a_loader(A + row_block * K, K,
                                              M - row_block, tx);
  GlobalToSharedLoader<Config, false> b_loader(B + col_block * K, K,
                                               N - col_block, tx);

  Matmul<Config> matmul;
  RegisterLoader<Config, true> a_reg_loader(warpIdx_y, lane_idx);
  RegisterLoader<Config, false> b_reg_loader(warpIdx_x, lane_idx);

  int compute_stage = 0;
  int fetch_stage = 0;
  for (; fetch_stage < MMF_NUM_BUFFERS - 1; fetch_stage++) {
    // a_loader.load(a_addr_fetch);
    // b_loader.load(b_addr_fetch);
    a_loader.load(a_addr_base + fetch_stage * a_stride);
    b_loader.load(b_addr_base + fetch_stage * b_stride);
    asm volatile("cp.async.commit_group;");
    // a_addr_fetch += a_stride;
    // b_addr_fetch += b_stride;
  }

  const int tiles = CEIL_DIV(K, Config::TILE_WIDTH);

  asm volatile("cp.async.wait_group %0;" ::"n"(MMF_NUM_BUFFERS - 2));
  __syncthreads();
  // a_reg_loader.load(a_addr_compute, 0);
  // b_reg_loader.load(b_addr_compute, 0);
  a_reg_loader.load(a_addr_base, 0);
  b_reg_loader.load(b_addr_base, 0);

  for (int t = 0; t < tiles - MMF_NUM_BUFFERS + 1; t++) {
    // a_loader.load(a_addr_fetch);
    // b_loader.load(b_addr_fetch);
    a_loader.load(a_addr_base + fetch_stage * a_stride);
    b_loader.load(b_addr_base + fetch_stage * b_stride);

    // a_addr_fetch += a_stride;
    // b_addr_fetch += b_stride;

    // if (a_addr_fetch == a_max_offset) {
    //   a_addr_fetch = a_addr_base;
    //   b_addr_fetch = b_addr_base;
    // }
    fetch_stage = (fetch_stage == MMF_NUM_BUFFERS - 1) ? 0 : fetch_stage + 1;

#pragma unroll
    for (int k = 0; k < Config::WARP_COARSEN_K - 1; k++) {
      // a_reg_loader.load(a_addr_compute, k + 1);
      // b_reg_loader.load(b_addr_compute, k + 1);
      a_reg_loader.load(a_addr_base + compute_stage * a_stride, k + 1);
      b_reg_loader.load(b_addr_base + compute_stage * b_stride, k + 1);
      matmul.compute(a_reg_loader.frag[k % REG_BUFFERS],
                     b_reg_loader.frag[k % REG_BUFFERS]);
    }
    matmul.compute(
        a_reg_loader.frag[(Config::WARP_COARSEN_K - 1) % REG_BUFFERS],
        b_reg_loader.frag[(Config::WARP_COARSEN_K - 1) % REG_BUFFERS]);

    // fetch_stage += 1;
    asm volatile("cp.async.commit_group;");
    asm volatile("cp.async.wait_group %0;" ::"n"(MMF_NUM_BUFFERS - 2));
    __syncthreads();

    // a_addr_compute += a_stride;
    // b_addr_compute += b_stride;
    //
    // if (a_addr_compute == a_max_offset) {
    //   a_addr_compute = a_addr_base;
    //   b_addr_compute = b_addr_base;
    // }

    compute_stage =
        (compute_stage == MMF_NUM_BUFFERS - 1) ? 0 : compute_stage + 1;

    // a_reg_loader.load(a_addr_compute, 0);
    // b_reg_loader.load(b_addr_compute, 0);
    a_reg_loader.load(a_addr_base + compute_stage * a_stride, 0);
    b_reg_loader.load(b_addr_base + compute_stage * b_stride, 0);
  }

#pragma unroll
  for (int t = tiles - MMF_NUM_BUFFERS + 1; t < tiles; t++) {
#pragma unroll
    for (int k = 0; k < Config::WARP_COARSEN_K - 1; k++) {
      // a_reg_loader.load(a_addr_compute, k + 1);
      // b_reg_loader.load(b_addr_compute, k + 1);
      a_reg_loader.load(a_addr_base + compute_stage * a_stride, k + 1);
      b_reg_loader.load(b_addr_base + compute_stage * b_stride, k + 1);
      matmul.compute(a_reg_loader.frag[k % REG_BUFFERS],
                     b_reg_loader.frag[k % REG_BUFFERS]);
    }
    matmul.compute(
        a_reg_loader.frag[(Config::WARP_COARSEN_K - 1) % REG_BUFFERS],
        b_reg_loader.frag[(Config::WARP_COARSEN_K - 1) % REG_BUFFERS]);
    if (tiles - t > 1)
      asm volatile("cp.async.wait_group %0;" ::"n"(0));
    __syncthreads();
    if (tiles - t > 1) {
      // a_addr_compute += a_stride;
      // b_addr_compute += b_stride;
      //
      // if (a_addr_compute == a_max_offset) {
      //   a_addr_compute = a_addr_base;
      //   b_addr_compute = b_addr_base;
      // }
      // a_reg_loader.load(a_addr_compute, 0);
      // b_reg_loader.load(b_addr_compute, 0);
      compute_stage =
          (compute_stage == MMF_NUM_BUFFERS - 1) ? 0 : compute_stage + 1;
      a_reg_loader.load(a_addr_base + compute_stage * a_stride, 0);
      b_reg_loader.load(b_addr_base + compute_stage * b_stride, 0);
    }
  }

  const unsigned int lane_row = lane_idx / 4;
  const unsigned int lane_col = (lane_idx % 4) * 2;
  float *c_tile =
      smem_block +
      (warpIdx_y * Config::BLOCK_NWARPS_X + warpIdx_x) * FRAG_MN * FRAG_MN;
  float4 *C4 = reinterpret_cast<float4 *>(C);
  float4 *c_tile4 = reinterpret_cast<float4 *>(c_tile);

#pragma unroll
  for (int i = 0; i < Config::WARP_COARSEN_Y; i++) {
#pragma unroll
    for (int j = 0; j < Config::WARP_COARSEN_X; j++) {
#pragma unroll
      for (int r = 0; r < WARP_MMA_ACC_REGS; r++) {
        const unsigned int row_off = (r / 2) % 2 * 8 + lane_row;
        const unsigned int col_off = r % 2 + r / 4 * 8 + lane_col;
        c_tile[row_off * FRAG_MN + col_off] = matmul.acc[i][j][r];
      }
      __syncwarp();
      const unsigned int row_iter = i * FRAG_MN;
      const unsigned int col_iter = j * FRAG_MN / 4;
      for (unsigned int k = lane_idx; k < FRAG_MN * FRAG_MN / 4;
           k += WARP_SIZE) {
        const unsigned int row_off = k / (FRAG_MN / 4);
        const unsigned int col_off = k % (FRAG_MN / 4);
        if (row + row_iter + row_off < M &&
            col / 4 + col_iter + col_off < N / 4) {
          float4 val = c_tile4[row_off * FRAG_MN / 4 + col_off];
          C4[(row + row_iter + row_off) * (N / 4) + col / 4 + col_iter +
             col_off] = val;
        }
      }
      __syncwarp();
    }
  }
}

void matmul_optimized(const float *A, const float *B, float *C, int M, int K,
                      int N) {
  // Implement this
  auto launch_kernel = [&](auto config_instance) {
    using Config = decltype(config_instance);

    dim3 blockDim(Config::NUM_THREADS);
    dim3 gridDim(CEIL_DIV(N, Config::BLOCK_WIDTH) *
                 CEIL_DIV(M, Config::BLOCK_HEIGHT));

    cudaFuncSetAttribute(matmul_optimized_kernel<Config>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 98304);

    const unsigned int smem_size =
        max(MMF_NUM_BUFFERS * Config::TILE_WIDTH *
                (Config::BLOCK_HEIGHT + Config::BLOCK_WIDTH),
            Config::BLOCK_NWARPS_Y * Config::BLOCK_NWARPS_X * FRAG_MN *
                FRAG_MN) *
        sizeof(float);

    matmul_optimized_kernel<Config>
        <<<gridDim, blockDim, smem_size>>>(A, B, C, M, K, N);
  };

  launch_kernel(GemmConfig<4, 2, 2, 2, 4, 2>{});
}

#endif // __MATMUL_KERNEL_CUH__
