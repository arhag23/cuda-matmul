#ifndef __UTILS_CUH__
#define __UTILS_CUH__

#include <cstdlib>

#include <cuda_runtime.h>

void init_random_matrix(float *mat, int N) {
  for (int i = 0; i < N; i++) {
    float r = (float)rand() / RAND_MAX;
    // mat[i] = (r * 2.0f) - 1.0f;
    mat[i] = r;
  }
}

bool verify_matrix(float *ref, float *inp, int N) {
  constexpr float REL_EPS = 5e-3f;
  constexpr float ABS_EPS = 1e-1f;

  for (int i = 0; i < N; i++) {
    float abs_ref = fabs(ref[i]);
    float abs_diff = fabs(ref[i] - inp[i]);
    if (abs_diff > ABS_EPS && abs_diff / (abs_ref + 1e-6f) > REL_EPS)
      return false;
  }

  return true;
}

#endif
