// SPDX-License-Identifier: Apache-2.0
// Generate a private quaternion adjoint for the native activation contract.
#include "clad/Differentiator/Differentiator.h"
#include <cmath>
#include <cuda_runtime.h>

namespace quaternion_activation {
__device__ float clamped_loss(const float* q, float s0, float s1, float s2,
                              float s3) {
  return q[0] / 1e-12f * s0 + q[1] / 1e-12f * s1 + q[2] / 1e-12f * s2 +
         q[3] / 1e-12f * s3;
}

__device__ float loss(const float* q, float s0, float s1, float s2, float s3) {
  float squared = q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3];
  float norm = std::sqrt(squared);
  return q[0] / norm * s0 + q[1] / norm * s1 + q[2] / norm * s2 +
         q[3] / norm * s3;
}

__device__ inline void gradient(const float* q, const float* seed, float norm,
                                float* result) {
  // Reuse the forward norm so rounding cannot select a different clamp piece.
  // Keep sqrt's singular zero-norm derivative outside the clamped piece.
  if (norm < 1e-12f) {
    auto derivative = clad::gradient(clamped_loss, "q");
    derivative.execute(q, seed[0], seed[1], seed[2], seed[3], result);
  } else {
    auto derivative = clad::gradient(loss, "q");
    derivative.execute(q, seed[0], seed[1], seed[2], seed[3], result);
  }
}

__global__ void backward(int n, const float* q, const float* seed,
                         const float* norms, float* out) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float result[4] = {};
  gradient(q + 4 * i, seed + 4 * i, norms[i], result);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    out[4 * i + j] = result[j];
}

// Local perturbations reuse native forward values, including rounded
// saturation.
__device__ float scale_loss(float delta, float activated, float seed) {
  return activated * std::exp(delta) * seed;
}

__device__ float opacity_loss(float delta, float activated, float seed) {
  return activated / (activated + (1 - activated) * std::exp(-delta)) * seed;
}

struct AdamGroup {
  float* parameter;
  float* first;
  float* second;
  double rate;
  float bias1;
  float bias2;
};

struct ActivationStep {
  AdamGroup groups[3];
  const float* seeds[3];
  const float* saved[3];
};

__device__ inline void update(AdamGroup state, int k, float g) {
  const double beta1 = .9, beta2 = .999, eps = 1e-15;
  const float step_size = state.rate / state.bias1;
  state.first[k] = beta1 * state.first[k] + (1 - beta1) * g;
  state.second[k] = beta2 * state.second[k] + (1 - beta2) * g * g;
  const float denominator = sqrtf(state.second[k]) / state.bias2 + eps;
  state.parameter[k] -= step_size * state.first[k] / denominator;
}

__global__ void consume_activations(int n, ActivationStep step) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n)
    return;
  float rotation[4] = {};
  gradient(step.groups[0].parameter + 4 * i, step.seeds[0] + 4 * i,
           step.saved[0][i], rotation);
#pragma unroll
  for (int j = 0; j < 4; ++j)
    update(step.groups[0], 4 * i + j, rotation[j]);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    float result = 0;
    auto derivative = clad::gradient(scale_loss, "delta");
    derivative.execute(0, step.saved[1][3 * i + j], step.seeds[1][3 * i + j],
                       &result);
    update(step.groups[1], 3 * i + j, result);
  }
  float result = 0;
  auto derivative = clad::gradient(opacity_loss, "delta");
  derivative.execute(0, step.saved[2][i], step.seeds[2][i], &result);
  update(step.groups[2], i, result);
}
} // namespace quaternion_activation

extern "C" void
launch_activation_consume(int n, float* const* parameters,
                          const float* const* seeds, const float* const* saved,
                          float* const* first, float* const* second,
                          const double* rates, const float* bias1,
                          const float* bias2, cudaStream_t stream) {
  quaternion_activation::ActivationStep step;
  for (int j = 0; j < 3; ++j) {
    step.groups[j] = {parameters[j], first[j], second[j],
                      rates[j],      bias1[j], bias2[j]};
    step.seeds[j] = seeds[j];
    step.saved[j] = saved[j];
  }
  if (n)
    quaternion_activation::
        consume_activations<<<(n + 255) / 256, 256, 0, stream>>>(n, step);
}

extern "C" void launch_quaternion_backward(int n, const float* q,
                                           const float* seed,
                                           const float* norms, float* out,
                                           cudaStream_t stream) {
  if (n)
    quaternion_activation::backward<<<(n + 255) / 256, 256, 0, stream>>>(
        n, q, seed, norms, out);
}
