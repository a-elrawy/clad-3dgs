// SPDX-License-Identifier: Apache-2.0
// Per-term SH derivatives, then an Adam update on those coefficients.
#include "clad/Differentiator/Differentiator.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

struct AdamConfig {
  float inv_bias1, inv_bias2;
};

struct TorchAdamConfig {
  float bias1, bias2_sqrt;
};

namespace sh_consumer {
template <class T>
__host__ __device__ T term(const T* d, const T* c, int j, T sr, T sg, T sb) {
  T x = d[0], y = d[1], z = d[2];
  T xx = x * x, yy = y * y, zz = z * z, b = 0;
  T xy = x * y, yz = y * z, xz = x * z;
  switch (j) {
  case 0:
    b = T(0.28209479177387814f);
    break;
  case 1:
    b = -T(0.4886025119029199f) * y;
    break;
  case 2:
    b = T(0.4886025119029199f) * z;
    break;
  case 3:
    b = -T(0.4886025119029199f) * x;
    break;
  case 4:
    b = T(1.0925484305920792f) * xy;
    break;
  case 5:
    b = -T(1.0925484305920792f) * yz;
    break;
  case 6:
    b = T(0.31539156525252005f) * (2 * zz - xx - yy);
    break;
  case 7:
    b = -T(1.0925484305920792f) * xz;
    break;
  case 8:
    b = T(0.5462742152960396f) * (xx - yy);
    break;
  case 9:
    b = -T(0.5900435899266435f) * y * (3 * xx - yy);
    break;
  case 10:
    b = T(2.890611442640554f) * xy * z;
    break;
  case 11:
    b = -T(0.4570457994644658f) * y * (4 * zz - xx - yy);
    break;
  case 12:
    b = T(0.3731763325901154f) * z * (2 * zz - 3 * xx - 3 * yy);
    break;
  case 13:
    b = -T(0.4570457994644658f) * x * (4 * zz - xx - yy);
    break;
  case 14:
    b = T(1.445305721320277f) * z * (xx - yy);
    break;
  case 15:
    b = -T(0.5900435899266435f) * x * (xx - 3 * yy);
    break;
  }
  return b * (c[0] * sr + c[1] * sg + c[2] * sb);
}

template <class T>
__host__ __device__ T direction_loss(const T* p, T cx, T cy, T cz, T sx, T sy,
                                     T sz) {
  T x = p[0] - cx, y = p[1] - cy, z = p[2] - cz;
  T r = std::sqrt(x * x + y * y + z * z);
  return x / r * sx + y / r * sy + z / r * sz;
}

__device__ void term_gradient(const float* d, const float* c, int j, float sr,
                              float sg, float sb, float* dd, float* dc) {
  auto derivative = clad::gradient(term<float>, "d,c");
  derivative.execute(d, c, j, sr, sg, sb, dd, dc);
}

struct SGD {
  static constexpr bool update_invisible = false;
  float* parameter;
  __device__ const float* coefficients(int k) const { return parameter + k; }
  __device__ void apply(int k, const float* gradient) const {
#pragma unroll
    for (int ch = 0; ch < 3; ++ch)
      parameter[k + ch] -= 1e-3f * gradient[ch];
  }
};

__device__ inline void adam_step(float& parameter, float g, float& first,
                                 float& second, float rate, AdamConfig config) {
  first = .9f * first + .1f * g;
  second = .999f * second + .001f * g * g;
  parameter -= rate * (first * config.inv_bias1) /
               (std::sqrt(second * config.inv_bias2) + 1e-15f);
}

struct Adam {
  static constexpr bool update_invisible = true;
  float *parameter, *first, *second;
  AdamConfig config;
  __device__ const float* coefficients(int k) const { return parameter + k; }
  __device__ void apply(int k, const float* gradient) const {
#pragma unroll
    for (int ch = 0; ch < 3; ++ch) {
      int i = k + ch;
      float rate = i % 48 < 3 ? .0025f : .000125f;
      adam_step(parameter[i], gradient[ch], first[i], second[i], rate, config);
    }
  }
};

__device__ inline void coefficient_step(float& parameter, float g, float& first,
                                        float& second, bool is_dc,
                                        AdamConfig config) {
  adam_step(parameter, g, first, second, is_dc ? .0025f : .000125f, config);
}

__device__ inline void coefficient_step(float& parameter, float g, float& first,
                                        float& second, bool is_dc,
                                        TorchAdamConfig config) {
  // PyTorch fused Adam: double beta and lr, float moments.
  const double beta1 = .9, beta2 = .999, eps = 1e-15;
  const double rate = is_dc ? .0025 : .000125;
  first = beta1 * first + (1 - beta1) * g;
  second = beta2 * second + (1 - beta2) * g * g;
  const float step_size = rate / config.bias1;
  const float denom = (sqrtf(second) / config.bias2_sqrt) + eps;
  parameter -= step_size * first / denom;
}

template <class Config> struct SplitAdam {
  static constexpr bool update_invisible = true;
  const float* parameter;
  float *dc, *rest, *dc_first, *dc_second, *rest_first, *rest_second;
  Config config;
  __device__ const float* coefficients(int k) const {
    if (parameter)
      return parameter + k;
    return k % 48 == 0 ? dc + (k / 48) * 3
                      : rest + (k / 48) * 45 + k % 48 - 3;
  }
  __device__ void apply(int k, const float* gradient) const {
    bool is_dc = k % 48 == 0;
    int base = is_dc ? (k / 48) * 3 : (k / 48) * 45 + k % 48 - 3;
    float* output = is_dc ? dc : rest;
    float* first = is_dc ? dc_first : rest_first;
    float* second = is_dc ? dc_second : rest_second;
#pragma unroll
    for (int ch = 0; ch < 3; ++ch)
      coefficient_step(output[base + ch], gradient[ch], first[base + ch],
                       second[base + ch], is_dc, config);
  }
};

template <class Update>
__global__ void consume(int n, int degree, const float* means, const int* radii,
                        const bool* clamped, const float* camera,
                        const float* seed, float* dm, Update update) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  int i = t / 16, j = t % 16;
  if (i >= n)
    return;
  bool visible = radii[i] > 0;
  if (!visible && !Update::update_invisible)
    return;
  unsigned mask = __activemask();
  float x = 0, y = 0, z = 0;
  if (j == 0 && visible) {
    float dx = means[3 * i] - camera[0];
    float dy = means[3 * i + 1] - camera[1];
    float dz = means[3 * i + 2] - camera[2];
    float r = sqrtf(dx * dx + dy * dy + dz * dz);
    x = dx / r;
    y = dy / r;
    z = dz / r;
  }
  x = __shfl_sync(mask, x, 0, 16);
  y = __shfl_sync(mask, y, 0, 16);
  z = __shfl_sync(mask, z, 0, 16);
  float dir[3] = {x, y, z}, dd[3] = {}, dc[3] = {};
  if (visible && j < (degree + 1) * (degree + 1))
    term_gradient(dir, update.coefficients(48 * i + 3 * j), j,
                  clamped[3 * i] ? 0.f : seed[3 * i],
                  clamped[3 * i + 1] ? 0.f : seed[3 * i + 1],
                  clamped[3 * i + 2] ? 0.f : seed[3 * i + 2], dd, dc);
  // This lane is done reading its three coefficients.
  // Zero gradients still update Adam moments.
  update.apply(48 * i + 3 * j, dc);
  for (int offset = 8; offset > 0; offset /= 2) {
    dd[0] += __shfl_down_sync(mask, dd[0], offset, 16);
    dd[1] += __shfl_down_sync(mask, dd[1], offset, 16);
    dd[2] += __shfl_down_sync(mask, dd[2], offset, 16);
  }
  if (j == 0 && visible) {
    float result[3] = {};
    auto derivative = clad::gradient(direction_loss<float>, "p");
    derivative.execute(means + 3 * i, camera[0], camera[1], camera[2], dd[0],
                       dd[1], dd[2], result);
    dm[3 * i] += result[0];
    dm[3 * i + 1] += result[1];
    dm[3 * i + 2] += result[2];
  }
}

void validate(int degree) {
  if (degree < 0 || degree > 3)
    throw std::invalid_argument("SH degree must be between zero and three");
}
} // namespace sh_consumer

extern "C" void sh_consumer_sgd(int n, int degree, const float* means,
                                const int* radii, float* coeff,
                                const bool* clamped, const float* camera,
                                const float* seed, float* dm) {
  if (n <= 0)
    return;
  sh_consumer::validate(degree);
  sh_consumer::consume<<<(size_t(n) * 16 + 255) / 256, 256>>>(
      n, degree, means, radii, clamped, camera, seed, dm,
      sh_consumer::SGD{coeff});
}

extern "C" void sh_consumer_adam(int n, int degree, const float* means,
                                 const int* radii, float* coeff,
                                 const bool* clamped, const float* camera,
                                 const float* seed, float* dm, float* first,
                                 float* second, AdamConfig config) {
  if (n <= 0)
    return;
  sh_consumer::validate(degree);
  sh_consumer::consume<<<(size_t(n) * 16 + 255) / 256, 256>>>(
      n, degree, means, radii, clamped, camera, seed, dm,
      sh_consumer::Adam{coeff, first, second, config});
}

extern "C" void sh_consumer_split_adam(int n, int degree, const float* means,
                                       const int* radii,
                                       const float* coefficient,
                                       const bool* clamped, const float* camera,
                                       const float* seed, float* dm, float* dc,
                                       float* rest, float* dc_first,
                                       float* dc_second, float* rest_first,
                                       float* rest_second, AdamConfig config) {
  if (n <= 0)
    return;
  sh_consumer::validate(degree);
  sh_consumer::consume<<<(size_t(n) * 16 + 255) / 256, 256>>>(
      n, degree, means, radii, clamped, camera, seed, dm,
      sh_consumer::SplitAdam<AdamConfig>{coefficient, dc, rest, dc_first,
                                         dc_second, rest_first, rest_second,
                                         config});
}

/// A null coefficient pointer reads dc/rest per lane.
extern "C" void sh_consumer_split_torch_adam(
    int n, int degree, const float* means, const int* radii,
    const float* coefficient, const bool* clamped, const float* camera,
    const float* seed, float* dm, float* dc, float* rest, float* dc_first,
    float* dc_second, float* rest_first, float* rest_second,
    TorchAdamConfig config) {
  if (n <= 0)
    return;
  sh_consumer::validate(degree);
  sh_consumer::consume<<<(size_t(n) * 16 + 255) / 256, 256>>>(
      n, degree, means, radii, clamped, camera, seed, dm,
      sh_consumer::SplitAdam<TorchAdamConfig>{coefficient, dc, rest, dc_first,
                                              dc_second, rest_first,
                                              rest_second, config});
}

#ifndef SH_CONSUMER_NO_MAIN
#include "WarpAccumulation.cuh"

namespace {
void checked(cudaError_t status) {
  if (status != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(status));
}

template <class T> T* upload(const std::vector<T>& values) {
  T* result;
  checked(cudaMalloc(&result, values.size() * sizeof(T)));
  checked(cudaMemcpy(result, values.data(), values.size() * sizeof(T),
                     cudaMemcpyHostToDevice));
  return result;
}

__global__ void shared_term_gradients(int n, const int* keys,
                                      const unsigned char* active, float* out,
                                      int* grouped) {
  int i = blockIdx.x * blockDim.x * blockDim.y * blockDim.z +
          (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
  if (i == 0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
    grouped[n] = 1;
#endif
  }
  if (i >= n || !active[i])
    return;
  float direction[3] = {.3f, -.4f, .5f}, coefficient[3] = {.2f, -.1f, .4f};
  float dd[3] = {}, dc[3] = {};
  float seed = .125f * (i % 11 - 5);
  sh_consumer::term_gradient(direction, coefficient, 4, seed, -2 * seed,
                             .5f * seed, dd, dc);
  bool peers = clad::cuda::full_warp_key(keys[i]);
  grouped[i] = peers;
#pragma unroll
  for (int ch = 0; ch < 3; ++ch)
    clad::cuda::warp_accumulate(peers, out + 3 * keys[i] + ch, dc[ch]);
}

size_t check_warp_accumulation() {
  size_t comparisons = 0;
  for (int n : {0, 1, 33, 257}) {
    for (int pattern = 0; pattern < 5; ++pattern) {
      std::vector<int> keys(std::max(n, 1)), flags(n + 1, 0);
      std::vector<unsigned char> active(std::max(n, 1), 1);
      std::vector<float> initial(3 * 32), actual(3 * 32);
      std::vector<double> expected(3 * 32);
      for (int k = 0; k < 3 * 32; ++k)
        initial[k] = expected[k] = .125f * (k % 5 - 2);
      for (int i = 0; i < n; ++i) {
        keys[i] = pattern == 1   ? i % 32
                  : pattern == 2 ? (i % 32) / 8
                  : pattern == 4 ? i / 32 % 3
                                 : 0;
        active[i] = pattern != 3 || i % 3 != 0;
        if (!active[i])
          continue;
        double seed = .125 * (i % 11 - 5);
        double basis = double(1.0925484305920792f) * double(.3f) * double(-.4f);
        for (int ch = 0; ch < 3; ++ch)
          expected[3 * keys[i] + ch] += basis * seed *
                                        (ch == 0   ? 1
                                         : ch == 1 ? -2
                                                   : .5);
      }
      auto* key = upload(keys);
      auto* mask = upload(active);
      auto* out = upload(initial);
      auto* grouped = upload(flags);
      if (n)
        shared_term_gradients<<<(n + 31) / 32,
                                pattern == 0 ? dim3(8, 2, 2) : dim3(16, 2)>>>(
            n, key, mask, out, grouped);
      checked(cudaGetLastError());
      checked(cudaMemcpy(actual.data(), out, actual.size() * sizeof(float),
                         cudaMemcpyDeviceToHost));
      checked(cudaMemcpy(flags.data(), grouped, flags.size() * sizeof(int),
                         cudaMemcpyDeviceToHost));
      for (size_t k = 0; k < actual.size(); ++k) {
        if (!std::isfinite(actual[k]) ||
            std::abs(actual[k] - expected[k]) >
                5e-6 + 2e-5 * std::abs(expected[k]))
          throw std::runtime_error(
              "Warp accumulation lost a generated adjoint");
        ++comparisons;
      }
      for (int i = 0; i < n; ++i) {
        bool full = flags[n] && (pattern == 0 || pattern == 4) &&
                    (i / 32 + 1) * 32 <= n;
        if (flags[i] != full)
          throw std::runtime_error(
              "Warp accumulation selected the wrong group");
        ++comparisons;
      }
      for (void* p : {(void*)key, (void*)mask, (void*)out, (void*)grouped})
        checked(cudaFree(p));
    }
  }
  return comparisons;
}

std::vector<float> reorder_groups(const std::vector<float>& values,
                                  bool split) {
  size_t n = values.size() / 48;
  std::vector<float> result(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    size_t gaussian = i / 48, term = i % 48;
    size_t j =
        term < 3 ? 3 * gaussian + term : 3 * n + 45 * gaussian + term - 3;
    if (split)
      result[j] = values[i];
    else
      result[i] = values[j];
  }
  return result;
}

// Reference loss uses every basis. The switch above is what Clad
// differentiated.
double reference_loss(const double* p, const double* c, const double* seed,
                      const std::vector<float>& camera, int degree) {
  double dx = p[0] - camera[0], dy = p[1] - camera[1], dz = p[2] - camera[2];
  double r = std::sqrt(dx * dx + dy * dy + dz * dz);
  double x = dx / r, y = dy / r, z = dz / r;
  double xx = x * x, yy = y * y, zz = z * z;
  double b[] = {double(.28209479177387814f),
                -double(.4886025119029199f) * y,
                double(.4886025119029199f) * z,
                -double(.4886025119029199f) * x,
                double(1.0925484305920792f) * x * y,
                -double(1.0925484305920792f) * y * z,
                double(.31539156525252005f) * (2 * zz - xx - yy),
                -double(1.0925484305920792f) * x * z,
                double(.5462742152960396f) * (xx - yy),
                -double(.5900435899266435f) * y * (3 * xx - yy),
                double(2.890611442640554f) * x * y * z,
                -double(.4570457994644658f) * y * (4 * zz - xx - yy),
                double(.3731763325901154f) * z * (2 * zz - 3 * xx - 3 * yy),
                -double(.4570457994644658f) * x * (4 * zz - xx - yy),
                double(1.445305721320277f) * z * (xx - yy),
                -double(.5900435899266435f) * x * (xx - 3 * yy)};
  double result = 0;
  for (int j = 0; j < (degree + 1) * (degree + 1); ++j)
    result += b[j] * (c[3 * j] * seed[0] + c[3 * j + 1] * seed[1] +
                      c[3 * j + 2] * seed[2]);
  return result;
}
} // namespace

int main() try {
  constexpr int n = 257;
  std::vector<float> means(3 * n), coeff(48 * n), seed(3 * n), initial(3 * n);
  std::vector<float> camera{.1f, -.2f, .3f}, first(48 * n), second(48 * n);
  std::vector<int> radii(n);
  std::vector<unsigned char> clamp(3 * n);
  for (int i = 0; i < n; ++i) {
    means[3 * i] = camera[0] + .03f * (i % 19 - 9);
    means[3 * i + 1] = camera[1] + .04f * (i % 13 - 6);
    means[3 * i + 2] = camera[2] + (i % 2 ? 1.2f : -1.4f);
    radii[i] = i % 7 ? 1 : 0;
    for (int ch = 0; ch < 3; ++ch) {
      clamp[3 * i + ch] = (i + ch) % (5 + 2 * ch) == 0;
      seed[3 * i + ch] = i % 17 ? .1f * ((i + 3 * ch) % 17 - 8) : 0;
      initial[3 * i + ch] = .0625f * (ch - 1);
    }
    for (int k = 0; k < 48; ++k) {
      int field = 48 * i + k;
      coeff[field] = .03f * ((i + k) % 23 - 11);
      first[field] = 1e-5f * float(field % 23 - 11) / 11;
      second[field] = 1e-8f * (1 + float(field % 17) / 17);
      if (field % 19 == 0 && (k < 3 || !radii[i]))
        first[field] = second[field] = 0;
    }
    if (i % 14 == 0)
      for (int ch = 0; ch < 3; ++ch)
        means[3 * i + ch] = camera[ch];
  }
  AdamConfig config{float(1 / (1 - std::pow(.9, 123))),
                    float(1 / (1 - std::pow(.999, 123)))};
  TorchAdamConfig torch_config{float(1 - std::pow(.9, 123)),
                               float(std::sqrt(1 - std::pow(.999, 123)))};
  sh_consumer_sgd(0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                  nullptr);
  sh_consumer_adam(0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                   nullptr, nullptr, nullptr, config);
  sh_consumer_split_adam(0, 0, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                         nullptr, nullptr, config);
  sh_consumer_split_torch_adam(
      0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, torch_config);
  checked(cudaGetLastError());
  float *m = upload(means), *c = upload(coeff), *s = upload(seed);
  float *cam = upload(camera), *dm = upload(initial);
  float *m1 = upload(first), *m2 = upload(second);
  auto separate_coeff = reorder_groups(coeff, true);
  auto separate_first = reorder_groups(first, true);
  auto separate_second = reorder_groups(second, true);
  float *split_c = upload(separate_coeff), *split_m1 = upload(separate_first),
        *split_m2 = upload(separate_second);
  int* r = upload(radii);
  unsigned char* b = upload(clamp);
  bool rejected = false;
  try {
    sh_consumer_sgd(n, 4, m, r, c, (bool*)b, cam, s, dm);
  } catch (const std::invalid_argument&) {
    rejected = true;
  }
  if (!rejected)
    throw std::runtime_error("unsupported SH degree was accepted");
  size_t components = 0, bad = 0;
  for (int degree = 0; degree < 4; ++degree) {
    std::vector<double> gradient(51 * n);
    for (int i = 0; i < n; ++i) {
      if (!radii[i])
        continue;
      double p[3], h[48], weights[3];
      for (int j = 0; j < 3; ++j) {
        p[j] = means[3 * i + j];
        weights[j] = clamp[3 * i + j] ? 0 : seed[3 * i + j];
      }
      for (int j = 0; j < 48; ++j)
        h[j] = coeff[48 * i + j];
      for (int k = 0; k < 51; ++k) {
        double* value = k < 3 ? p + k : h + k - 3;
        double old = *value;
        *value = old + 1e-5;
        double plus = reference_loss(p, h, weights, camera, degree);
        *value = old - 1e-5;
        double minus = reference_loss(p, h, weights, camera, degree);
        *value = old;
        gradient[51 * i + k] = radii[i] ? (plus - minus) / 2e-5 : 0;
      }
    }
    for (int mode : {0, 1, 2, 3, 4}) {
      bool adam = mode != 0, split = mode >= 2, torch_adam = mode >= 3;
      checked(cudaMemcpy(c, coeff.data(), coeff.size() * 4,
                         cudaMemcpyHostToDevice));
      checked(cudaMemcpy(dm, initial.data(), initial.size() * 4,
                         cudaMemcpyHostToDevice));
      checked(cudaMemcpy(m1, first.data(), first.size() * 4,
                         cudaMemcpyHostToDevice));
      checked(cudaMemcpy(m2, second.data(), second.size() * 4,
                         cudaMemcpyHostToDevice));
      if (split) {
        checked(cudaMemcpy(split_c, separate_coeff.data(), coeff.size() * 4,
                           cudaMemcpyHostToDevice));
        checked(cudaMemcpy(split_m1, separate_first.data(), first.size() * 4,
                           cudaMemcpyHostToDevice));
        checked(cudaMemcpy(split_m2, separate_second.data(), second.size() * 4,
                           cudaMemcpyHostToDevice));
        if (torch_adam)
          sh_consumer_split_torch_adam(n, degree, m, r, mode == 4 ? nullptr : c,
                                       (bool*)b, cam, s, dm,
                                       split_c, split_c + 3 * n, split_m1,
                                       split_m2, split_m1 + 3 * n,
                                       split_m2 + 3 * n, torch_config);
        else
          sh_consumer_split_adam(n, degree, m, r, c, (bool*)b, cam, s, dm,
                                 split_c, split_c + 3 * n, split_m1, split_m2,
                                 split_m1 + 3 * n, split_m2 + 3 * n, config);
      } else if (adam)
        sh_consumer_adam(n, degree, m, r, c, (bool*)b, cam, s, dm, m1, m2,
                         config);
      else
        sh_consumer_sgd(n, degree, m, r, c, (bool*)b, cam, s, dm);
      checked(cudaDeviceSynchronize());
      std::vector<float> got_mean(3 * n), got_coeff(48 * n);
      std::vector<float> got_first(48 * n), got_second(48 * n);
      checked(cudaMemcpy(got_mean.data(), dm, got_mean.size() * 4,
                         cudaMemcpyDeviceToHost));
      checked(cudaMemcpy(got_coeff.data(), split ? split_c : c,
                         got_coeff.size() * 4, cudaMemcpyDeviceToHost));
      checked(cudaMemcpy(got_first.data(), split ? split_m1 : m1,
                         got_first.size() * 4, cudaMemcpyDeviceToHost));
      checked(cudaMemcpy(got_second.data(), split ? split_m2 : m2,
                         got_second.size() * 4, cudaMemcpyDeviceToHost));
      if (split) {
        got_coeff = reorder_groups(got_coeff, false);
        got_first = reorder_groups(got_first, false);
        got_second = reorder_groups(got_second, false);
      }
      auto compare = [&](double got, double expected, double atol,
                         double rtol) {
        ++components;
        if (!std::isfinite(got) ||
            std::abs(got - expected) > atol + rtol * std::abs(expected)) {
          if (bad < 5)
            std::fprintf(
                stderr, "degree=%d mode=%d component=%zu got=%.12g ref=%.12g\n",
                degree, mode, components, got, expected);
          ++bad;
        }
      };
      for (int i = 0; i < n; ++i) {
        for (int k = 0; k < 3; ++k)
          compare(got_mean[3 * i + k],
                  initial[3 * i + k] + gradient[51 * i + k], 2e-6, 2e-4);
        for (int k = 0; k < 48; ++k) {
          int field = 48 * i + k;
          double g = gradient[51 * i + k + 3];
          double expected = coeff[field] - 1e-3 * g;
          if (adam) {
            double a = .9 * first[field] + .1 * g;
            double v = .999 * second[field] + .001 * g * g;
            if (torch_adam) {
              double rate = k < 3 ? .0025 : .000125;
              expected = coeff[field] -
                         (rate / torch_config.bias1) * a /
                             (std::sqrt(v) / torch_config.bias2_sqrt + 1e-15);
            } else {
              double rate = k < 3 ? double(.0025f) : double(.000125f);
              expected =
                  coeff[field] - rate * (a * config.inv_bias1) /
                                     (std::sqrt(v * config.inv_bias2) + 1e-15);
            }
            compare(got_first[field], a, 2e-9, 2e-4);
            compare(got_second[field], v, 2e-11, 2e-4);
          }
          compare(got_coeff[field], expected, 5e-8, 4e-7);
        }
      }
    }
  }
  for (void* p : {(void*)m, (void*)c, (void*)s, (void*)cam, (void*)dm,
                  (void*)m1, (void*)m2, (void*)r, (void*)b, (void*)split_c,
                  (void*)split_m1, (void*)split_m2})
    checked(cudaFree(p));
  if (bad)
    throw std::runtime_error("SH gradient consumer comparison failed");
  size_t warp_checks = check_warp_accumulation();
  std::printf("SH gradient consumer checks: %zu OK\n", components);
  std::printf("Warp accumulation checks: %zu OK\n", warp_checks);
  return 0;
} catch (const std::exception& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 1;
}
#endif
