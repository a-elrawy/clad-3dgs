// SPDX-License-Identifier: Apache-2.0
#include <ATen/MemoryOverlap.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <climits>
#include <cmath>
#include <torch/extension.h>

extern "C" void launch_quaternion_backward(int, const float*, const float*,
                                           const float*, float*, cudaStream_t);
extern "C" void launch_activation_consume(
    int, float* const*, const float* const*, const float* const*, float* const*,
    float* const*, const double*, const float*, const float*, cudaStream_t);

void validate_quaternion(torch::Tensor q, torch::Tensor seed,
                         torch::Tensor norms) {
  TORCH_CHECK(q.is_cuda() && seed.is_cuda() && norms.is_cuda() &&
                  q.device() == seed.device() && q.device() == norms.device(),
              "Quaternion inputs must share a CUDA device");
  TORCH_CHECK(q.scalar_type() == torch::kFloat32 &&
                  seed.scalar_type() == torch::kFloat32 &&
                  norms.scalar_type() == torch::kFloat32,
              "Quaternion inputs must be float32");
  TORCH_CHECK(q.dim() == 2 && q.size(1) == 4 && q.sizes() == seed.sizes() &&
                  q.size(0) <= INT_MAX / 4,
              "Quaternion inputs must have matching [N, 4] shapes");
  TORCH_CHECK(norms.dim() == 2 && norms.size(0) == q.size(0) &&
                  norms.size(1) == 1,
              "Saved quaternion norms must have shape [N, 1]");
}

torch::Tensor quaternion_backward(torch::Tensor q, torch::Tensor seed,
                                  torch::Tensor norms) {
  validate_quaternion(q, seed, norms);
  c10::cuda::CUDAGuard guard(q.device());
  auto input = q.contiguous();
  auto incoming = seed.contiguous();
  auto saved_norms = norms.contiguous();
  auto output = torch::empty_like(input);
  launch_quaternion_backward(
      input.size(0), input.data_ptr<float>(), incoming.data_ptr<float>(),
      saved_norms.data_ptr<float>(), output.data_ptr<float>(),
      at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

void activation_consume(std::vector<torch::Tensor> parameters,
                        std::vector<torch::Tensor> seeds,
                        std::vector<torch::Tensor> saved,
                        std::vector<torch::Tensor> first,
                        std::vector<torch::Tensor> second,
                        std::vector<double> rates, std::vector<double> bias1,
                        std::vector<double> bias2) {
  TORCH_CHECK(parameters.size() == 3 && seeds.size() == 3 &&
                  saved.size() == 3 && first.size() == 3 &&
                  second.size() == 3 && rates.size() == 3 &&
                  bias1.size() == 3 && bias2.size() == 3,
              "Activation consumption requires rotation, scaling and opacity");
  validate_quaternion(parameters[0], seeds[0], saved[0]);
  auto device = parameters[0].device();
  auto n = parameters[0].size(0);
  TORCH_CHECK(n <= INT_MAX / 4, "Activation batch exceeds index range");
  c10::cuda::CUDAGuard guard(device);
  std::vector<torch::Tensor> writable;
  const int widths[] = {4, 3, 1};
  for (int j = 0; j < 3; ++j) {
    auto parameter = parameters[j];
    TORCH_CHECK(parameter.is_cuda() && parameter.device() == device &&
                    parameter.scalar_type() == torch::kFloat32 &&
                    parameter.dim() == 2 && parameter.size(0) == n &&
                    parameter.size(1) == widths[j] && parameter.is_contiguous(),
                "Unsupported activation parameter layout");
    for (const auto& value : {first[j], second[j]})
      TORCH_CHECK(
          value.device() == device && value.scalar_type() == torch::kFloat32 &&
              value.sizes() == parameter.sizes() && value.is_contiguous(),
          "Activation moments must match their parameter");
    TORCH_CHECK(seeds[j].device() == device &&
                    seeds[j].scalar_type() == torch::kFloat32 &&
                    seeds[j].sizes() == parameter.sizes(),
                "Invalid activation seed");
    TORCH_CHECK(saved[j].device() == device &&
                    saved[j].scalar_type() == torch::kFloat32 &&
                    saved[j].dim() == 2 && saved[j].size(0) == n &&
                    saved[j].size(1) == (j == 1 ? 3 : 1),
                "Invalid saved activation");
    TORCH_CHECK(std::isfinite(rates[j]) && rates[j] >= 0 &&
                    std::isfinite(bias1[j]) && bias1[j] > 0 && bias1[j] <= 1 &&
                    std::isfinite(bias2[j]) && bias2[j] > 0 && bias2[j] <= 1,
                "Invalid activation Adam configuration");
    seeds[j] = seeds[j].contiguous();
    saved[j] = saved[j].contiguous();
    writable.insert(writable.end(), {parameter, first[j], second[j]});
  }
  for (size_t j = 0; j < writable.size(); ++j) {
    for (size_t k = j + 1; k < writable.size(); ++k)
      at::assert_no_overlap(writable[j], writable[k]);
    for (const auto& value : seeds)
      at::assert_no_overlap(writable[j], value);
    for (const auto& value : saved)
      at::assert_no_overlap(writable[j], value);
  }
  float* p[3];
  float* m[3];
  float* v[3];
  const float* d[3];
  const float* s[3];
  float b1[3], b2[3];
  for (int j = 0; j < 3; ++j) {
    p[j] = parameters[j].data_ptr<float>();
    m[j] = first[j].data_ptr<float>();
    v[j] = second[j].data_ptr<float>();
    d[j] = seeds[j].data_ptr<float>();
    s[j] = saved[j].data_ptr<float>();
    b1[j] = bias1[j];
    b2[j] = bias2[j];
  }
  launch_activation_consume(n, p, d, s, m, v, rates.data(), b1, b2,
                            at::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("backward", &quaternion_backward);
  module.def("consume_activations", &activation_consume);
}
