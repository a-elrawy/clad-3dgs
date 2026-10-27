"""Pinned rasterizer copy that can read separate DC and rest buffers."""

import re
import shutil

INPUTS = r"""#pragma once
#include <cuda_runtime.h>
#include <glm/glm.hpp>

struct SHInputs {
  const float *packed, *dc, *rest;
  __host__ __device__ SHInputs(const float* values = nullptr)
      : packed(values), dc(nullptr), rest(nullptr) {}
  __host__ __device__ SHInputs(const float* first, const float* higher)
      : packed(nullptr), dc(first), rest(higher) {}
  __host__ __device__ explicit operator bool() const {
    return packed != nullptr || dc != nullptr;
  }
};

template <bool Split> struct SHRecord {
  SHInputs inputs;
  int index, width;
  __device__ glm::vec3 operator[](int term) const {
    if constexpr (Split) {
      const float* p = term == 0 ? inputs.dc + 3 * index
                               : inputs.rest + 45 * index + 3 * (term - 1);
      return glm::vec3(p[0], p[1], p[2]);
    } else {
      return reinterpret_cast<const glm::vec3*>(inputs.packed)[index * width + term];
    }
  }
};
"""


def replace(text, old, new, count=1):
    if text.count(old) != count:
        raise ValueError(f"Unexpected pinned source pattern: {old!r}")
    return text.replace(old, new)


def helper(text, arguments):
    start = text.index("__device__", text.index("namespace cg"))
    start = (
        text.index("__device__", start + 1)
        if "float sq(" in text[start : text.index("{", start)]
        else start
    )
    signature_end = text.index("{", start)
    signature = text[start:signature_end]
    if "computeColorFromSH(" not in signature:
        raise ValueError("Expected the SH helper")
    depth, end = 1, signature_end + 1
    while depth:
        depth += (text[end] == "{") - (text[end] == "}")
        end += 1
    body = text[signature_end:end]
    body = replace(
        body,
        "glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;",
        "const SHRecord<Split> sh{shs, idx, max_coeffs};",
    )
    implementation = (
        "template <bool Split>\n"
        + signature.replace("computeColorFromSH(", "computeColorFromSHImpl(")
        + body
    )
    wrapper = (
        signature
        + "{\n  if (shs.rest)\n    return computeColorFromSHImpl<true>("
        + arguments
        + ");\n  return computeColorFromSHImpl<false>("
        + arguments
        + ");\n}"
    )
    return text[:start] + implementation + "\n\n" + wrapper + text[end:]


def prepare(native, output):
    copied = output / "direct_native"
    copied.mkdir()
    shutil.copytree(native / "cuda_rasterizer", copied / "cuda_rasterizer")
    for name in ("rasterize_points.cu", "rasterize_points.h", "LICENSE.md"):
        shutil.copy2(native / name, copied / name)
    (copied / "cuda_rasterizer/sh_inputs.h").write_text(INPUTS)
    for name in (
        "forward.cu",
        "forward.h",
        "backward.cu",
        "backward.h",
        "rasterizer.h",
        "rasterizer_impl.cu",
    ):
        path = copied / "cuda_rasterizer" / name
        text = path.read_text()
        text = re.sub(r"const float\* shs\b", "SHInputs shs", text)
        if name in ("forward.h", "backward.h", "rasterizer.h"):
            text = '#include "sh_inputs.h"\n' + text
        if name == "forward.cu":
            text = helper(text, "idx, deg, max_coeffs, means, campos, shs, clamped")
        elif name == "backward.cu":
            text = helper(
                text,
                "idx, deg, max_coeffs, means, campos, shs, clamped, dL_dcolor, dL_dmeans, dL_dshs",
            )
        path.write_text(text)

    source = (native / "rasterize_points.cu").read_text()
    forward_start = source.index("std::tuple<int,")
    backward_start = source.index("std::tuple<torch::Tensor,", forward_start)
    end = source.index("\ntorch::Tensor markVisible", backward_start)
    functions = []
    for text, original, renamed in (
        (
            source[forward_start:backward_start],
            "RasterizeGaussiansCUDA",
            "RasterizeDirectForwardCUDA",
        ),
        (
            source[backward_start:end],
            "RasterizeGaussiansBackwardCUDA",
            "RasterizeDirectBackwardCUDA",
        ),
    ):
        text = replace(text, original + "(", renamed + "(")
        text = replace(
            text,
            "const torch::Tensor& sh,",
            "const torch::Tensor& dc, const torch::Tensor& rest,",
        )
        text = re.sub(
            r"int M = 0;\s*if\(sh.size\(0\) != 0\)\s*\{\s*M = sh.size\(1\);\s*\}",
            "int M = 16;",
            text,
        )
        text, replacements = re.subn(
            r"\bsh\.contiguous\(\)\.data(?:_ptr)?<float>\(\),",
            "SHInputs(dc.data_ptr<float>(), rest.data_ptr<float>()),",
            text,
        )
        if replacements != 1:
            raise ValueError("Expected one native SH input")
        text = (
            text[: text.index("{") + 1]
            + "\n  check_inputs(means3D, dc, rest, degree);\n"
            + text[text.index("{") + 1 :]
        )
        functions.append(text)

    consumed = (output / "split_backward.cu").read_text()
    begin = consumed.index("std::tuple<torch::Tensor,")
    consumed = consumed[begin:]
    consumed = replace(
        consumed, "RasterizeSplitBackwardCUDA(", "RasterizeDirectConsumeBackwardCUDA("
    )
    consumed = replace(
        consumed, "const auto packed_coefficients = sh.contiguous();", ""
    )
    consumed = replace(
        consumed,
        'TORCH_CHECK(degree >= 0 && degree <= 3 && sh.size(1) == 16, "unsupported SH layout");',
        'check_inputs(means3D, dc, rest, degree);\n      TORCH_CHECK(sh.numel() == 0, "direct input path needs no coefficient snapshot");\n      check_disjoint({dc, rest, dc_first, dc_second, rest_first, rest_second});',
    )
    consumed = re.sub(
        r"int M = 0;\s*if\(sh.size\(0\) != 0\)\s*\{\s*M = sh.size\(1\);\s*\}",
        "int M = 16;",
        consumed,
    )
    consumed = replace(consumed, "packed_coefficients.data_ptr<float>()", "nullptr")
    functions.append(consumed)

    prefix = source[:forward_start]
    prefix = prefix[: prefix.index("std::function<char*")]
    prefix += r"""
#include <c10/cuda/CUDAStream.h>
#include "cuda_rasterizer/rasterizer_impl.h"
struct TorchAdamConfig { float bias1, bias2_sqrt; };
extern "C" void sh_consumer_split_torch_adam(int,int,const float*,const int*,const float*,const bool*,const float*,const float*,float*,float*,float*,float*,float*,float*,float*,TorchAdamConfig);
std::function<char*(size_t)> resizeFunctional(torch::Tensor&);

static void check_disjoint(std::initializer_list<torch::Tensor> tensors) {
  for (auto a = tensors.begin(); a != tensors.end(); ++a)
    for (auto b = a + 1; b != tensors.end(); ++b) {
      const auto x = reinterpret_cast<uintptr_t>(a->data_ptr());
      const auto y = reinterpret_cast<uintptr_t>(b->data_ptr());
      TORCH_CHECK(x + a->nbytes() <= y || y + b->nbytes() <= x,
                  "direct SH coefficient and moment buffers must be disjoint");
    }
}

static void check_inputs(const torch::Tensor& means, const torch::Tensor& dc,
                         const torch::Tensor& rest, int degree) {
  TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream() == nullptr,
              "direct SH inputs require the default CUDA stream");
  const auto n = means.size(0);
  TORCH_CHECK(degree >= 0 && degree <= 3 &&
              dc.sizes() == torch::IntArrayRef({n, 1, 3}) &&
              rest.sizes() == torch::IntArrayRef({n, 15, 3}), "unsupported direct SH layout");
  for (const auto& t : {dc, rest})
    TORCH_CHECK(t.is_cuda() && t.scalar_type() == torch::kFloat32 &&
                t.is_contiguous() && t.device() == means.device(),
                "direct SH tensors must be contiguous CUDA float32 on the renderer device");
  check_disjoint({dc, rest});
}
"""
    (output / "direct_inputs.cu").write_text(prefix + "\n".join(functions))
    declarations = "\n".join(f[: f.index("{")] + ";\n" for f in functions)
    (output / "direct_inputs.h").write_text(
        "#pragma once\n#include <torch/extension.h>\n" + declarations
    )
    ext = (output / "ext.cpp").read_text()
    ext = '#include "direct_inputs.h"\n' + ext
    ext = ext.replace(
        "\n}",
        '\n  m.def("rasterize_gaussians_direct", &RasterizeDirectForwardCUDA);\n  m.def("rasterize_gaussians_backward_direct", &RasterizeDirectBackwardCUDA);\n  m.def("rasterize_gaussians_backward_consume_direct", &RasterizeDirectConsumeBackwardCUDA);\n}',
        1,
    )
    (output / "ext.cpp").write_text(ext)
    return copied
