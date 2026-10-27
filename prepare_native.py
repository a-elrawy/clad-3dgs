"""Add the consumer entry. Native forward and backward stay."""

import re
from pathlib import Path
import shutil


def prepare_warp_accumulation(native, output):
    if output.resolve() not in native.resolve().parents:
        raise ValueError("Apply warp accumulation only to the copied native sources")
    path = native / "cuda_rasterizer/backward.cu"
    source = path.read_text()
    start = source.index("// Backward version of the rendering procedure.")
    end = source.index("void BACKWARD::preprocess(", start)
    shader = source[start:end]
    key = "const int global_id = collected_id[j];"
    if shader.count(key) != 1:
        raise ValueError("Unsupported raster backward contribution loop")
    shader = shader.replace(
        key, key + "\n            bool peers = clad::cuda::full_warp_key(global_id);"
    )
    shader, count = re.subn(
        r"atomicAdd\(([^;]+)\);", r"clad::cuda::warp_accumulate(peers, \1);", shader
    )
    if count != 8:
        raise ValueError("Unsupported raster backward atomic update sites")
    header = Path(__file__).parent / "WarpAccumulation.cuh"
    shutil.copy2(header, output / header.name)
    path.write_text('#include "WarpAccumulation.cuh"\n' + source[:start] + shader + source[end:])


def prepare(native, output):
    s = (native / "rasterize_points.cu").read_text()
    start = s.index(
        "std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>"
    )
    end = s.index("\ntorch::Tensor markVisible", start)
    f = s[start:end].strip()
    f = f.replace("RasterizeGaussiansBackwardCUDA(", "RasterizeSplitBackwardCUDA(", 1)
    f = f.replace(
        "const bool debug)",
        """const bool debug,
        torch::Tensor dc, torch::Tensor rest,
        torch::Tensor dc_first, torch::Tensor dc_second,
        torch::Tensor rest_first, torch::Tensor rest_second,
        const float bias1, const float bias2_sqrt)""",
        1,
    )
    assert "RasterizeSplitBackwardCUDA" in f and "const float bias2_sqrt)" in f
    f = f.replace(
        "  const int P = means3D.size(0);",
        """  TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream() == nullptr,
                  "SH consumption requires the default CUDA stream");
      const auto packed_means = means3D.contiguous();
      const auto packed_visibility = radii.contiguous();
      const auto packed_coefficients = sh.contiguous();
      const auto packed_camera = campos.contiguous();
      const int P = means3D.size(0);
      TORCH_CHECK(degree >= 0 && degree <= 3 && sh.size(1) == 16, "unsupported SH layout");
      for (const auto& tensor : {dc,rest,dc_first,dc_second,rest_first,rest_second}) {
        TORCH_CHECK(tensor.is_cuda() && tensor.scalar_type() == torch::kFloat32 &&
                    tensor.is_contiguous() && tensor.device() == means3D.device(),
                    "split coefficient/moment tensors must be contiguous CUDA float32 on the renderer device");
      }
      TORCH_CHECK(dc.numel()==3*P && dc_first.numel()==3*P && dc_second.numel()==3*P &&
                  rest.numel()==45*P && rest_first.numel()==45*P && rest_second.numel()==45*P,
                  "bad split coefficient or optimizer-state shape");""",
        1,
    )
    f = f.replace(
        "torch::zeros({P, M, 3}, means3D.options())",
        "torch::empty({0}, means3D.options())",
        1,
    )
    f = f.replace("sh.contiguous().data<float>(),", "nullptr,", 1)
    f = f.replace("dL_dsh.contiguous().data<float>(),", "nullptr,", 1)
    needle = "\n  return std::make_tuple("
    assert needle in f
    f = f.replace(
        needle,
        """
      if(P != 0) {
        char* g=reinterpret_cast<char*>(geomBuffer.data_ptr());
        auto geom=CudaRasterizer::GeometryState::fromChunk(g,P);
        sh_consumer_split_torch_adam(P,degree,means3D.data_ptr<float>(),radii.data_ptr<int>(),
          sh.data_ptr<float>(),geom.clamped,campos.data_ptr<float>(),
          dL_dcolors.data_ptr<float>(),dL_dmeans3D.data_ptr<float>(),
          dc.data_ptr<float>(),rest.data_ptr<float>(),dc_first.data_ptr<float>(),
          dc_second.data_ptr<float>(),rest_first.data_ptr<float>(),rest_second.data_ptr<float>(),
          TorchAdamConfig{bias1,bias2_sqrt});
        const auto status=cudaGetLastError();
        TORCH_CHECK(status == cudaSuccess,cudaGetErrorString(status));
      }
      return std::make_tuple(""",
        1,
    )
    for original, packed, scalar in [
        ("means3D", "packed_means", "float"),
        ("radii", "packed_visibility", "int"),
        ("sh", "packed_coefficients", "float"),
        ("campos", "packed_camera", "float"),
    ]:
        f = re.sub(
            rf"\b{original}\.(?:contiguous\(\)\.)?data(?:_ptr)?<{scalar}>\(\)",
            f"{packed}.data_ptr<{scalar}>()",
            f,
        )
    header = f[: f.index("{")] + ";\n"
    (output / "split_backward.h").write_text(
        "#pragma once\n#include <torch/extension.h>\n" + header
    )
    (output / "split_backward.cu").write_text(
        """// Adapter entry. The native backward below keeps its license.
    """
        + s[:start].split("std::function<char*")[0]
        + """
    #include <c10/cuda/CUDAStream.h>
    #include "cuda_rasterizer/rasterizer_impl.h"
    struct TorchAdamConfig { float bias1, bias2_sqrt; };
    extern "C" void sh_consumer_split_torch_adam(int,int,const float*,const int*,const float*,const bool*,const float*,const float*,float*,float*,float*,float*,float*,float*,float*,TorchAdamConfig);
    """
        + f
        + "\n"
    )
    ext = (
        (native / "ext.cpp")
        .read_text()
        .replace(
            '#include "rasterize_points.h"',
            '#include "rasterize_points.h"\n#include "split_backward.h"',
            1,
        )
    )
    ext = ext.replace(
        '  m.def("mark_visible", &markVisible);',
        '  m.def("mark_visible", &markVisible);\n  m.def("rasterize_gaussians_backward_consume", &RasterizeSplitBackwardCUDA);',
        1,
    )
    (output / "ext.cpp").write_text(ext)
