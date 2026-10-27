# Private SH gradients with a local optimizer step

Standalone demo of the idea behind the trainer integration: Clad differentiates each SH colour term and the view direction, and the kernel applies the coefficient adjoint in Adam before anything is written to a global SH gradient.

`SHGradientConsumer.cu` covers degrees 0 through 3. Sixteen CUDA lanes handle one Gaussian. Each lane finishes one term, then updates its three RGB coefficients. The lanes sum their direction adjoints and one lane adds that sum into the Gaussian's position adjoint.

The SH basis keeps the native renderer's products, including `xy`, `yz`, and `xz`. Direction derivatives, the lane reduction, and the normalization derivative evaluate in a different order from native backward. Pixel rasterization is a separate kernel. The full trainer hookup is [INTEGRATION.md](INTEGRATION.md).

Layouts, coefficient-major RGB:

- `means[N][3]`, `coeff[N][16][3]`, `clamped[N][3]`, `seed[N][3]`, `radii[N]`
- camera position: 3 floats
- radii and clamp flags come from the forward renderer
- visible Gaussians have positive radii and a position different from the camera
- position adjoints accumulate into an existing `dm[N][3]`

Host entry points, all using those term derivatives:

- `sh_consumer_sgd`: SGD at learning rate `1e-3`.
- `sh_consumer_adam`: betas `0.9` and `0.999`, epsilon `1e-15`, rates `0.0025`
  (DC) and `0.000125` (the rest). `AdamConfig` holds the two inverse bias
  corrections. Moments are `N*48` each. Invisible Gaussians, inactive degrees,
  and zero seeds still update moments and parameters.
- `sh_consumer_split_adam`: reads a merged `coefficient[N][16][3]` snapshot
  and writes `dc[N][1][3]` and `rest[N][15][3]`. Each group has its own moment
  buffers of the same shape. Same Adam settings. The snapshot must hold the
  old values of both groups (a forward renderer's concatenated coefficients
  are fine) and is not written.
- `sh_consumer_split_torch_adam`: same 3DGS settings and split buffers, with
  the expression types from PyTorch 2.5.1's
  [fused Adam helper](https://github.com/pytorch/pytorch/blob/v2.5.1/aten/src/ATen/native/cuda/fused_adam_utils.cuh):
  double beta and learning-rate arithmetic, float moments and bias corrections,
  float division, then the double epsilon is added.
  `TorchAdamConfig` is `bias1 = 1 - pow(0.9, step)` and
  `bias2_sqrt = sqrt(1 - pow(0.999, step))`, rounded to float. That is the
  opposite of `AdamConfig`'s inverse corrections. No PyTorch dependency.
  Matching the expression types does not make two compiler backends bitwise
  equal.

Writable parameters, the position adjoint, and moment buffers must not overlap
each other or the other inputs. Other accesses belong on the default CUDA
stream, ordered by the caller. Each term reads only its own three coefficients,
and those reads finish before that lane writes them. The caller must be done
with the old coefficients, and must not need a materialized SH gradient for
another loss, accumulation, or clipping. First order, one consumer.

Compile with CUDA-enabled Clang and the Clad plugin, then link the CUDA
runtime. The CUDA 12.1 setup used locally:

```sh
clang++ -std=c++17 -O3 --cuda-path=/usr/local/cuda-12.1 \
  --cuda-gpu-arch=sm_86 -I/path/to/clad/include -I/path/to/clad/build/include \
  -Xclang -load -Xclang /path/to/clad.so \
  -Xclang -add-plugin -Xclang clad \
  -c SHGradientConsumer.cu -o consumer.o
nvcc -arch=sm_86 consumer.o -o consumer
./consumer
```

The self-check compares position adjoints, updated coefficients, and Adam
moments with a separate finite-difference evaluator and the Adam equations.
It covers degrees 0–3, clipping, invisibility, zero and non-unit seeds,
nonzero incoming adjoints, zero and nonzero moments, and a partial warp. The
warp helper is also checked with generated adjoints and 3D thread blocks.
Empty launches and unsupported degrees are checked. Define
`SH_CONSUMER_NO_MAIN` to link the entry points into a renderer. The demo
checks correctness. Timing belongs on the native fixture.

The [trainer integration](INTEGRATION.md) binds dense Adam and keeps native
backward on steps where the trainer still mutates state.
