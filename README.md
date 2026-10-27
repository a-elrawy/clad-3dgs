# Clad integration for native 3D Gaussian Splatting

Train 3D Gaussian Splatting with Clad-generated CUDA derivatives inside the official Graphdeco trainer. The generated kernels run within 0.3% per step of the same kernels written by hand.

Clad differentiates spherical-harmonic colour, view direction, and the rotation, scale, and opacity activations. Those adjoints go straight into dense Adam, so the training step never writes out the SH gradient buffer. Geometry, sorting, pixel blending, and their derivatives stay in the native rasterizer. Every Gaussian parameter still trains.

## Results

All numbers are from Tanks and Temples scenes on an RTX 3080 Ti. Details are in [results/README.md](results/README.md).

- **Generated versus handwritten derivatives.** With the same schedule, handwritten SH and activation derivatives were 0.11–0.30% faster per step on four fixed-count Train and Truck checks. Two runs of the identical handwritten kernels differed by up to 0.23%, so the cost of using Clad is about the size of the run-to-run variation.
- **Training time against the original renderer.** The fused schedule (direct coefficient reads, Adam consumption, and full-warp accumulation in pixel backward) cut training-loop time from 1037.625 s to 696.802 s, **32.85%**, over 30k iterations with seed 42. This speedup comes from the schedule, not from automatic differentiation. Clad supplies the derivatives that the fused kernels need. The handwritten control uses the same schedule.
- **Quality.** Mean held-out PSNR went from 22.256 dB to 22.276 dB over 38 views. Per-view changes range from -1.90 dB to +2.37 dB, so quality is comparable, not better.

## Docs

- [SETUP.md](SETUP.md): Python, external sources, Clad.
- [QUICKSTART.md](QUICKSTART.md): versions, build, checks, train/render/metrics.
- [INTEGRATION.md](INTEGRATION.md): layouts, Adam, activations, fallback.
- [results/README.md](results/README.md): the Train run.
- [reproduction/README.md](reproduction/README.md): how to run that comparison again.
- [controls/README.md](controls/README.md): the handwritten derivatives.
- [VALIDATION.md](VALIDATION.md): what was checked.
- [THIRD_PARTY.md](THIRD_PARTY.md).

A native build needs Linux, Python 3.11 or 3.12, PyTorch 2.5.1+cu121, CUDA 12.1, CUDA-enabled Clang, a built Clad plugin, and the pinned renderer and trainer. Build outputs and datasets are not included.

## Layout

`SHGradientConsumer.cu` and `normalization.cu` are the differentiated functions. The root Python modules build them and connect them to the trainer. `WarpAccumulation.cuh` reduces a full warp before the atomic. `reproduction/runner/` is the worker that produced the Train timing. `controls/` is the handwritten schedule. `results/` is the measurement.

## License

Like Clad, this repository is under the GNU LGPL v3 ([License.txt](License.txt)). Files with an `SPDX-License-Identifier: Apache-2.0` line are under Apache 2.0 ([licenses/Apache-2.0.txt](licenses/Apache-2.0.txt)). The renderer, trainer, and datasets keep their own licenses. The Graphdeco trainer and renderer are for non-commercial research use only. See [THIRD_PARTY.md](THIRD_PARTY.md).
