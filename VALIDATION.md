# Validation

What was checked, and with which toolchain. These checks rebuild the code and exercise it on small fixtures. They do not repeat the 30k Train timing in [results/README.md](results/README.md).

## Builds and the dataset-free checker

Fresh direct-input/warp adapter build and activation link, using Clang
20.1.6, unmodified pinned Clad, GCC 12, CUDA 12.1, Python 3.12.3, and Torch
2.5.1+cu121. Both external object paths contained spaces. The required-feature
checker passed native SH/Adam comparisons, 424 packed and strided tensor
comparisons, 181,864 quaternion component comparisons, 60 combined activation
cases, and 105,480 parameter and moment comparisons. Lifecycle checks covered
17 activation fallbacks and nine rejected unsafe backwards. SH ownership
checks reject post-forward state changes, overlapping
parameters/moments/counters, and bad counter layouts. Missing direct-input
and activation features were rejected. `run.py --help` loaded every supported
option and did not train.

The checker also rejects nine post-forward SH ownership changes and thirteen
unsupported or overlapping state layouts. A four-iteration public-trainer
fixture consumed SH and activations on iterations 1 and 3, fell back while
saving on iteration 2, and skipped the update on iteration 4. Offline receipt
tests reject wrong policies, changed modules or manifests, and missing
receipts. Matching native/candidate preparation and a no-training preview
pass. An optimized module labeled native, and a mismatched toolchain, are
rejected before paired training. These guards were added after the Train
timing was recorded.

## Raster comparison and smoke test

The original-native versus warp raster comparison passed (eight fixtures, 64
gradient tensors), as did the two post-forward activation gradient and freeze
rejections. A seven-iteration fixture plus a three-iteration resume passed
topology, reset, and fallback checks. Rendering a saved model left all six
Gaussian parameter tensors unchanged, and PSNR/SSIM/LPIPS came back finite.
Missing and empty evaluations exited nonzero. That image test reused fixture
views. An eval-enabled prune-everything attempt hit the native empty-scene
backward limit described in [INTEGRATION.md](INTEGRATION.md).

The standalone `SHGradientConsumer.cu` self-check (656,892 SH comparisons,
3,375 warp checks, including 3D thread blocks) and the handwritten
`controls/ManualSH.cu` self-check pass.

## Clean environment, 2026-10-27

A new Python 3.12 virtualenv installed Torch `2.5.1+cu121`, the auxiliary
requirements, and torchvision `0.20.1+cu121`. The trainer and its rasterizer,
KNN, and fused-SSIM submodules were cloned fresh. Interface hashes matched
`inputs.json`. KNN and fused-SSIM were compiled in that environment. Fresh
Clad objects and separate original-native and candidate adapters were built.
The installed LLVM/Clang 20.1.6 and the already built Clad plugin were reused.

The full dataset-free GPU checker (eight native/warp raster fixtures, 64 gradient tensors), and the synthetic seven-iteration plus three-resume smoke test passed. Rendering left the saved model unchanged. Metrics completed after VGG and LPIPS weights landed in an empty Torch cache. Missing and empty metric inputs exited nonzero. One RTX 3080 Ti.
