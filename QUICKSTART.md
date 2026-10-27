# Build and run

Build the Clad SH and activation objects, link them into the Graphdeco trainer, and check that training applies those derivatives in dense Adam. The Train timing this integration was measured with is in [results/README.md](results/README.md).

No dataset, checkpoint, compiled extension, or external repo is included. Build, then run the dataset-free check. Full training is the launcher below.

## Requirements

[SETUP.md](SETUP.md) covers the environment, checkouts, and the Clad build.
Linux, Python 3.11 or 3.12, PyTorch `2.5.1+cu121`, CUDA 12.1, CUDA-enabled
Clang, and a Clad plugin built against that Clang. Packaging used Clang/LLVM
20.1.6 and unmodified Clad `8c3ac4379dd3a7a0a068581873a00ce50e22fd78`. The
native CUDA extension also needs a host compiler such as GCC 12, plus Ninja,
`patch`, and the trainer's Python dependencies.

External checkouts, submodules initialized:

- Graphdeco `gaussian-splatting` at `54c035f7834b564019656c3e3fcc3646292f727d`.
- `diff-gaussian-rasterization` at `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`,
  including its GLM submodule.

Install the trainer's pinned `simple-knn` and `fused-ssim` into the same
environment. `inputs.json` checks the trainer and native interfaces by hash.
Attribution is in [THIRD_PARTY.md](THIRD_PARTY.md).

## Build

Set paths for your machine. This repository does not have to sit next to the
Clad source.

```sh
PKG=.
CLAD_CLANG=/path/to/clang++
CLAD_SOURCE=/path/to/clad
CLAD_BUILD=/path/to/clad/build
CUDA_ROOT=/usr/local/cuda-12.1
TRAINER=/path/to/gaussian-splatting
NATIVE=/path/to/diff-gaussian-rasterization
OBJECTS=/path/to/new-clad-objects
CANDIDATE=/path/to/new-native-clad-build

python "$PKG/build_objects.py" --clang "$CLAD_CLANG" \
  --clad-source "$CLAD_SOURCE" --clad-build "$CLAD_BUILD" \
  --cuda-home "$CUDA_ROOT" --arch sm_86 --output "$OBJECTS"
python "$PKG/build.py" --native-source "$NATIVE" --trainer-source "$TRAINER" \
  --consumer-object "$OBJECTS/sh-consumer.o" \
  --normalization-object "$OBJECTS/normalization.o" \
  --build-dir "$CANDIDATE" --cuda-home "$CUDA_ROOT" \
  --cxx /usr/bin/g++-12 --arch 8.6 --direct-inputs --warp-accumulation
python "$PKG/check.py" --build-dir "$CANDIDATE" \
  --require-direct-inputs --require-activations
```

Use fresh output directories. `build_objects.py --dry-run` prints commands and
input hashes and does not compile. A real run writes `objects.json` with the
commands, toolchain, inputs, and object hashes. The adapter builder leaves the
external checkouts alone and records sources and libraries in its manifest.
`build.json` records direct-input and warp policy, host and NVCC versions,
flags, compiler environment, and hashes of the modules and the manifest.
Version-2 receipts cover the prepared trainer, Ninja files, and matching
object receipts when those exist. Launch and comparison preparation check
every manifest entry. Keep the recorded paths, and rebuild after edits. The
paired reproduction wants receipts on both builds. Turning consumption off
inside an optimized module is not the native control.

The dataset-free checker covers gradients, optimizer state, layouts, rejected
mutations, lifecycle and fallback, and the supported activations. It also
compares SH consumption with native renderer gradients followed by fused Adam,
including nonzero optimizer state. `--require-direct-inputs` and
`--require-activations` fail the run when those features are missing.
Contracts are in [INTEGRATION.md](INTEGRATION.md). CUDA objects can be
compiled without a GPU. This checker needs a GPU and the pinned Torch runtime.
The standalone SH example has its own finite-difference check.

## Smoke test

With the same dependencies, plus torchvision from [SETUP.md](SETUP.md):

```sh
python "$PKG/smoke.py" --build-dir "$CANDIDATE" --native-source "$NATIVE" \
  --output /path/to/new-smoke-check
```

This makes two 64×64 procedural images and 1,024 initial points, trains seven
iterations, and resumes three from checkpoint four. It checks consumption,
topology changes, opacity reset, save and final-step fallback, restored
counters, and saved-model rendering with parameters left unchanged. No
external images or dataset. `--metrics` also checks PSNR/SSIM/LPIPS and the
missing and empty input failures. That may download LPIPS and VGG weights the
first time. Without `--metrics`, rendering needs no weights.

The output directory must be new. Logs, commands, traces, checkpoints, and
`smoke.json` stay on failure. The rendered test view was also used while
training the fixture, so the numbers are a functional check. Checkpoint
resume here is not a bitwise trajectory match.

## Train

```sh
python "$PKG/run.py" --build-dir "$CANDIDATE" --native-source "$NATIVE" \
  --direct-inputs --fused-adam --activation-consume --quaternion-backward \
  --summary /path/to/new-routing.json -- \
  -s /path/to/scene -m /path/to/new-model --disable_viewer --clad_consume
```

The launcher keeps the pinned trainer's loop and topology schedule.
Unsupported steps use native handling. The summary counts consumption and
fallback. It is not a timer. `--trace` synchronizes and copies tensors for
small checks. Do not time a run that has it on.

Dropping `--clad_consume` turns consumption off. Warp aggregation is still a
build-time flag on every entry point in that module, including native
fallback. The original-native comparison needs a separate build without
`--direct-inputs` or `--warp-accumulation`, without the activation and
quaternion replacements, and with fused Adam and the other trainer settings
matched. The reproduction worker loads that module itself. See
[reproduction/README.md](reproduction/README.md).

## Render and evaluate

Install torchvision from [SETUP.md](SETUP.md). Use a model saved by the public
launcher. The tiny fixtures below are not quality benchmarks. Rendering loads
the saved PLY and exposure file and writes PNGs under
`train/ours_<iteration>/` and `test/ours_<iteration>/` in the model directory.
The source dataset and `cfg_args` have to still be there. Existing images and
metric files at those paths follow native overwrite behavior. Copy the model
if you need to keep an earlier evaluation.

```sh
MODEL=/path/to/model
python "$PKG/run.py" --mode render --build-dir "$CANDIDATE" \
  --native-source "$NATIVE" -- -m "$MODEL" --iteration 30000 --skip_train
python "$PKG/run.py" --mode metrics --build-dir "$CANDIDATE" \
  --native-source "$NATIVE" -- -m "$MODEL"
```

Train with `--eval` (after the launcher's `--`) to keep held-out cameras.
Pick an iteration the trainer actually saved. Omit `--skip_train` to export
training cameras too. A training-only scene has no `test` split, which the
native metrics command requires. Pass `-s /path/to/scene` if the dataset
moved. Consumption, optimizer, and trace options are rejected outside
training mode.

These modes call the pinned native tools. Rendering does not build an optimizer. Evaluation writes `results.json` and `per_view.json` with PSNR, SSIM, and LPIPS. Failures are re-raised, and empty or nonfinite metrics exit nonzero. LPIPS may download weights on first use.

## Pixel backward against original native

Build the baseline from [reproduction/README.md](reproduction/README.md), then:

```sh
python "$PKG/check.py" --build-dir "$CANDIDATE" --baseline-build "$BASELINE" \
  --require-direct-inputs --require-activations
```

The checker reads both build policies and toolchains, then loads the modules
in separate processes. Eight small fixtures cover full warps, partial edge
tiles, occlusion, and empty input, each with antialiasing off and on and with
nonzero RGB and depth adjoints. It compares native forward outputs and all
eight backward outputs, then runs the existing SH/Adam and activation checks.
Correctness only.

## Recorded run

[results/README.md](results/README.md) is the Train/seed-42 measurement: 32.85% less training-loop time, mean held-out PSNR 22.256 dB to 22.276 dB, handwritten control within 0.30% per step. [controls/README.md](controls/README.md) has those kernels. [reproduction/README.md](reproduction/README.md) reruns the pair. The checks above rebuild this repository and exercise it. They do not repeat the 30k timing.

What was checked, and with which toolchain, is in [VALIDATION.md](VALIDATION.md).
