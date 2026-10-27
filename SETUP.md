# Setup

Environment for the native Clad integration in [README.md](README.md).

Linux, an NVIDIA GPU and driver, the CUDA 12.1 toolkit (`nvcc` included),
GCC/G++ 12, Git, CMake, Ninja, and `patch`. Native builds were checked with
Python 3.12.3. Use Python 3.11 or 3.12 for the pinned Torch wheel. The offline
checker runs on Python 3.11+ and needs none of the GPU stack.

The trainer's `environment.yml` is Python 3.7 / Torch 1.12 / CUDA 11.6. Use
the environment below for this integration. Commands assume Bash and this
repository as the working directory. Pick fresh absolute paths for the
checkouts and the virtualenv.

## Python and CUDA

```sh
python3.12 -m venv /path/to/clad-3dgs-venv
source /path/to/clad-3dgs-venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt

export CUDA_HOME=/usr/local/cuda-12.1
export PATH="$CUDA_HOME/bin:$PATH"
export CXX=/usr/bin/g++-12
export CC=/usr/bin/gcc-12
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
```

The Torch install follows the official
[PyTorch 2.5.1 instructions](https://pytorch.org/get-started/previous-versions/#v251).
The wheel has the runtime libraries. Compiling these extensions still needs
the CUDA toolkit. Set the native arch (`8.6`) and the Clang arch (`sm_86` in
the quickstart) for your GPU. The recorded timings are SM86 / RTX 3080 Ti.

A clean Python 3.12 environment completed these steps on 2026-10-27, including
fresh KNN and fused-SSIM builds and the optional torchvision wheel. Validation
scope is in [VALIDATION.md](VALIDATION.md). The Clang/Clad install was reused.

`requirements.txt` is the auxiliary runtime and build set, not a lockfile of
the old benchmark environment. Torch and the native source interfaces are
pinned. Newer auxiliary packages do not reproduce the old binaries.

## Renderer and trainer

```sh
TRAINER=/path/to/gaussian-splatting

git clone https://github.com/graphdeco-inria/gaussian-splatting.git "$TRAINER"
git -C "$TRAINER" checkout --detach 54c035f7834b564019656c3e3fcc3646292f727d
git -C "$TRAINER" submodule update --init --recursive \
  submodules/diff-gaussian-rasterization submodules/simple-knn submodules/fused-ssim
NATIVE="$TRAINER/submodules/diff-gaussian-rasterization"
git -C "$NATIVE" checkout --detach 9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0
git -C "$NATIVE" submodule update --init --recursive

python -m pip install --no-build-isolation "$TRAINER/submodules/simple-knn"
python -m pip install --no-build-isolation "$TRAINER/submodules/fused-ssim"
```

The adapter loads the renderer Python package from `--native-source` and
supplies its own compiled module. You do not need a separate rasterizer wheel.
The trainer still needs the two extensions above. The SIBR viewer and sparse
Adam are unused on the dense path.

Check the environment before compiling:

```sh
python - <<'PY'
import torch
import cv2
import numpy
import PIL
import plyfile
from simple_knn._C import distCUDA2
from fused_ssim import fused_ssim
assert torch.__version__ == "2.5.1+cu121", torch.__version__
assert torch.cuda.is_available(), "A usable NVIDIA CUDA device is required"
print(torch.__version__, torch.cuda.get_device_name())
PY
```

## Saved-model rendering and evaluation

Training itself does not need torchvision. PNG export and PSNR/SSIM/LPIPS do.
Install the matching wheel:

```sh
python -m pip install torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

The builder copies the pinned trainer's `render.py`, `metrics.py`, and
`lpipsPyTorch` into the prepared trainer directory. LPIPS uses pretrained VGG
weights and calibrated LPIPS weights. The first evaluation may download them
into the PyTorch cache. Rendering alone does not. Commands are in
[QUICKSTART.md](QUICKSTART.md).

## Clang and Clad

Use a CUDA-enabled LLVM/Clang 20.1.6 build with the NVPTX backend, its
development libraries and CMake files, and a Clad plugin built against that
same LLVM. A system Clang plus another LLVM's plugin will not work. Building
LLVM is covered in the
[pinned Clad install guide](https://github.com/vgvassilev/clad/blob/8c3ac4379dd3a7a0a068581873a00ce50e22fd78/README.md#how-to-install).
Include `NVPTX` in `LLVM_TARGETS_TO_BUILD`.

With that LLVM in place:

```sh
LLVM_ROOT=/path/to/llvm20-build-or-install
CLAD_SOURCE=/path/to/clad
CLAD_BUILD=/path/to/new-clad-build
python -m pip install lit

git clone https://github.com/vgvassilev/clad.git "$CLAD_SOURCE"
git -C "$CLAD_SOURCE" checkout --detach 8c3ac4379dd3a7a0a068581873a00ce50e22fd78
cmake -S "$CLAD_SOURCE" -B "$CLAD_BUILD" -G Ninja \
  -DLLVM_DIR="$LLVM_ROOT/lib/cmake/llvm" \
  -DClang_DIR="$LLVM_ROOT/lib/cmake/clang" \
  -DLLVM_EXTERNAL_LIT="$(command -v lit)" -DCMAKE_BUILD_TYPE=Release
cmake --build "$CLAD_BUILD" --target clad --parallel 2
CLAD_CLANG="$LLVM_ROOT/bin/clang++"
```

The object builder looks for `$CLAD_BUILD/lib/clad.so`. Pass `--plugin` if the
plugin landed elsewhere. The native integration does not need deferred
template-identity or other experimental compiler branches.

Next: [QUICKSTART.md](QUICKSTART.md), to compile the two objects, build the
adapter, and run the feature checker. Keep build directories outside the
dependency source trees. Use fresh output directories. Keep a failed build's
logs.

## Scene input and outputs

The launcher takes the pinned trainer's datasets and arguments. The recorded
COLMAP / Tanks and Temples Train comparison starts from the original scene:
`images/` and `sparse/0/{cameras,images,points3D}.bin`. A fitted Gaussian
checkpoint is the wrong initialization. The dataset stays external and keeps
its own terms. The reproduction runner fixes the held-out split for this Train scene.

Ordinary training writes the trainer's model, evaluation, and checkpoint
outputs under `-m`. The optional routing summary counts consumption and
fallback. It is not a timer. The measured paired protocol and its separate
original-native build are in [reproduction/README.md](reproduction/README.md).
