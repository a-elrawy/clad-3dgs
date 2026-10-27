# Handwritten control

The same training schedule as the Clad path, with the SH, direction, and activation derivatives written out by hand. Layouts and Adam consumption match. On the four fixed-count Train and Truck runs these kernels were 0.11–0.30% faster per step than the generated ones. Two runs of these same kernels differed by up to 0.23%. Data is in `../results/timing.json`.

`ManualSH.cu` changes the self-check header path for this package, and the comments are shorter. The math is the recorded control.

To build a fresh control, compile both files with the same Clang, architecture,
optimization, and floating-point flags as the generated objects, and omit every
Clad plugin flag:

```sh
"$CLAD_CLANG" --driver-mode=g++ -std=c++17 -O3 -fPIC \
  --cuda-path="$CUDA_ROOT" --cuda-gpu-arch=sm_86 -DSH_CONSUMER_NO_MAIN \
  -c "$PKG/controls/ManualSH.cu" -o "$OBJECTS/manual-sh.o"
"$CLAD_CLANG" --driver-mode=g++ -std=c++17 -O3 -fPIC -ffp-contract=off \
  --cuda-path="$CUDA_ROOT" --cuda-gpu-arch=sm_86 \
  -c "$PKG/controls/ManualActivations.cu" -o "$OBJECTS/manual-activations.o"
```

Pass the objects to the same `build.py` entry, use a separate build directory,
and keep direct inputs, warp aggregation, optimizer, activation policy,
scene, checkpoint, and counters the same. `check.py` accepts that extension.
The runs are in `../results/README.md`.
