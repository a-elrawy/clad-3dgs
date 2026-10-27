# SH consumption in the native trainer

How the Clad derivatives actually enter a Graphdeco training step. Build commands are in [QUICKSTART.md](QUICKSTART.md).

[`SHGradientConsumer.cu`](SHGradientConsumer.cu) replaces SH backward. The generated local VJP is applied to the DC and rest coefficients inside dense Adam before the ordinary optimizer runs, so that step has no SH gradient tensor. The position contribution still reaches geometry differentiation. Sorting, rasterization, pixel backward, geometry derivatives, and raw activation gradients stay native. Pixel atomics stay.

The lane schedule and the Adam update are part of this integration. Clad supplies the local derivatives. The same fusion with handwritten derivatives is the control in [controls/README.md](controls/README.md).

## Dependencies and build

`inputs.json` pins the supported interfaces:

- `graphdeco-inria/gaussian-splatting` at
  `54c035f7834b564019656c3e3fcc3646292f727d`.
- Its rasterizer at `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`, including GLM.
- PyTorch `2.5.1+cu121`, CUDA 12.1, a host C++ compiler, Clad with
  CUDA-enabled Clang, and the trainer's Python dependencies.
- The trainer's pinned `simple-knn` and `fused-ssim`, built for that PyTorch.
  The renderer and trainer keep their own licenses.
- Ninja and `patch`.

Compile the consumer as a position-independent object. SM86:

```sh
clang++ -std=c++17 -O3 -fPIC -DSH_CONSUMER_NO_MAIN \
  --cuda-path=/usr/local/cuda-12.1 --cuda-gpu-arch=sm_86 \
  -I/path/to/clad/include -I/path/to/clad/build/include \
  -Xclang -load -Xclang /path/to/clad.so \
  -Xclang -add-plugin -Xclang clad \
  -Xclang -plugin-arg-clad -Xclang -enable-va \
  -Xclang -plugin-arg-clad -Xclang -fno-numerical-diff-fallback \
  -c SHGradientConsumer.cu -o /tmp/sh-consumer.o
```

Then the extension and a separate trainer copy:

```sh
python build.py \
  --native-source /path/to/diff-gaussian-rasterization \
  --trainer-source /path/to/gaussian-splatting \
  --consumer-object /tmp/sh-consumer.o \
  --build-dir /tmp/native-sh-consumer \
  --cuda-home /usr/local/cuda-12.1 --cxx /usr/bin/g++-12 --arch 8.6
```

The build directory must be empty. The builder checks the pinned hashes, adds
a separate backward entry, and patches the copied trainer. Both external
checkouts are left untouched. It records source, object, and library hashes.
The launcher checks the version-2 receipt and every manifest entry before it
loads anything. Keep the recorded inputs at those paths. Edits or missing
inputs mean a fresh build. No handwritten SH derivative is linked. The
original native forward and backward entries stay available for the baseline
and for fallback.

The wrapper keeps contiguous inputs for the generated helper. Camera centres
are often strided views of inverse matrices, and a raw pointer drops that
stride. Position, radius, and coefficient views are packed the same way.

Check without a dataset:

```sh
python check.py \
  --build-dir /tmp/native-sh-consumer
```

The regression compares parameters, moments, and returned adjoints for packed
inputs and equivalent strided views, degrees 0 through 3. The fixture is two
Gaussians and one pixel, so the comparisons are deterministic.

## Direct SH inputs

`--direct-inputs` on the build compiles a separate copy of the pinned native
sources with readers for the existing contiguous DC and rest tensors. Neither
checkout is edited. The extra native entries and the generated consumer then
read those tensors directly, with no concatenated coefficient snapshot. The
packed entries stay.

### Warp accumulation

`--warp-accumulation` together with `--direct-inputs` combines pixel adjoints
before the global update. The copied native backward groups participating
lanes by Gaussian id. When all 32 lanes share the id, each gradient component
is reduced in registers and one lane does the atomic. Partial warps and mixed
ids keep the original per-lane atomics. Inverse-depth contributions use the
same path when they are present. Forward rendering, the local gradient
formulas, the trainable parameters, and the optimizer updates are the same.
Floating-point accumulation order can change.

The flag applies to the whole copied renderer module, including native
fallback entries. A native atomic control is a build without the flag. The
pinned checkout is never edited.

`WarpAccumulation.cuh` also accepts private Clad adjoints. The caller supplies
a key for the destination at each update site and keeps every participating
lane in the same collective. The helper is the usual full-warp reduction
described by [DISTWAR](https://arxiv.org/abs/2401.05345). Atomics between
warps stay. The standalone SH demo checks the helper with generated adjoints,
different keys, divergence, partial warps, and nonzero initial destinations.

Pass `--direct-inputs` before `--` in the launcher to use this path on
eligible consumed steps. It still needs `--clad_consume` and the fused Adam
setup below. `--fused-adam` selects that optimizer. Evaluation and
unsupported steps use the packed native fallback.

Custom callers can construct `DenseSHConsumer(..., direct_inputs=True)`.
Each lane finishes its coefficient reads before it writes its own values.
The position contribution is reduced from those saved local adjoints. DC,
rest, and their four moment tensors must be disjoint contiguous CUDA float32
buffers. The C++ entry rejects overlapping storage. The single-backward,
exclusive-use, and first-order rules still apply. There is no cached
coefficient copy to drop after an optimizer or topology update.

The same build exposes `render_direct_native` for a matched native
derivative and layout. It materializes coefficient gradients and leaves the
update to the ordinary optimizer. Direct storage is available to both paths.
Neither path removes pixel atomics.

When the entries exist, `check.py` adds direct and strided comparisons and an
aliased-moment rejection. The CUDA consumer demo also checks direct reads
with nonzero moments, inactive degrees, and invisible or clamped Gaussians.
Deleting a packed tensor is not, by itself, a peak-memory or step-time
result. Measure those separately.

## Quaternion activation backward

`normalization.cu` generates a private four-component quaternion adjoint.
Compile it like the consumer, with `normalization.cu` in and
`/tmp/quaternion.o` out. Drop `-DSH_CONSUMER_NO_MAIN` and add
`-ffp-contract=off`. Pass `--normalization-object /tmp/quaternion.o` to
`build.py`, then `--quaternion-backward` before `--` in `run.py`.

Forward keeps `torch.nn.functional.normalize` on dimension 1 with epsilon
`1e-12`, and saves the norm for the backward clamp. Only the first-order
CUDA float32 `[N, 4]` backward is replaced. Other dtypes and devices stay on
the PyTorch path. The generated pieces handle the epsilon clamp on their own,
so the singular square-root derivative at zero is not evaluated. One thread
owns one output quaternion. This op does not touch optimizer state or
parameters. Native geometry backward and Adam stay. It can be enabled without
SH consumption or direct SH inputs.

With the extension present, `check.py` also checks native forward equality,
condition-scaled gradient error, finite differences, empty and partial
batches, strided inputs and seeds, non-default streams, and clamp-boundary
cases with several nonzero components.

The launcher installs the activation when it builds the copied trainer's
model. Checkpoint restore and topology changes keep using the current raw
rotation tensor. There is no saved parameter-pointer cache. Higher-order
backward is rejected on the generated path. The C++ wrapper packs strided
inputs and seeds and launches on the current CUDA stream.

## Combined activation consumption

The same extension exposes
`consume_activations(parameters, seeds, saved, first, second, rates, bias1, bias2)`.
Each list has three entries: rotation, scaling, opacity. Parameters, seeds,
and moments have shapes `[N, 4]`, `[N, 3]`, and `[N, 1]`. Saved values are
the native forward quaternion norm `[N, 1]`, exponential scale `[N, 3]`, and
sigmoid opacity `[N, 1]`. `bias2` holds the square roots of the second-moment
bias corrections. Everything is CUDA float32. Writable buffers are contiguous
and disjoint from each other and from every read-only input. Seeds and saved
activations may be strided. The wrapper packs those.

One thread computes the three activation VJPs and applies their eight raw
parameter gradients in dense Adam, betas `0.9` / `0.999`, epsilon `1e-15`.
For the scalar activations, Clad differentiates a local perturbation at
`delta = 0`: `scale * exp(delta)` and
`opacity / (opacity + (1 - opacity) * exp(-delta))`. The saved native outputs
stay constant during that differentiation, so the original exponentials are
not recomputed and rounded opacity saturation at 0 and 1 holds. Quaternion
differentiation keeps the saved native norm and the separate clamp piece. No
handwritten scalar VJP is linked.

Build with the combined `normalization.cu` object and pass
`--activation-consume` before `--` in `run.py`, plus `--clad_consume` in the
trainer arguments. Eligible steps then use the generated activation VJPs and
their updates. The flag can sit next to `--direct-inputs` and
`--quaternion-backward`. The quaternion flag controls standalone quaternion
backward on fallback steps. The launcher requires the matching extension.
`--fused-adam` selects the supported optimizer after setup and restore.
Default unfused Adam keeps native updates even if consumption was requested.

The copied renderer binds all three activations once per render, with an
explicit exclusive-use check. Evaluation, save, topology change, opacity
reset, and skipped-update iterations stay on native backward. Ordinary
eligible steps create missing dense Adam state at zero, read each group's
counter, and bind the current parameters and moments. Unsupported optimizer
settings, frozen or accumulated gradients, bad layouts or counters, or
overlapping writable storage take the native path before any activation
update. After restore, append, prune, or reset, the next frame binds the new
objects. The end-of-step hook releases the frame. Nothing is cached past the
step.

Custom callers use `DenseActivationConsumer.bind(model, policy, exclusive_use=True)`
from `activation.py`. It returns rotation, scale, and opacity outputs, or
`None` when the step falls back. `gradient_processing=True` or
`higher_order=True` requests fallback for those cases. The policy fields
cover accumulation and scaled gradients. Call `end_step()` after the ordinary
optimizer. Extra raw-parameter losses need the native path: exclusive use is
the caller's promise. Higher-order requests, a second backward, gradients
added or parameters frozen after forward, or other parameter or state changes
between forward and backward raise before any activation write.

The low-level C++ entry does not bump optimizer counters or PyTorch version
counters. The Python frame advances all three counters and the parameter and
moment versions once, and leaves the consumed leaf gradients absent so
ordinary Adam skips them. Direct users of the C++ entry have to do the same.
Consumption wants finite forward activations and one first-order backward.
The compiler does not infer ownership or fusion. Native geometry and pixel
atomics are unchanged.

`check.py` finds this entry on its own. It compares scalar activation
gradients and Adam updates with PyTorch, and checks the quaternion VJP
against native AD, with the conditioning bound, before it isolates optimizer
arithmetic. Cases include saturated opacity, zero and tiny quaternions, empty
and partial blocks, strided inputs, unequal counters, nonzero moments, and
aliased storage rejected before any write. Public binding checks cover policy
fallback, rejected unsafe backwards, fresh state, unequal counters, and
ordinary Adam skipping consumed groups. With tracing on, the lifecycle report
adds an `activation` record: the three counters, moment shapes, routing
reason, and a once-only update check. Those snapshots synchronize. Use them
on small fixtures.

## Run

Trainer arguments go after `--`:

```sh
python run.py \
  --build-dir /tmp/native-sh-consumer \
  --native-source /path/to/diff-gaussian-rasterization --fused-adam -- \
  -s /path/to/scene -m /tmp/clad-training --disable_viewer --clad_consume
```

Omit `--clad_consume` for native gradients inside this build, and keep
`--fused-adam`. A `--warp-accumulation` build still aggregates inside its
native entries. The original-native comparison needs a separate
build without warp aggregation, direct inputs, or activation replacements.
See [QUICKSTART.md](QUICKSTART.md). Exposure stays in the original renderer
after rasterization. `--start_checkpoint`, scene saves, camera sampling, the
position learning-rate schedule, and progressive SH degree stay in the same
loop.

The pinned trainer does not pick fused Adam by default. `--fused-adam`
selects dense fused Gaussian Adam with `foreach=False` after setup and
checkpoint restore. Gaussian parameters, moment values, and step values are
kept. Restored step tensors move to the parameter device, which fused Adam
requires. The exposure optimizer is separate. Unsupported optimizer types,
Gaussian devices or dtypes, or counter dtypes raise before training. Leaving
the flag off keeps native behavior.

Consumption requires this backend. Otherwise the adapter uses native
backward. Use the same flag on the native control so the optimizer arithmetic
matches. Custom callers can call `optimizer_setup.fused_adam` after setup and
restore. Picking the optimizer does not make unsupported renderer or
lifecycle steps eligible.

`--summary /tmp/routing.json` before `--` writes a routing report: completed
training steps, consumed SH and activation steps, and fallback reasons.
Decisions are captured after backward, before evaluation can replace them,
and a step is counted only after its lifecycle hooks finish. The report reads
Python state. It does not snapshot tensors, read CUDA counters, synchronize,
or write a file each step. SH and activation counts are separate. Do not add
them together as if they were different training steps. A fallback count
includes the final skipped update.

An existing output path is refused. The file says `running` at launch and
`completed` or `failed` at exit, with partial counts if the failure was
handled. A hard kill can leave `running`. After these Python hooks change,
prepare a fresh trainer copy. The compiled kernels can be reused. The summary
is routing, not speed or image quality.

`--trace /tmp/lifecycle.json` is for a small functional fixture. It turns on
parameter snapshots and synchronized readback. It checks that consumed SH
state advances once, that the following Adam call skips those groups, and
that fallback and final steps do not update SH early. The trace records point
counts, SH degree, moment shapes, loss, and counters. Do not time a traced
run. Snapshots get large on a full model. An existing trace path is rejected.

## Lifecycle

The trainer reads parameters, changes topology, and may reset opacity between
backward and Adam. It also skips Adam on the last iteration. Applying the SH
update inside backward on every step would change those operations.

`official_step_policy` picks native backward for evaluation, save,
densification, opacity reset, and skipped-update iterations. Ordinary
consumed steps bind the current DC and rest parameters and the actual
optimizer moments. They advance both Adam counters once and leave the SH leaf
gradients absent, so the following Adam call skips them. Parameter and moment
version counters advance once after the consumed update. Each frame binds
state again after a checkpoint restore, append, or prune. Overlapping SH
parameters, moments, or counters, counters that are not device float32, and
counters outside the integers below `2**24` fall back to native. Changes to
parameters, optimizer groups, options, state, moments, or counters after
binding raise before the consumed update. The merged forward coefficients
stay a fixed snapshot for the geometry and SH position VJP.

Supported SH optimizer settings: DC and rest rates `0.0025` and `0.000125`,
betas `0.9` and `0.999`, epsilon `1e-15`, dense fused Adam, no weight decay,
`foreach` off. That matches the consumer's update. Other optimizer settings,
tensor or moment layouts, and the precomputed-colour or covariance paths use
native backward. Position-rate scheduling is independent of these fixed SH
rates. The rasterizer requires the default CUDA stream.

Consumption is one unscaled backward, exclusive SH use, and no gradient
processing after backward. Extra SH losses, accumulation, clipping, AMP, and
distributed synchronization stay on the native path. Custom callers set the
adapter policy for those cases. Higher-order backward is rejected. There is
no sparse Adam, CUDA graph capture, or new checkpoint content. The pinned
trainer's checkpoint format is unchanged, and it does not store the
camera-sampling RNG or the exposure optimizer. A resumed run is not promised
to match an uninterrupted one bitwise.

The generated CUDA demo checks its own derivatives and optimizer and runs without PyTorch. The trainer integration needs the dependencies above. The full Train result is in [results/README.md](results/README.md).

The pinned renderer cannot backpropagate a packed SH tensor after every Gaussian has been pruned. Empty-scene backward returns `[0, 0, 3]` where the input coefficient shape is `[0, 16, 3]`. Fallback steps keep that behavior. Use a scene that still has Gaussians.
