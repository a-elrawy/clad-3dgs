# Reproduce the training comparison

Rerun the measurement in [../results/README.md](../results/README.md): original native renderer against the Clad consumer, Train, seed 42, stages at 7k, 15k, and 30k. Ordinary training uses the launcher in [../QUICKSTART.md](../QUICKSTART.md). Enter here through `run_pair.py`.

The wrappers only replace machine-specific paths. Worker math is the recorded worker. `runner/campaign.py` still has an older orchestration path. Both modes share the persistent split sampler. Loop time is the sum of the three stage loops in the instrumented trainer. Held-out quality is PSNR.

## Two builds

Build the candidate from `../QUICKSTART.md`. Build a separate baseline of the
same adapter without direct inputs or warp aggregation:

```sh
BASELINE=/path/to/new-native-control-build
python "$PKG/build.py" --native-source "$NATIVE" --trainer-source "$TRAINER" \
  --consumer-object "$OBJECTS/sh-consumer.o" --build-dir "$BASELINE" \
  --cuda-home "$CUDA_ROOT" --cxx /usr/bin/g++-12 --arch 8.6
```

Both builds must come from the current `build.py`. Preparation requires
`build.json` version 2 and the matching `manifest.json`. It checks every
listed file, the recorded module hashes, and matching toolchains and
architectures, including host and NVCC versions, compiler options, and the
recorded compiler environment. The native receipt must have direct inputs and
warp aggregation off. The candidate receipt must have both on, and must name
the activation module you pass in. Rebuild anything that has no receipt. Leave
the recorded source, object, and runtime paths in place through verification.
The receipts are copied into the campaign configuration.

Native mode calls the original entry points. It does not call the generated
SH or activation consumers. Candidate mode uses the candidate build, direct
inputs, guarded activation consumption, and generated quaternion backward on
eligible fallback steps. Both use dense fused Adam, the same fused SSIM, and
the same training schedules. Comparing consumption off and on inside one
warp-enabled binary is a different experiment.

## Prepare, then preview

Use the original COLMAP files and images. A pretrained Gaussian checkpoint is
the wrong start. The dataset is external.

```sh
CAMPAIGN=/path/to/new-training-pair
python "$PKG/reproduction/prepare.py" --source /path/to/Train \
  --trainer-source "$TRAINER" --native-source "$NATIVE" \
  --candidate-build "$CANDIDATE" --baseline-build "$BASELINE" \
  --normalization-build "$CANDIDATE/normalization" --output "$CAMPAIGN"
python "$PKG/reproduction/run_pair.py" --config "$CAMPAIGN/config.json"
```

Preparation checks the pinned interfaces, makes a fresh trainer copy and
COLMAP cache, copies the frozen workers, and records input, source, and
module hashes. The two builds must be different directories with different
binary hashes. Datasets, checkouts, and compiled modules are read-only
inputs. The frozen config uses absolute paths. Prepare again on another
machine. Use a fresh campaign directory. Preparation and preview do not
import Torch or touch the GPU.

Preview prints the stage order and `executes_training: false`. It writes no
execution record. To actually run the pair:

```sh
python "$PKG/reproduction/run_pair.py" --config "$CAMPAIGN/config.json" --run
```

This is optional and expensive. A failed campaign is not retried or resumed.
Keep the failure and use a new output directory for another attempt.
`EXECUTION.json` records worker completion, process time, camera and SH
schedule checks, and mean quality at each endpoint. The worker records
iteration counts, optimizer counters, losses, and Gaussian counts before and
after. Native and candidate checkpoints use the same sidecar protocol.

`fixture-plan.json` is the short functional plan. Pass it with `--plan` and a
compatible small fixture. Those results are not full-scene quality or timing.
Packaging validation prepared this plan and previewed it without running
updates.

## Reading a reproduction

Compare loop time, held-out PSNR, and the Gaussian counts with [../results/README.md](../results/README.md). Loop time sums the three stage loops and leaves out shared initialization and the gaps between workers. Worker-process time also includes loading, evaluation, and checkpoints. Counts can diverge during densification. Report them with the held-out views.

The 32.85% is the run at Clad `9b4619dbf9db5404429f2dd9e72d3b882999feb9` on that GPU. The consumer and activation math here match that feature. The warp helper also handles 3D block lane ids. The measured renderer uses 2D blocks. [PROTOCOL.md](PROTOCOL.md) is the protocol that run followed.
