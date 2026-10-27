# Full-warp accumulation: full-training comparison

Aim, declared before launch: time one full Train run of the Clad SH and activation consumer with full-warp pixel accumulation against the original native renderer, and compare held-out PSNR.

One fresh Train scene, seed 42, up to 30,000 iterations per trajectory.
Both sides start from the same original COLMAP/SfM initialization.
Same dense fused Adam, native learning-rate, SH-degree, and topology
schedules, loss, SSIM, resolution, and camera sampling. The shared persistent
BLAKE2b/Box-Muller split sampler stays, so split randomness does not move
when point counts diverge. The trainer is instrumented so both sides share that sampler.

Baseline: the original native renderer binary, with ordinary PyTorch SH
gradients, activations, and Adam. Candidate: the Clad SH consumer with direct
DC/rest inputs and guarded rotation, scale, and opacity VJP/Adam consumption.
Generated quaternion backward stays on for fallback steps. Native sorting and
geometry stay. On Clad steps, raster pixel adjoints use the full-warp
reduction. Sparse warps and updates between warps still use atomics. Native
mode loads the original baseline binary. All 59 raw components train in both
modes. Wrong routing is rejected.

Before the long run, the same worker and runtime drive the training binary
through one 16-update fixture pair, checkpointed at stages 8 and 16. That is
a functional preflight. Full training then has three endpoint pairs: 7k, 15k,
and 30k. The middle stage reverses mode order. Each stage resumes the same
trajectory, including camera, sampling, exposure, optimizer, and RNG sidecar.
No checkpoint probes, extra seeds or scenes, automatic retries, or silent
replacement.

Report held-out PSNR for all 38 views, plus ten training views. Stop on nonfinite state or loss, or a worker error.

Report loop wall time, CUDA-event time, full worker-process wall time,
evaluation and checkpoint cost, and peak allocated memory separately. Sum the
three training stages per mode. Leave out shared initialization and
orchestration pauses, and report those separately. Keep every iteration's
loss, camera, Gaussian count, and timing. Counts can diverge under independent
densification, so full-training time is the time of those two trajectories.
One scene, one seed, one GPU.

The CUDA libraries are the checked builds, byte for byte. Routing and all six
optimizer counters are checked. Activation consumption is recorded every
iteration and kept across stages. The worker change for this run is accounting
for the three consumed activation groups. No clipping, loss, threshold, or
sampler change.

Feature under test: warp-raster-accumulation at
`9b4619dbf9db5404429f2dd9e72d3b882999feb9`. Full-warp aggregation is the usual
register reduction. Clad does not infer it.
