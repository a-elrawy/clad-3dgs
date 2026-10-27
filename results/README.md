# Recorded results

Train scene, seed 42, 30,000 iterations, RTX 3080 Ti. Clad SH and activation derivatives, direct DC/rest reads, dense Adam consumption, and full-warp pixel accumulation, against the original native renderer in the same instrumented trainer.

| Measurement | Native reference | Clad training path |
|---|---:|---:|
| Training-loop time | 1037.625 s | 696.802 s |
| Mean held-out PSNR | 22.256 dB | 22.276 dB |
| Final Gaussians | 1092900 | 1089567 |

Training-loop time dropped 32.85%. Mean held-out PSNR rose by 0.020 dB. Across the 38 held-out views, PSNR deltas run from -1.90 dB to +2.37 dB. The candidate finished with 0.305% fewer Gaussians.

Native geometry and pixel derivatives are the same on both sides. Both use dense fused Adam, fused SSIM, and the same learning-rate, SH-degree, topology, and camera schedules. Native mode loaded the original renderer binary. The time is the sum of the 7k, 15k, and 30k stage loops, after shared initialization. Densification diverged, so the two trajectories did different amounts of work.

The same schedule with handwritten derivatives was 0.11–0.30% faster per step on four fixed-count Train and Truck checks. Two runs of the identical handwritten kernels differed by up to 0.23% in the same checks, so the gap between Clad and the hand-written kernels is about the size of the run-to-run variation. The Truck checks had two held-out views.

Files:

- `timing.json`: loop times for the Train comparison, and per-step medians for the handwritten control with its repeat run, by wall clock and CUDA events.
- `final-metrics.json`: per-view PSNR for both modes, ten training views and 38 held-out views.
- `held-out-views.csv`: per-view PSNR deltas at 7k, 15k, and 30k. `split` marks training views versus held-out views.
- `gaussian-counts.csv`: counts before and after every update, both trajectories.

Measured with Clad `9b4619dbf9db5404429f2dd9e72d3b882999feb9`. The launcher here is `1d4d0fc296afbfdac17ffa6f1f552a9eeaf31341` and adds fused Adam and routing counts on top of the same consumer, activation math, and native preparation. To run the pair again, use `../reproduction/README.md`.
