# Attribution

This integration comes from Clad and uses the same license, the GNU LGPL v3
(`License.txt`). The LGPL builds on the GPL v3, whose text is in
`licenses/GPL-3.0.txt`. Files with an `SPDX-License-Identifier: Apache-2.0`
line are under Apache 2.0 (`licenses/Apache-2.0.txt`). Every other file is
under the LGPL v3.

The builder reads checkouts you supply and keeps their licenses in the copied
native and trainer trees. The Graphdeco trainer and renderer are under the
Gaussian-Splatting License, which allows non-commercial research and
evaluation use only. A trainer built here inherits that restriction.
`trainer.patch` and `reproduction/runner/trainer.patch` modify that trainer.

- `graphdeco-inria/gaussian-splatting` at
  `54c035f7834b564019656c3e3fcc3646292f727d`.
- `graphdeco-inria/diff-gaussian-rasterization` at
  `9c5c2028f6fbee2be239bc4c9421ff894fe4fbe0`, including its GLM submodule.
- The trainer's `simple-knn` and `fused-ssim` pins, and PyTorch, keep their
  own licenses. Build them in the trainer environment.

No dataset, pretrained checkpoint, external checkout, or compiled extension
is shipped here. Projection and pixel-gradient formulas stay attributed to
the external renderer. The warp aggregation is the usual full-warp reduction.
Clad does not infer it.
