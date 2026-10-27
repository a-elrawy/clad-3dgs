"""Compare fixed-topology native and Clad refinement from one checkpoint."""

from pathlib import Path
import argparse
import bisect
import hashlib
import json
import math
import random
import statistics
import sys
import time


MODES = ("native", "native-aa", "clad")


def host_clock(timer=time):
    # A raw Linux clock avoids rate adjustments during a comparison.
    if hasattr(timer, "CLOCK_MONOTONIC_RAW"):
        return "CLOCK_MONOTONIC_RAW", lambda: timer.clock_gettime_ns(timer.CLOCK_MONOTONIC_RAW)
    return "perf_counter", timer.perf_counter_ns


def schedule(views, quality_steps, quality_rounds, samples, warmups, order):
    # Whole camera cycles give every block the same first view.
    quality_steps = math.ceil(quality_steps / views) * views
    blocks = []
    for round_index in range(quality_rounds):
        for mode in MODES:
            blocks.append(dict(phase="quality", round=round_index, mode=mode,
                               steps=quality_steps))
    for round_index in range(warmups + samples):
        for offset in range(3):
            mode = order[(offset + round_index) % 3]
            blocks.append(dict(phase="warmup" if round_index < warmups else "timing",
                               round=round_index - warmups, mode=mode, steps=views))
    ends = []
    for block in blocks:
        ends.append((ends[-1] if ends else 0) + block["steps"])
    return blocks, ends


def compare_quality(reference, candidate, *, psnr_drop_db=0.02,
                    objective_increase_fraction=0.001):
    if (not math.isfinite(objective_increase_fraction)
            or objective_increase_fraction < 0):
        raise ValueError("Loss tolerance must be finite and nonnegative")
    if psnr_drop_db is not None and (
            not math.isfinite(psnr_drop_db) or psnr_drop_db < 0):
        raise ValueError("PSNR tolerance must be finite and nonnegative")
    if len(reference) != len(candidate):
        raise ValueError("Quality camera lists differ")
    comparisons = []
    for ref, got in zip(reference, candidate):
        if (ref["split"], ref["camera"]) != (got["split"], got["camera"]):
            raise ValueError("Quality camera order differs")
        psnr_delta = got["psnr"] - ref["psnr"]
        objective_delta = (got["objective"] - ref["objective"]) / max(
            abs(ref["objective"]), 1e-30
        )
        if not math.isfinite(psnr_delta) or not math.isfinite(objective_delta):
            raise ValueError("Nonfinite quality metrics")
        loss_passed = objective_delta <= objective_increase_fraction
        psnr_passed = psnr_delta >= -psnr_drop_db if psnr_drop_db is not None else None
        comparisons.append(dict(
            split=ref["split"], camera=ref["camera"],
            psnr_delta_db=psnr_delta, objective_delta=objective_delta,
            loss_passed=loss_passed, psnr_passed=psnr_passed,
            passed=loss_passed and psnr_passed is not False,
        ))
    return comparisons


def timing_summary(samples):
    medians = {
        mode: {clock: statistics.median(row[clock] for row in samples[mode])
               for clock in ("event_ms", "wall_ms")}
        for mode in MODES
    }
    comparisons = {}
    for clock in ("event_ms", "wall_ms"):
        baseline = medians["native"][clock]
        aa = 100 * (medians["native-aa"][clock] / baseline - 1)
        comparisons[clock] = dict(
            saving_percent=100 * (1 - medians["clad"][clock] / baseline),
            native_aa_percent=aa, reliable=abs(aa) <= 1,
        )
    return dict(medians_ms=medians, comparisons=comparisons,
                reliable=all(row["reliable"] for row in comparisons.values()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--views", type=int, default=10)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--quality-steps", type=int, default=100)
    parser.add_argument("--quality-rounds", type=int, default=3)
    parser.add_argument("--max-loss-increase-percent", type=float, default=0.1,
                        help="Maximum relative objective increase on each evaluated view")
    parser.add_argument("--max-psnr-drop-db", type=float, default=0.02)
    parser.add_argument("--psnr-report-only", action="store_true",
                        help="Report PSNR differences without using them as an acceptance gate")
    parser.add_argument("--quality-only", action="store_true")
    parser.add_argument("--order", default="native,native-aa,clad")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if min(args.views, args.samples, args.warmups,
           args.quality_steps, args.quality_rounds) <= 0:
        parser.error("views, samples, warmups and quality counts must be positive")
    if any(not math.isfinite(value) or value < 0 for value in
           (args.max_loss_increase_percent, args.max_psnr_drop_db)):
        parser.error("quality tolerances must be finite and nonnegative")
    quality_limits = dict(
        psnr_drop_db=None if args.psnr_report_only else args.max_psnr_drop_db,
        objective_increase_fraction=args.max_loss_increase_percent / 100,
    )
    order = args.order.split(",")
    if len(order) != 3 or set(order) != set(MODES):
        parser.error("order must contain native,native-aa,clad exactly once")
    build = args.build_dir.resolve()
    trainer = build / "trainer"
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file() or not (trainer / "train.py").is_file():
        parser.error("use an existing checkpoint and a prepared trainer build")
    if output.exists():
        parser.error("use a new output directory; existing results are preserved")

    import numpy as np
    import torch

    pins = json.loads((Path(__file__).parent / "inputs.json").read_text())
    if torch.__version__ != pins["torch"]:
        parser.error("use the PyTorch version pinned in inputs.json")
    sys.path.insert(0, str(build))
    import current_native_sh_consumer as extension

    sys.modules["diff_gaussian_rasterization._C"] = extension
    sys.path[:0] = [str(trainer), str(args.native_source.resolve())]
    import train
    import clad_training
    from arguments import ModelParams, OptimizationParams, PipelineParams
    from sh_consumer_adapter import DenseSHConsumer, StepPolicy
    import diff_gaussian_rasterization as native

    native_parser = argparse.ArgumentParser(description="Native dataset/options")
    lp = ModelParams(native_parser)
    op = OptimizationParams(native_parser)
    pp = PipelineParams(native_parser)
    rest = args.trainer_args
    if rest and rest[0] == "--":
        rest = rest[1:]
    options = native_parser.parse_args(rest)
    if options.model_path:
        parser.error("use --output instead of the native model_path option")
    if not options.source_path:
        parser.error("pass the native dataset source_path after --")
    if (options.train_test_exp or options.depths or options.random_background
            or options.optimizer_type != "default"
            or options.convert_SHs_python or options.compute_cov3D_python
            or options.debug):
        parser.error("benchmark requires dense Adam, native SH/covariance, "
                     "and no exposure, depth, random background or debug mode")
    if options.feature_lr != 0.0025:
        parser.error("the consumer requires feature_lr=0.0025")
    output.mkdir(parents=True)
    options.model_path = str(output / "trainer")
    dataset, opt, pipe = lp.extract(options), op.extract(options), pp.extract(options)
    opt.densify_until_iter = 0
    clad_training.TRACE = None
    train.TENSORBOARD_FOUND = False
    train.randint = lambda low, high: 0
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    clock_name, host_time = host_clock()

    report = dict(status="initializing", arguments=vars(args) | {"order": order},
                  native_options=vars(options), quality=[], quality_metrics=[],
                  host_clock=clock_name,
                  quality_limits=quality_limits, gaussian_counts=[],
                  samples={mode: [] for mode in MODES},
                  scope="Fixed-topology checkpoint refinement, explicit Clad SH "
                        "consumption versus native fused PyTorch Adam; "
                        "not full training or independent densification.")

    def save():
        (output / "results.json").write_text(
            json.dumps(report, indent=2, default=str, allow_nan=False) + "\n"
        )

    state = {}
    original_scene = train.Scene
    original_restore = train.GaussianModel.restore
    original_rate = train.GaussianModel.update_learning_rate
    original_begin = train.begin_step
    original_backward = train.after_backward
    original_after = train.after_step

    class BenchmarkScene(original_scene):
        def __init__(self, *positional, **kwargs):
            kwargs["shuffle"] = False
            super().__init__(*positional, **kwargs)
            cameras = self.getTrainCameras()
            count = min(args.views, len(cameras))
            if count == 0:
                raise ValueError("Dataset has no training cameras")
            indices = [i * (len(cameras) - 1) // (count - 1)
                       for i in range(count)] if count > 1 else [0]
            self.train_cameras[1.0] = [cameras[i] for i in indices]
            state["scene"] = self
            report["cameras"] = dict(
                selected_indices=indices,
                train=[c.image_name for c in self.getTrainCameras()],
                heldout=[c.image_name for c in self.getTestCameras()],
                note="Held out from this refinement, not necessarily unseen "
                     "by the supplied pretrained model.",
            )
            blocks, ends = schedule(count, args.quality_steps, args.quality_rounds,
                                    0 if args.quality_only else args.samples,
                                    0 if args.quality_only else args.warmups, order)
            state.update(blocks=blocks, ends=ends)

    def locate(iteration):
        step = iteration - state["start"] - 1
        if step >= state["ends"][-1]:
            return None, None, None
        index = bisect.bisect_right(state["ends"], step)
        offset = step - (state["ends"][index - 1] if index else 0)
        return state["blocks"][index], offset, index

    def restore(model, captured, training_options):
        original_restore(model, captured, training_options)
        names = [g["name"] for g in model.optimizer.param_groups]
        if len(names) != 6 or set(names) != {"xyz", "f_dc", "f_rest", "opacity", "scaling", "rotation"}:
            raise ValueError("Expected all six native Gaussian parameter groups")
        model.optimizer.defaults.update(fused=True, foreach=False)
        initial = []
        for group in model.optimizer.param_groups:
            group.update(fused=True, foreach=False)
            param = group["params"][0]
            moment = model.optimizer.state[param]
            if not moment:
                moment.update(step=torch.zeros((), device=param.device),
                              exp_avg=torch.zeros_like(param),
                              exp_avg_sq=torch.zeros_like(param))
            moment["step"] = moment["step"].to(device=param.device)
            if set(moment) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("Unsupported optimizer checkpoint state")
            for value in (param, moment["exp_avg"], moment["exp_avg_sq"]):
                if not bool(torch.isfinite(value).all()):
                    raise ValueError("Checkpoint has nonfinite parameters/moments")
            initial.append((param.detach().clone(),
                            {key: value.clone() for key, value in moment.items()}))
        probe = DenseSHConsumer(model.optimizer, native=native, extension=extension)
        if probe.bind(model._features_dc, model._features_rest, StepPolicy()) is None:
            raise ValueError("Consumer cannot run: " + probe.last_reason)
        state["initial"] = initial
        report.update(gaussians=len(model.get_xyz), degree=model.active_sh_degree,
                      initial_counters={g["name"]: int(m[1]["step"])
                                        for g, m in zip(model.optimizer.param_groups, initial)})

    def rate(model, iteration):
        _, offset, _ = locate(iteration)
        return original_rate(model, state["start"] + 1 + (offset or 0))

    def begin(model, iteration, options, dataset, testing, saving, enabled):
        block, offset, index = locate(iteration)
        if block is None:
            return original_begin(model, iteration, options, dataset, testing, saving, False)
        if offset == 0:
            state["gaussians_before"] = len(model.get_xyz)
            with torch.no_grad():
                for group, (initial, moments) in zip(model.optimizer.param_groups, state["initial"]):
                    param = group["params"][0]
                    param.copy_(initial)
                    for key, value in moments.items():
                        model.optimizer.state[param][key].copy_(value)
                model.optimizer.zero_grad(set_to_none=True)
            if block["phase"] != "quality":
                torch.cuda.synchronize()
                state["wall_start"] = host_time()
                state["event_start"].record()
        return original_begin(model, iteration, options, dataset, testing, saving,
                              block["mode"] == "clad")

    def backward(model, iteration, loss):
        original_backward(model, iteration, loss)
        block, _, _ = locate(iteration)
        if block is None:
            return
        owner = getattr(model, "_clad_consumer", None)
        consumed = bool(owner is not None and owner.active is not None and owner.active.consumed)
        if consumed != (block["mode"] == "clad"):
            raise RuntimeError("Unexpected consumer/native routing; comparison stopped")
        if block["phase"] == "quality" and not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite quality loss")

    @torch.no_grad()
    def metrics(model):
        rows = []
        scene = state["scene"]
        bg = torch.tensor([1, 1, 1] if dataset.white_background else [0, 0, 0],
                          device="cuda", dtype=torch.float32)
        for split, cameras in [("train", scene.getTrainCameras()),
                               ("heldout", scene.getTestCameras())]:
            for camera in cameras:
                image = train.render(camera, model, pipe, bg)["render"]
                if camera.alpha_mask is not None:
                    image = image * camera.alpha_mask.cuda()
                target = camera.original_image.cuda()
                mse = float((image.double() - target.double()).square().mean())
                objective = ((1 - opt.lambda_dssim) * train.l1_loss(image, target)
                             + opt.lambda_dssim * (1 - train.fused_ssim(image[None], target[None])))
                rows.append(dict(camera=camera.image_name, split=split,
                                 psnr=-10 * math.log10(max(mse, 1e-300)),
                                 objective=float(objective)))
        return rows

    def after(model, iteration):
        original_after(model, iteration)
        block, offset, index = locate(iteration)
        if block is None or offset + 1 != block["steps"]:
            return
        if block["phase"] != "quality":
            state["event_end"].record()
            state["event_end"].synchronize()
            wall = (host_time() - state["wall_start"]) / 1e6 / block["steps"]
            event = state["event_start"].elapsed_time(state["event_end"]) / block["steps"]
            if block["phase"] == "timing":
                report["samples"][block["mode"]].append(dict(event_ms=event, wall_ms=wall))
        # Counter and quality readbacks are outside every measured interval.
        counts = dict(phase=block["phase"], round=block["round"], mode=block["mode"],
                      before=state["gaussians_before"], after=len(model.get_xyz))
        report["gaussian_counts"].append(counts)
        if counts["before"] != report["gaussians"] or counts["after"] != report["gaussians"]:
            raise RuntimeError("Gaussian count changed during fixed-topology refinement")
        for group, (_, initial) in zip(model.optimizer.param_groups, state["initial"]):
            param = group["params"][0]
            moment = model.optimizer.state[param]
            if int(moment["step"]) != int(initial["step"]) + block["steps"]:
                raise RuntimeError("An Adam counter did not advance exactly once per update")
            if block["phase"] == "quality":
                if not all(bool(torch.isfinite(x).all()) for x in
                           [param, moment["exp_avg"], moment["exp_avg_sq"]]):
                    raise RuntimeError("Nonfinite quality parameters/moments")
        if block["phase"] == "quality":
            row = dict(round=block["round"], mode=block["mode"],
                       steps=block["steps"], metrics=metrics(model))
            report["quality_metrics"].append(row)
            if block["mode"] == "clad":
                native_rows = report["quality_metrics"][-3:]
                comparisons = {
                    r["mode"]: compare_quality(native_rows[0]["metrics"], r["metrics"],
                                              **quality_limits)
                    for r in native_rows[1:]
                }
                passed = all(row["passed"] for rows in comparisons.values() for row in rows)
                report["quality"].append(dict(round=block["round"], passed=passed,
                                              comparisons=comparisons))
                save()
                if not passed:
                    report["status"] = "quality_failed"
                    raise RuntimeError("Quality gate failed, including native A/A; timing stopped")
        print(f"Completed {block['phase']} block {index}: {block['mode']}", flush=True)

    train.Scene = BenchmarkScene
    train.GaussianModel.restore = restore
    train.GaussianModel.update_learning_rate = rate
    train.GaussianModel.oneupSHdegree = lambda model: None
    train.begin_step = begin
    train.after_backward = backward
    train.after_step = after
    try:
        report["source_sha256"] = {}
        paths = [Path(__file__).resolve(), checkpoint, Path(extension.__file__),
                 Path(train.__file__), Path(native.__file__),
                 trainer / "clad_training.py", trainer / "sh_consumer_adapter.py",
                 trainer / "gaussian_renderer/__init__.py"]
        for path in paths:
            with path.open("rb") as stream:
                report["source_sha256"][str(path)] = hashlib.file_digest(stream, "sha256").hexdigest()
        captured, start = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(start, int) or start < 0:
            raise ValueError("Checkpoint iteration must be a nonnegative integer")
        # Scene determines the selected camera count before the loop starts.
        # Assign iterations when restore runs, after Scene construction.
        state["start"] = start
        del captured
        original_benchmark_restore = train.GaussianModel.restore

        def restore_and_set_iterations(model, captured, training_options):
            original_benchmark_restore(model, captured, training_options)
            training_options.iterations = start + state["ends"][-1] + 1
            report["optimizer_options"] = vars(training_options).copy()

        train.GaussianModel.restore = restore_and_set_iterations
        if not train.FUSED_SSIM_AVAILABLE:
            raise RuntimeError("Install the pinned fused-SSIM extension for this comparison")
        state["event_start"] = torch.cuda.Event(enable_timing=True)
        state["event_end"] = torch.cuda.Event(enable_timing=True)
        report.update(status="running", torch=torch.__version__,
                      device=torch.cuda.get_device_name(), checkpoint_iteration=start,
                      schedule="Camera cycles, fixed SH degree, checkpoint learning-rate "
                               "schedule replayed per block; dense fused Adam; no topology, "
                               "evaluation/save I/O, depth, exposure or TensorBoard.")
        train.training(dataset, opt, pipe, [], [], [], str(checkpoint), -1, clad_consume=True)
        report["quality_passed"] = all(row["passed"] for row in report["quality"])
        if args.quality_only:
            report["status"] = "quality_passed"
        else:
            report["timing"] = timing_summary(report["samples"])
            report["status"] = "completed" if report["timing"]["reliable"] else "timing_unreliable"
    except Exception as error:
        if report["status"] not in ("quality_failed", "timing_unreliable"):
            report["status"] = "failed"
        report["error"] = str(error)
        raise
    finally:
        save()
    print(json.dumps({key: report[key] for key in ("status", "gaussians", "quality_passed")}))
    if report["status"] == "timing_unreliable":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
