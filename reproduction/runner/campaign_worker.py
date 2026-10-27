"""One continuation of a paired native training trajectory."""

from pathlib import Path
import argparse
import gc
import json
import math
import random
import sys
import weakref

from benchmark import host_clock
from sampling import Lineage, SplitSampler
from training_state import CameraOrder, load_sidecar, restore_rng, save_bundle
from training_diagnostics import write_failure


class StageComplete(Exception):
    pass


def runtime(config, mode):
    import numpy as np
    import torch

    if torch.__version__ != config["torch_version"]:
        raise ValueError("Use the pinned Torch version")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    sys.path.insert(0, config["native_build_dir"] if mode in ("initial", "native") else config["build_dir"])
    import current_native_sh_consumer as extension

    sys.modules["diff_gaussian_rasterization._C"] = extension
    sys.path[:0] = [config["trainer_dir"], config["native_source"]]
    import train
    from arguments import ModelParams, OptimizationParams, PipelineParams
    import clad_training
    clad_training.DIRECT_INPUTS = mode == "clad"
    if mode == "clad":
        sys.path.insert(0, config["normalization_build_dir"])
        import clad_quaternion
        from clad_normalization import enable_quaternion_backward
        enable_quaternion_backward(train.GaussianModel, clad_quaternion)
        clad_training.ACTIVATION_EXTENSION = clad_quaternion

    if not train.FUSED_SSIM_AVAILABLE:
        raise RuntimeError("The pinned fused SSIM extension is required")
    original = train.GaussianModel.training_setup

    def setup(model, options):
        original(model, options)
        model.optimizer.defaults.update(fused=True, foreach=False)
        for group in model.optimizer.param_groups:
            group.update(fused=True, foreach=False)

    train.GaussianModel.training_setup = setup
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    parser = argparse.ArgumentParser()
    lp, op, pp = ModelParams(parser), OptimizationParams(parser), PipelineParams(parser)
    options = parser.parse_args([])
    options.source_path = config["source"]
    options.model_path = config["worker_output"]
    options.eval = True
    options.data_device = "cpu"
    options.iterations = config["plan"]["final_global_iteration"]
    for name, value in config["plan"].get("native_options", {}).items():
        if not hasattr(options, name):
            raise ValueError(f"Unknown native option: {name}")
        setattr(options, name, value)
    if (
        options.iterations != config["plan"]["final_global_iteration"]
        or options.optimizer_type != "default"
        or options.train_test_exp
        or options.depths
        or options.random_background
        or options.sh_degree != 3
        or options.convert_SHs_python
        or options.compute_cov3D_python
        or options.debug
    ):
        raise ValueError("Use the declared dense Adam/SH paired-training configuration")
    Path(config["worker_output"]).mkdir(exist_ok=False)
    return (
        train,
        lp.extract(options),
        op.extract(options),
        pp.extract(options),
        vars(options),
    )


class TrainingStage:
    def __init__(
        self, config, train, dataset, options, pipe, mode, endpoint, checkpoint
    ):
        import torch
        from sh_consumer_adapter import official_step_policy

        self.torch, self.train = torch, train
        self.config, self.dataset, self.options, self.pipe = (
            config,
            dataset,
            options,
            pipe,
        )
        self.mode, self.endpoint, self.checkpoint = mode, endpoint, checkpoint
        self.directory = Path(config["worker_output"])
        self.clock_name, self.clock = host_clock()
        self.policy_for = official_step_policy
        self.started = False
        self.finished = False
        self.rows = []
        self.losses = []
        self.expected = None
        self.consumed_total = 0
        self.consumed_stage = 0
        self.activation_consumed_total = 0
        self.activation_consumed_stage = 0
        self.peak_allocated = self.peak_reserved = 0
        self.event_counts = dict(topology=0, skipped=0, ordinary=0)
        self.header = None
        self.sampler = None
        self.phase_start = torch.cuda.Event(enable_timing=True)
        self.phase_end = torch.cuda.Event(enable_timing=True)
        if checkpoint is not None:
            self.header, self.lineage, self.order = load_sidecar(
                checkpoint, config["metadata"]
            )
            if (
                self.lineage.max_iteration != config["plan"]["final_global_iteration"]
                or self.lineage.seed != config["seed"]
            ):
                raise ValueError(
                    "Sampler horizon/seed differs from the frozen campaign"
                )
            auxiliary = self.header["auxiliary"]
            if auxiliary["trajectory"] not in (mode, "initial") or (
                auxiliary["trajectory"] == "initial" and self.header["iteration"] != 0
            ):
                raise ValueError("Cannot continue a different training trajectory")

    def counters(self):
        return {
            g["name"]: int(
                self.model.optimizer.state.get(g["params"][0], {}).get("step", 0)
            )
            for g in self.model.optimizer.param_groups
        }

    def bind(self, model, scene, iteration, options):
        self.model, self.scene = model, scene
        self.start_iteration = self.completed_iteration = iteration
        self.start_points = len(model.get_xyz)
        generated_rotation = model.rotation_activation.__module__ == "clad_normalization"
        if generated_rotation != (self.mode == "clad"):
            raise RuntimeError("Unexpected quaternion activation routing")
        if options.iterations != self.config["plan"]["final_global_iteration"]:
            raise ValueError(
                "Stage endpoint must not replace the global training horizon"
            )
        cameras = scene.getTrainCameras()
        self.by_name = {c.image_name: c for c in cameras}
        if len(self.by_name) != len(cameras):
            raise ValueError("Duplicate camera names")
        if self.header is None:
            if iteration != 0 or self.mode != "initial" or self.endpoint != 0:
                raise ValueError("Create the common initial state before training")
            if (
                self.config["expected_initial_points"] is not None
                and len(model.get_xyz) != self.config["expected_initial_points"]
            ):
                raise ValueError(
                    "Initialization differs from the original SfM point count"
                )
            self.lineage = Lineage(
                len(model.get_xyz), options.iterations, self.config["seed"]
            )
            self.order = CameraOrder([c.image_name for c in cameras])
            self.expected = self.counters()
        else:
            if iteration != self.header["iteration"] or set(self.order.names) != set(
                self.by_name
            ):
                raise ValueError("Native and experiment checkpoint state disagree")
            extra = self.header["auxiliary"]
            if model.exposure_mapping != extra["exposure_mapping"]:
                raise ValueError("Exposure/camera mapping changed during restoration")
            model._exposure.data.copy_(extra["exposure"].to(model._exposure.device))
            model.exposure_optimizer.load_state_dict(extra["exposure_optimizer"])
            self.expected = extra["expected_counters"].copy()
            self.consumed_total = extra["consumed_total"]
            self.activation_consumed_total = extra["activation_consumed_total"]
            if self.counters() != self.expected:
                raise RuntimeError("Native optimizer counters did not restore")
            restore_rng(self.header["rng"])
        self.sampler = SplitSampler(model, self.lineage)
        self.check_state(full=False)
        gc.collect()
        self.torch.cuda.synchronize()
        self.torch.cuda.empty_cache()
        self.setup_memory = dict(
            allocated=self.torch.cuda.memory_allocated(),
            reserved=self.torch.cuda.memory_reserved(),
        )
        if self.mode == "initial":
            self.finish(iteration)
        if not iteration < self.endpoint <= options.iterations:
            raise ValueError("Invalid continuation interval")

    def next_camera(self):
        name, index = self.order.next()
        self.camera_name = name
        return self.by_name[name], index

    def begin_step(self, model, iteration):
        if iteration != self.completed_iteration + 1:
            raise RuntimeError("Repeated or skipped training iteration")
        self.sampler.set_iteration(iteration)
        if not self.started:
            self.torch.cuda.synchronize()
            self.torch.cuda.reset_peak_memory_stats()
            self.phase_wall_start = self.clock()
            self.phase_start.record()
            self.started = True
        start = self.torch.cuda.Event(enable_timing=True)
        end = self.torch.cuda.Event(enable_timing=True)
        self.policy = self.policy_for(
            iteration, self.options, white_background=self.dataset.white_background
        )
        self.active = dict(
            iteration=iteration,
            start=start,
            end=end,
            count_before=len(model.get_xyz),
            reason=self.policy.reason(),
            degree_before=model.active_sh_degree,
            previous_structural_iteration=self.lineage.last_event,
            densify=(
                iteration < self.options.densify_until_iter
                and iteration > self.options.densify_from_iter
                and iteration % self.options.densification_interval == 0
            ),
            opacity_reset=(
                iteration < self.options.densify_until_iter
                and (
                    iteration % self.options.opacity_reset_interval == 0
                    or self.dataset.white_background
                    and iteration == self.options.densify_from_iter
                )
            ),
        )
        start.record()

    def after_backward(self, model, iteration, loss):
        owner = getattr(model, "_clad_consumer", None)
        consumed = bool(
            owner is not None and owner.active is not None and owner.active.consumed
        )
        expected = self.mode == "clad" and self.policy.reason() is None
        if consumed != expected:
            raise RuntimeError(
                f"Unexpected routing at {iteration}: expected consumption={expected}"
            )
        if consumed and not owner.direct_inputs:
            raise RuntimeError("Combined candidate did not use direct SH inputs")
        self.active["consumed"] = consumed
        activation = getattr(model, "_clad_activations", None)
        activation_consumed = bool(activation is not None and activation.active is not None and activation.active.consumed)
        if activation_consumed != expected:
            raise RuntimeError(f"Unexpected activation routing at {iteration}: expected consumption={expected}")
        self.active["activation_consumed"] = activation_consumed
        consumed_names = ({"f_dc", "f_rest"} if consumed else set())
        if activation_consumed:
            consumed_names |= {"rotation", "scaling", "opacity"}
        self.grad_info = {
            g["name"]: (weakref.ref(g["params"][0]), g["params"][0].grad is not None)
            for g in model.optimizer.param_groups
        }
        if any(self.grad_info[n][1] for n in consumed_names):
            raise RuntimeError("Consumed leaf gradients must remain absent")
        if any(not present for name, (_, present) in self.grad_info.items() if name not in consumed_names):
            raise RuntimeError("A native raw-parameter gradient is missing")

    def after_step(self, model, iteration, loss):
        row = self.active
        row["end"].record()
        consumed = row["consumed"]
        consumed_names = {"f_dc", "f_rest"} if consumed else set()
        if row["activation_consumed"]:
            consumed_names |= {"rotation", "scaling", "opacity"}
        for group in model.optimizer.param_groups:
            name, param = group["name"], group["params"][0]
            original, had_gradient = self.grad_info[name]
            if name in consumed_names:
                if original() is not param:
                    raise RuntimeError(
                        "Consumed parameter changed before the native optimizer"
                    )
                self.expected[name] += 1
            elif (
                iteration < self.options.iterations
                and original() is param
                and had_gradient
            ):
                self.expected[name] += 1
        self.consumed_total += int(consumed)
        self.consumed_stage += int(consumed)
        self.activation_consumed_total += int(row["activation_consumed"])
        self.activation_consumed_stage += int(row["activation_consumed"])
        category = (
            "skipped"
            if not self.policy.optimizer_step
            else "topology" if self.policy.topology_change else "ordinary"
        )
        self.event_counts[category] += 1
        row.update(
            camera=self.camera_name,
            count_after=len(model.get_xyz),
            degree=model.active_sh_degree,
            category=category,
            peak_allocated=self.torch.cuda.max_memory_allocated(),
            peak_reserved=self.torch.cuda.max_memory_reserved(),
            allocated=self.torch.cuda.memory_allocated(),
            reserved=self.torch.cuda.memory_reserved(),
        )
        self.peak_allocated = max(self.peak_allocated, row["peak_allocated"])
        self.peak_reserved = max(self.peak_reserved, row["peak_reserved"])
        expected_event = (
            iteration if row["densify"] else row["previous_structural_iteration"]
        )
        if self.lineage.last_event != expected_event:
            raise RuntimeError("The declared native structural event did not execute")
        if not len(model.get_xyz):
            raise RuntimeError("All Gaussians were pruned")
        self.rows.append(row)
        self.losses.append(loss.detach())
        self.completed_iteration = iteration
        if row["opacity_reset"]:
            row["maximum_opacity_after_reset"] = float(model.get_opacity.max())
            if row["maximum_opacity_after_reset"] > 0.010001:
                raise RuntimeError("Native opacity reset did not cap the opacity")
        if iteration == self.endpoint:
            self.phase_end.record()
            self.phase_end.synchronize()
            self.phase_wall_ms = (self.clock() - self.phase_wall_start) / 1e6
            self.phase_event_ms = self.phase_start.elapsed_time(self.phase_end)
            self.training_memory = dict(
                peak_allocated=self.peak_allocated,
                peak_reserved=self.peak_reserved,
                allocated=self.torch.cuda.memory_allocated(),
                reserved=self.torch.cuda.memory_reserved(),
            )
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError("Nonfinite final training loss")
            self.finish(iteration)
        if iteration % 100 == 0 or self.policy.reason() is not None:
            self.check_state(full=False)
            if not math.isfinite(float(loss.detach())):
                raise RuntimeError("Nonfinite training loss")
        self.torch.cuda.reset_peak_memory_stats()

    def check_state(self, full):
        if self.counters() != self.expected:
            raise RuntimeError(
                f"Optimizer counters disagree: {self.counters()} versus {self.expected}"
            )
        for group in self.model.optimizer.param_groups:
            param = group["params"][0]
            state = self.model.optimizer.state.get(param, {})
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and state[key].shape != param.shape:
                    raise RuntimeError("Optimizer moment shape mismatch")
            if full:
                for key, tensor in [("parameter", param)] + [
                    (key, state[key])
                    for key in ("exp_avg", "exp_avg_sq")
                    if key in state
                ]:
                    if not bool(self.torch.isfinite(tensor).all()):
                        raise RuntimeError(f"Nonfinite {group['name']}.{key}")
        if len(self.lineage.ids) != len(self.model.get_xyz):
            raise RuntimeError("Point lineage is not aligned with the model")

    def metrics(self):
        cameras = sorted(self.scene.getTrainCameras(), key=lambda c: c.image_name)
        count = min(10, len(cameras))
        chosen = [
            cameras[i * (len(cameras) - 1) // max(1, count - 1)] for i in range(count)
        ]
        sets = [
            ("train", chosen),
            (
                "heldout",
                sorted(self.scene.getTestCameras(), key=lambda c: c.image_name),
            ),
        ]
        background = self.torch.tensor(
            [1 if self.dataset.white_background else 0] * 3,
            dtype=self.torch.float32,
            device="cuda",
        )
        rows = []
        with self.torch.no_grad():
            for split, views in sets:
                for camera in views:
                    image = self.train.render(
                        camera, self.model, self.pipe, background
                    )["render"]
                    if camera.alpha_mask is not None:
                        image = image * camera.alpha_mask.cuda()
                    target = camera.original_image.cuda()
                    mse = float((image.double() - target.double()).square().mean())
                    l1 = float((image - target).abs().mean())
                    ssim = float(self.train.fused_ssim(image[None], target[None]))
                    clipped = image.clamp(0, 1)
                    native_mse = float(
                        (clipped.double() - target.double()).square().mean()
                    )
                    values = dict(
                        split=split,
                        camera=camera.image_name,
                        psnr=-10 * math.log10(max(mse, 1e-30)),
                        l1=l1,
                        ssim=ssim,
                        objective=(1 - self.options.lambda_dssim) * l1
                        + self.options.lambda_dssim * (1 - ssim),
                        native_psnr=-10 * math.log10(max(native_mse, 1e-30)),
                    )
                    if not all(
                        math.isfinite(v)
                        for v in values.values()
                        if isinstance(v, float)
                    ):
                        raise RuntimeError("Nonfinite image metric")
                    rows.append(values)
        return rows

    def record_losses(self):
        # Read losses after timing, rather than synchronizing each Adam update.
        values = self.torch.stack(self.losses).cpu().tolist() if self.losses else []
        if len(values) != len(self.rows):
            raise RuntimeError("Loss history and iteration history differ")
        for row, value in zip(self.rows, values):
            row["loss"] = value
        self.losses.clear()
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Nonfinite recorded training loss")

    def finish(self, iteration):
        loss_readback_start = self.clock()
        self.record_losses()
        loss_readback_ms = (self.clock() - loss_readback_start) / 1e6
        verification_start = self.clock()
        self.check_state(full=True)
        self.torch.cuda.synchronize()
        verification_ms = (self.clock() - verification_start) / 1e6
        evaluation_start = self.clock()
        metrics = [] if self.mode == "initial" else self.metrics()
        self.torch.cuda.synchronize()
        evaluation_ms = (self.clock() - evaluation_start) / 1e6
        checkpoint_start = self.clock()
        exposure_optimizer = self.model.exposure_optimizer.state_dict()
        for group in exposure_optimizer["param_groups"]:
            group["lr"] = float(group["lr"])
        auxiliary = dict(
            trajectory=self.mode,
            expected_counters=self.expected,
            consumed_total=self.consumed_total,
            activation_consumed_total=self.activation_consumed_total,
            exposure=self.model._exposure.detach().cpu(),
            exposure_mapping=self.model.exposure_mapping,
            exposure_optimizer=exposure_optimizer,
        )
        checkpoint = self.directory / f"chkpnt{iteration}.pth"
        bundle = save_bundle(
            checkpoint,
            self.model.capture(),
            iteration,
            self.sampler.state_dict(),
            self.order.state_dict(),
            self.config["metadata"],
            auxiliary,
        )
        checkpoint_ms = (self.clock() - checkpoint_start) / 1e6
        self.torch.cuda.synchronize()
        timings = []
        for row in self.rows:
            start, end = row.pop("start"), row.pop("end")
            timings.append(dict(row, event_ms=start.elapsed_time(end)))
        result = dict(
            status="completed",
            mode=self.mode,
            start=self.start_iteration,
            end=iteration,
            iterations=iteration - self.start_iteration,
            roots=self.lineage.roots,
            points=len(self.model.get_xyz),
            gaussian_counts=dict(
                before=self.start_points,
                after=len(self.model.get_xyz),
                minimum=min(
                    [self.start_points]
                    + [min(row["count_before"], row["count_after"]) for row in timings]
                ),
                maximum=max(
                    [self.start_points]
                    + [max(row["count_before"], row["count_after"]) for row in timings]
                ),
            ),
            degree=self.model.active_sh_degree,
            camera_draws=self.order.draws,
            counters=self.counters(),
            metrics=metrics,
            consumed_stage=self.consumed_stage,
            activation_consumed_stage=self.activation_consumed_stage,
            consumed_total=self.consumed_total,
            activation_consumed_total=self.activation_consumed_total,
            event_counts=self.event_counts,
            host_clock=self.clock_name,
            compute_wall_ms=getattr(self, "phase_wall_ms", 0),
            compute_event_ms=getattr(self, "phase_event_ms", 0),
            training_memory=getattr(self, "training_memory", {}),
            setup_memory=self.setup_memory,
            evaluation_ms=evaluation_ms,
            verification_ms=verification_ms,
            checkpoint_ms=checkpoint_ms,
            loss_readback_ms=loss_readback_ms,
            bundle=bundle,
            torch=self.torch.__version__,
            device=self.torch.cuda.get_device_name(),
        )
        (self.directory / "iterations.json").write_text(json.dumps(timings))
        (self.directory / "results.json").write_text(json.dumps(result, indent=2))
        self.sampler.close()
        self.finished = True
        print(json.dumps(result), flush=True)
        raise StageComplete()


def main():
    from campaign import read_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mode", choices=("initial", "native", "clad"), required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = read_config(args.config)
    config["worker_output"] = str(args.output.resolve())
    _, wall_clock = host_clock()
    wall_start = wall_clock()
    train, dataset, options, pipe, effective = runtime(config, args.mode)
    (args.output / "options.json").write_text(json.dumps(effective, indent=2))
    stage = TrainingStage(
        config, train, dataset, options, pipe, args.mode, args.end, args.checkpoint
    )
    try:
        train.training(
            dataset,
            options,
            pipe,
            [],
            [],
            [],
            None if args.checkpoint is None else str(args.checkpoint),
            -1,
            clad_consume=args.mode == "clad",
            training_state=stage,
        )
    except StageComplete:
        if not stage.finished:
            raise
    except Exception as error:
        try:
            write_failure(args.output, error, stage)
        except Exception as diagnostic_error:
            print(
                f"Could not write training failure details: {diagnostic_error}",
                file=sys.stderr,
            )
        raise
    else:
        raise RuntimeError("Trainer returned without a completed stage checkpoint")
    result_path = args.output / "results.json"
    result = json.loads(result_path.read_text())
    result["worker_wall_ms"] = (wall_clock() - wall_start) / 1e6
    result_path.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
