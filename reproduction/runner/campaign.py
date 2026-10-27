"""Run one local native/Clad campaign with two checkpointed trajectories."""

from pathlib import Path
import argparse
import hashlib
import itertools
import json
import math
import shutil
import struct
import subprocess
import sys

from benchmark import compare_quality, host_clock
from training_state import file_digest

HERE = Path(__file__).resolve().parent
RUNNER_FILES = (
    "campaign.py",
    "campaign_worker.py",
    "sampling.py",
    "training_state.py",
    "training_diagnostics.py",
    "benchmark.py",
    "build.py",
    "prepare_native.py",
    "adapter.py",
    "integration.py",
    "inputs.json",
    "trainer.patch",
    "activation.py",
    "normalization.py",
    "normalization.cpp",
    "normalization.cu",
)


def validate_plan(plan):
    ends = plan["stage_endpoints"]
    final = plan["final_global_iteration"]
    if (
        plan["full_training_trajectories_max"] != 2
        or not ends
        or any(type(x) is not int or x <= 0 for x in ends)
        or ends != sorted(set(ends))
        or ends[-1] != final
        or final > 30000
    ):
        raise ValueError(
            "Use two trajectories with increasing stages ending at the global target"
        )
    orders = plan["mode_order_by_stage"]
    if len(orders) != len(ends) or any(
        sorted(order) != ["clad", "native"] for order in orders
    ):
        raise ValueError("Every stage must advance native and Clad once")
    if len(plan["seeds"]) != 1 or len(plan["scenes"]) != 1:
        raise ValueError("This campaign covers one declared scene and seed")
    limits = plan["quality_limits"]
    if set(limits) != {
        "psnr_drop_db_max_per_view",
        "objective_increase_fraction_max_per_view",
    }:
        raise ValueError("Declare both per-view quality limits")
    for key, value in limits.items():
        if key == "psnr_drop_db_max_per_view" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError("Quality limits must be finite and nonnegative")
    budget = plan["budget"]
    if (
        sorted(budget["probe_modes"]) != ["clad", "native", "native-aa"]
        or budget["probe_warmup_blocks_per_mode"] < 1
        or budget["probe_measured_blocks_per_mode"] < 6
        or budget["updates_per_probe_block"] < 1
    ):
        raise ValueError("Invalid three-mode diagnostic schedule")
    if 2 * final > budget["main_loop_updates_upper_bound"]:
        raise ValueError("Training budget exceeded")
    checkpoints = budget["short_probe_checkpoints"]
    if any(x not in ends for x in checkpoints) or len(set(checkpoints)) != len(
        checkpoints
    ):
        raise ValueError("Probes require distinct declared stage checkpoints")
    updates = (
        len(checkpoints)
        * 3
        * (
            budget["probe_warmup_blocks_per_mode"]
            + budget["probe_measured_blocks_per_mode"]
        )
        * budget["updates_per_probe_block"]
    )
    if updates != budget["short_probe_updates_total"] or updates > 1350:
        raise ValueError("Short diagnostic budget exceeded")
    if budget["probe_measured_blocks_per_mode"] % 6:
        raise ValueError("Measured probe rounds must balance all six mode orders")
    allowed = {"eval", "resolution", "data_device"}
    if plan.get("fixture", False):
        allowed |= {
            "densify_from_iter",
            "densify_until_iter",
            "densification_interval",
            "opacity_reset_interval",
            "densify_grad_threshold",
        }
    if set(plan.get("native_options", {})) - allowed:
        raise ValueError("Do not change the declared native training algorithm")
    return plan


def config_digest(config):
    return hashlib.sha256(
        json.dumps(
            {k: v for k, v in config.items() if k != "metadata"},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def quality_options(plan):
    limits = plan["quality_limits"]
    return dict(
        psnr_drop_db=limits["psnr_drop_db_max_per_view"],
        objective_increase_fraction=limits["objective_increase_fraction_max_per_view"],
    )


def read_config(path):
    config = json.loads(Path(path).read_text())
    if config["metadata"]["campaign_sha256"] != config_digest(config):
        raise ValueError("Campaign configuration changed after it was frozen")
    for name, digest in config["files"].items():
        if file_digest(name) != digest:
            raise ValueError(f"Campaign input/source changed: {name}")
    validate_plan(config["plan"])
    return config


def balanced_probe_schedule(original, *args, **kwargs):
    blocks, ends = original(*args, **kwargs)
    orders = list(itertools.permutations(("native", "native-aa", "clad")))
    groups = {}
    for block in blocks:
        if block["phase"] == "timing":
            groups.setdefault(block["round"], []).append(block)
    for round_index, rows in groups.items():
        for row, mode in zip(rows, orders[round_index % 6]):
            row["mode"] = mode
    return blocks, ends


def prepare(args):
    from build import prepare_trainer, verify

    plan = validate_plan(json.loads(args.plan.read_text()))
    source, output = args.source.resolve(), args.output.resolve()
    if output.exists():
        raise FileExistsError("Use a fresh campaign output directory")
    points = source / "sparse/0/points3D.bin"
    if not plan.get("fixture", False):
        for name in ("points3D.bin", "cameras.bin", "images.bin"):
            if not (source / "sparse/0" / name).is_file():
                raise ValueError("Full training requires original COLMAP inputs")
        with points.open("rb") as stream:
            expected_roots = struct.unpack("<Q", stream.read(8))[0]
    else:
        expected_roots = None
    pins = json.loads((HERE / "inputs.json").read_text())
    verify(args.trainer_source.resolve(), pins["trainer"])
    verify(args.native_source.resolve(), pins["native"])
    extension = args.build_dir.resolve() / "current_native_sh_consumer.so"
    if not extension.is_file():
        raise FileNotFoundError(extension)
    output.mkdir()
    original_source = source
    if not plan.get("fixture", False):
        # Use fresh derived COLMAP caches, not an existing points3D.ply that may
        # have come from a previous rendering fixture or pretrained model.
        source = output / "scene"
        inputs = list((original_source / "images").iterdir())
        inputs += [
            original_source / "sparse/0" / name
            for name in ("points3D.bin", "cameras.bin", "images.bin")
        ]
        for old in inputs:
            if not old.is_file():
                continue
            destination = source / old.relative_to(original_source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            try:
                destination.hardlink_to(old)
            except OSError:
                shutil.copy2(old, destination)
    runner = output / "runner"
    runner.mkdir()
    for name in RUNNER_FILES:
        shutil.copy2(HERE / name, runner / name)
    build = output / "build"
    build.mkdir()
    (build / extension.name).symlink_to(extension)
    prepare_trainer(args.trainer_source.resolve(), build / "trainer")
    files = list(runner.iterdir()) + [
        p for p in (build / "trainer").rglob("*") if p.is_file()
    ]
    files += [extension, build / extension.name]
    files += [args.native_source.resolve() / name for name in pins["native"]]
    files += sorted((source / "images").iterdir())
    files += [
        p
        for p in (source / "sparse/0").iterdir()
        if p.is_file() and p.name != "points3D.ply"
    ]
    config = dict(
        version=1,
        root=str(output),
        source=str(source),
        original_source=str(original_source),
        build_dir=str(build),
        trainer_dir=str(build / "trainer"),
        native_source=str(args.native_source.resolve()),
        plan=plan,
        seed=plan["seeds"][0],
        torch_version=pins["torch"],
        expected_initial_points=expected_roots,
        files={str(p): file_digest(p) for p in files},
    )
    config["metadata"] = dict(
        campaign_sha256=config_digest(config),
        final_iteration=plan["final_global_iteration"],
    )
    path = output / "config.json"
    path.write_text(json.dumps(config, indent=2))
    return path, config


def run_command(command, log, root):
    _, clock = host_clock()
    with (root / "commands.jsonl").open("a") as stream:
        stream.write(json.dumps(dict(command=command, log=str(log))) + "\n")
    print("START", log.name, flush=True)
    start = clock()
    with log.open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    elapsed = (clock() - start) / 1e6
    if result.returncode:
        raise RuntimeError(f"Worker failed ({result.returncode}); preserve {log}")
    return elapsed


def probe(config_path, checkpoint, output):
    import benchmark

    config = read_config(config_path)
    budget = config["plan"]["budget"]
    original = benchmark.schedule
    benchmark.schedule = lambda *a, **kw: balanced_probe_schedule(original, *a, **kw)
    # One quality block and two warmups make the declared three untimed blocks.
    arguments = [
        "benchmark.py",
        "--build-dir",
        config["build_dir"],
        "--native-source",
        config["native_source"],
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--views",
        str(budget["updates_per_probe_block"]),
        "--samples",
        str(budget["probe_measured_blocks_per_mode"]),
        "--warmups",
        str(budget["probe_warmup_blocks_per_mode"] - 1),
        "--quality-rounds",
        "1",
        "--quality-steps",
        str(budget["updates_per_probe_block"]),
        "--max-loss-increase-percent",
        str(
            100
            * config["plan"]["quality_limits"][
                "objective_increase_fraction_max_per_view"
            ]
        ),
    ]
    psnr_limit = config["plan"]["quality_limits"]["psnr_drop_db_max_per_view"]
    if psnr_limit is None:
        arguments.append("--psnr-report-only")
    else:
        arguments.extend(["--max-psnr-drop-db", str(psnr_limit)])
    arguments.extend(
        [
            "--",
            "-s",
            config["source"],
            "--data_device",
            "cpu",
            "--resolution",
            str(config["plan"].get("native_options", {}).get("resolution", -1)),
        ]
    )
    if config["plan"].get("native_options", {}).get("eval", True):
        arguments.append("--eval")
    sys.argv = arguments
    benchmark.main()


def run(args):
    config_path, config = prepare(args)
    root = Path(config["root"])
    runner = root / "runner"
    logs = root / "logs"
    logs.mkdir()
    report = dict(
        status="running",
        stages=[],
        probes=[],
        fixture=config["plan"].get("fixture", False),
        quality_limits=config["plan"]["quality_limits"],
        scope="One paired trajectory per mode, shared modified split sampler; no multi-seed or unmodified-stock qualification.",
    )
    status_path = root / "results.json"
    monitor_stream = (root / "gpu.csv").open("x")
    monitor = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,clocks.sm,utilization.gpu,temperature.gpu,power.draw,memory.used",
            "--format=csv",
            "--loop-ms=500",
        ],
        stdout=monitor_stream,
    )
    try:
        initial = root / "initial"
        elapsed = run_command(
            [
                sys.executable,
                str(runner / "campaign_worker.py"),
                "--config",
                str(config_path),
                "--mode",
                "initial",
                "--end",
                "0",
                "--output",
                str(initial),
            ],
            logs / "initial.log",
            root,
        )
        initial_result = json.loads((initial / "results.json").read_text())
        report["initial"] = dict(initial_result, process_wall_ms=elapsed)
        checkpoints = {
            mode: initial_result["bundle"]["checkpoint"] for mode in ("native", "clad")
        }
        saved_native = {}
        for endpoint, order in zip(
            config["plan"]["stage_endpoints"], config["plan"]["mode_order_by_stage"]
        ):
            stage_results = {}
            for mode in order:
                directory = root / mode / f"stage{endpoint}"
                directory.parent.mkdir(exist_ok=True)
                elapsed = run_command(
                    [
                        sys.executable,
                        str(runner / "campaign_worker.py"),
                        "--config",
                        str(config_path),
                        "--mode",
                        mode,
                        "--end",
                        str(endpoint),
                        "--checkpoint",
                        checkpoints[mode],
                        "--output",
                        str(directory),
                    ],
                    logs / f"{mode}-{endpoint}.log",
                    root,
                )
                result = json.loads((directory / "results.json").read_text())
                result["process_wall_ms"] = elapsed
                stage_results[mode] = result
                checkpoints[mode] = result["bundle"]["checkpoint"]
            saved_native[endpoint] = checkpoints["native"]
            comparisons = compare_quality(
                stage_results["native"]["metrics"],
                stage_results["clad"]["metrics"],
                **quality_options(config["plan"]),
            )
            passed = bool(comparisons) and all(r["passed"] for r in comparisons)
            record = dict(
                endpoint=endpoint,
                passed=passed,
                modes=stage_results,
                comparisons=comparisons,
            )
            report["stages"].append(record)
            status_path.write_text(json.dumps(report, indent=2))
            print(
                json.dumps(
                    dict(
                        endpoint=endpoint,
                        quality_passed=passed,
                        native_ms=stage_results["native"]["compute_wall_ms"],
                        clad_ms=stage_results["clad"]["compute_wall_ms"],
                        native_gaussians=stage_results["native"]["points"],
                        clad_gaussians=stage_results["clad"]["points"],
                    )
                ),
                flush=True,
            )
            if not passed:
                report["status"] = "quality_failed"
                break
        else:
            report["status"] = "training_completed"
        # Retain the declared diagnostics even after a quality stop, but only for
        # milestones actually reached. There are no new full training attempts.
        for endpoint in config["plan"]["budget"]["short_probe_checkpoints"]:
            if endpoint not in saved_native:
                continue
            output = root / f"probe{endpoint}"
            command = [
                sys.executable,
                str(runner / "campaign.py"),
                "probe",
                "--config",
                str(config_path),
                "--checkpoint",
                saved_native[endpoint],
                "--output",
                str(output),
            ]
            with (root / "commands.jsonl").open("a") as stream:
                stream.write(json.dumps(dict(command=command)) + "\n")
            print("START", f"probe{endpoint}", flush=True)
            with (logs / f"probe{endpoint}.log").open("x") as stream:
                result = subprocess.run(
                    command, stdout=stream, stderr=subprocess.STDOUT
                )
            data = (
                json.loads((output / "results.json").read_text())
                if (output / "results.json").exists()
                else {}
            )
            report["probes"].append(
                dict(
                    endpoint=endpoint,
                    returncode=result.returncode,
                    status=data.get("status", "failed"),
                    quality_passed=data.get("quality_passed", False),
                    timing=data.get("timing"),
                )
            )
            status_path.write_text(json.dumps(report, indent=2))
            if result.returncode not in (0, 2):
                report["diagnostic_failure"] = f"probe{endpoint} failed; no retry"
                break
        if report["status"] == "training_completed":
            if report.get("diagnostic_failure"):
                report["status"] = "training_completed_diagnostic_failed"
            elif any(row["status"] == "timing_unreliable" for row in report["probes"]):
                report["status"] = "training_completed_timing_unreliable"
            else:
                report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        monitor.terminate()
        monitor.wait(timeout=10)
        monitor_stream.close()
        status_path.write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                status=report["status"],
                stages=len(report["stages"]),
                probes=len(report["probes"]),
            )
        ),
        flush=True,
    )
    if report["status"] != "completed":
        raise SystemExit(2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    paired = commands.add_parser("run")
    for name in (
        "build-dir",
        "native-source",
        "trainer-source",
        "source",
        "output",
        "plan",
    ):
        paired.add_argument("--" + name, type=Path, required=True)
    local = commands.add_parser("probe")
    for name in ("config", "checkpoint", "output"):
        local.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        run(args)
    else:
        probe(args.config, args.checkpoint, args.output)


if __name__ == "__main__":
    main()
