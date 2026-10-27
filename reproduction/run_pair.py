"""Preview a pair, or run it with --run. Stop on a bad mean or nonfinite state."""

from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import math
import os
import argparse
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run", action="store_true",
                        help="Execute training; without this flag only print the stage plan")
    args = parser.parse_args()
    sys.path.insert(0, str(HERE / "runner"))
    from campaign import read_config
    import quality as criterion

    config = read_config(args.config)
    directory = Path(config["root"])
    fixture = config["plan"].get("fixture", False)
    status = directory / "EXECUTION.json"
    if status.exists():
        raise FileExistsError("Preserve the existing attempt; no automatic retry")
    if not args.run:
        print(json.dumps(dict(status="plan_only", executes_training=False,
                              output=str(directory),
                              endpoints=config["plan"]["stage_endpoints"],
                              mode_order=config["plan"]["mode_order_by_stage"]), indent=2))
        return 0
    env = os.environ.copy()
    env.pop("CUDA_LAUNCH_BLOCKING", None)
    criterion_path = HERE / "quality.py"
    report = dict(status="running", started_utc=datetime.now(timezone.utc).isoformat(),
                  fixture=fixture, pid=os.getpid(), stages=[], retries=0,
                  criterion="Mean held-out loss increase <=2%; worst views reported separately",
                  criterion_source=str(criterion_path),
                  criterion_sha256=hashlib.sha256(criterion_path.read_bytes()).hexdigest())
    write(status, report)

    def worker(mode, endpoint, checkpoint):
        destination = directory / ("initial" if mode == "initial" else f"{mode}/stage{endpoint}")
        destination.parent.mkdir(exist_ok=True)
        command = [sys.executable, str(directory / "runner/campaign_worker.py"),
                   "--config", str(directory / "config.json"), "--mode", mode,
                   "--end", str(endpoint), "--output", str(destination)]
        if checkpoint:
            command += ["--checkpoint", checkpoint]
        report["active"] = dict(mode=mode, endpoint=endpoint, output=str(destination), command=command)
        write(status, report)
        print("START", mode, endpoint, flush=True)
        start = time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW)
        with (directory / f"{mode}-{endpoint}.log").open("x") as stream:
            result = subprocess.run(command, env=env, stdout=stream, stderr=subprocess.STDOUT)
        elapsed = (time.clock_gettime_ns(time.CLOCK_MONOTONIC_RAW) - start) / 1e6
        if result.returncode:
            raise RuntimeError(f"{mode} {endpoint} worker failed ({result.returncode})")
        value = read(destination / "results.json")
        if value["status"] != "completed" or value["end"] != endpoint:
            raise ValueError("Unexpected stage completion")
        value["process_wall_ms"] = elapsed
        print(f"DONE {mode} {endpoint}: {value['points']} Gaussians, {value['compute_wall_ms']/1000:.3f}s loop", flush=True)
        return value

    try:
        initial = worker("initial", 0, None)
        report["initial"] = initial
        checkpoint = {mode: initial["bundle"]["checkpoint"] for mode in ("native", "clad")}
        start = 0
        for endpoint, order in zip(config["plan"]["stage_endpoints"], config["plan"]["mode_order_by_stage"]):
            stage = dict(endpoint=endpoint, modes={})
            report["stages"].append(stage)
            for mode in order:
                result = worker(mode, endpoint, checkpoint[mode])
                if result["start"] != start:
                    raise ValueError("Repeated or skipped stage")
                stage["modes"][mode] = result
                checkpoint[mode] = result["bundle"]["checkpoint"]
                write(status, report)
            native, clad = (stage["modes"][mode] for mode in ("native", "clad"))
            histories = [read(directory / f"{mode}/stage{endpoint}/iterations.json") for mode in ("native", "clad")]
            if [r["camera"] for r in histories[0]] != [r["camera"] for r in histories[1]]:
                raise ValueError("Camera schedules differ")
            if [r["degree"] for r in histories[0]] != [r["degree"] for r in histories[1]]:
                raise ValueError("SH-degree schedules differ")
            if not all(math.isfinite(row["loss"]) for history in histories for row in history):
                raise ValueError("Nonfinite loss history")
            stage["matched_cameras_and_degree"] = True
            if fixture:
                if native["points"] != clad["points"] or native["counters"] != clad["counters"]:
                    raise ValueError("Fixture point/counter behavior differs")
                stage["quality"] = dict(passed=True, scope="Functional fixture; not a quality claim")
            else:
                stage["quality"] = criterion.quality(stage["modes"])
            write(status, report)
            if not stage["quality"]["passed"]:
                report["status"] = "mean_quality_failed"
                break
            start = endpoint
        else:
            report["status"] = "completed"
    except Exception as error:
        report.update(status="worker_failed", error=str(error))
        raise
    finally:
        report.pop("active", None)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write(status, report)
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    sys.exit(main())
