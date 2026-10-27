"""Tiny synthetic train, restore, and render check."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
PARAMETERS = ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity")


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def make_scene(scene):
    import numpy as np
    from PIL import Image
    from plyfile import PlyData, PlyElement

    sparse = scene / "sparse/0"
    sparse.mkdir(parents=True)
    (scene / "images").mkdir()
    (sparse / "cameras.txt").write_text("1 PINHOLE 64 64 64 64 32 32\n")
    rows = []
    y, x = np.mgrid[:64, :64]
    for index, translation in enumerate((-.5, .5), 1):
        name = f"{index:05d}.png"
        pixels = np.stack((40 + 2*x, 40 + 2*y, 80 + ((x+y+index*8) % 96)), -1)
        Image.fromarray(pixels.astype(np.uint8)).save(scene / "images" / name)
        rows.append(f"{index} 1 0 0 0 {translation} 0 0 1 {name}\n\n")
    (sparse / "images.txt").write_text("".join(rows))
    rng = np.random.default_rng(42)
    xyz = rng.uniform((-.4, -.4, 2.5), (.4, .4, 3.5), (1024, 3))
    vertices = np.zeros(1024, dtype=[(key, "f4") for key in ("x", "y", "z", "nx", "ny", "nz")]
                        + [(key, "u1") for key in ("red", "green", "blue")])
    for index, key in enumerate(("x", "y", "z")):
        vertices[key] = xyz[:, index]
    for key in ("red", "green", "blue"):
        vertices[key] = 128
    PlyData([PlyElement.describe(vertices, "vertex")]).write(sparse / "points3D.ply")


def render_check(args):
    import torch
    from reproduction.prepare import verify_build

    build = args.build_dir.resolve()
    verify_build(build)
    sys.path[:0] = [str(build), str(build / "trainer"), str(args.native_source.resolve())]
    import current_native_sh_consumer
    sys.modules["diff_gaussian_rasterization._C"] = current_native_sh_consumer
    from scene.gaussian_model import GaussianModel
    import run

    loaded = []
    original = GaussianModel.load_ply

    def load(model, *positional, **keywords):
        original(model, *positional, **keywords)
        loaded.append((model, {key: getattr(model, key).detach().clone() for key in PARAMETERS}))

    GaussianModel.load_ply = load
    sys.argv = [str(HERE / "run.py"), "--mode", "render", "--build-dir", str(build),
                "--native-source", str(args.native_source.resolve()), "--", "-m",
                str(args.output / "initial-model"), "--iteration", "7", "--skip_train", "--eval"]
    run.main()
    require(len(loaded) == 1, "Expected one saved model")
    model, before = loaded[0]
    for key, value in before.items():
        require(torch.equal(getattr(model, key), value), f"Rendering changed {key}")
    require(getattr(model, "optimizer", None) is None, "Rendering constructed an optimizer")
    write(args.output / "render-check.json", dict(passed=True, parameter_families=PARAMETERS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="new directory; preserves failed attempts")
    parser.add_argument("--metrics", action="store_true", help="also run LPIPS/PSNR/SSIM; may download weights")
    parser.add_argument("--render-check", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.render_check:
        render_check(args)
        return
    from reproduction.prepare import verify_build

    receipt = verify_build(args.build_dir, direct_inputs=True, warp_accumulation=True)
    require(receipt["configuration"]["normalization"], "Smoke check requires the activation extension")
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    report = dict(status="running", commands=[], metrics_requested=args.metrics,
                  scope="Synthetic functional fixture; not quality, timing or trajectory-equivalence evidence")
    record = out / "smoke.json"
    write(record, report)

    def execute(label, command, expected=0):
        report["commands"].append(dict(label=label, command=command))
        write(record, report)
        with (out / (label + ".log")).open("x") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
        report["commands"][-1]["returncode"] = result.returncode
        write(record, report)
        require(result.returncode == expected if expected is not None else result.returncode != 0,
                f"{label} returned {result.returncode}; see {out / (label + '.log')}")
        print(label, "passed", flush=True)

    try:
        make_scene(out / "scene")
        common = [sys.executable, "-B", str(HERE / "run.py"), "--build-dir",
                  str(args.build_dir.resolve()), "--native-source", str(args.native_source.resolve())]
        consumption = ["--direct-inputs", "--fused-adam", "--activation-consume", "--quaternion-backward"]
        trainer = ["-s", str(out / "scene"), "--disable_viewer", "-r", "1", "--iterations", "7",
                   "--densify_from_iter", "1", "--densify_until_iter", "6",
                   "--densification_interval", "2", "--opacity_reset_interval", "5",
                   "--densify_grad_threshold", "1e-12", "--save_iterations", "4",
                   "--checkpoint_iterations", "4", "--clad_consume"]
        for mode in ("initial", "resume"):
            command = common + consumption + ["--trace", str(out / (mode + "-trace.json")),
                "--summary", str(out / (mode + "-routing.json")), "--", *trainer,
                "-m", str(out / (mode + "-model"))]
            if mode == "resume":
                command += ["--start_checkpoint", str(out / "initial-model/chkpnt4.pth")]
            execute(mode, command)
        initial, resumed = (read(out / (mode + "-trace.json")) for mode in ("initial", "resume"))
        for rows, iterations, consumed in ((initial, list(range(1, 8)), [1, 3, 6]),
                                            (resumed, [5, 6, 7], [6])):
            require([r["iteration"] for r in rows] == iterations, "Missing training iteration")
            for name in ("sh", "activation"):
                got = [r["iteration"] for r in rows if (r if name == "sh" else r[name])["consumed"]]
                require(got == consumed, f"Unexpected {name} consumption: {got}")
            require(all(r["count_after"] > 0 for r in rows), "Fixture pruned every Gaussian")
        for name in ("sh", "activation"):
            before = initial[4] if name == "sh" else initial[4][name]
            after = resumed[0] if name == "sh" else resumed[0][name]
            require(before["counters_before"] == after["counters_before"], "Restore changed counters")
        require(initial[1]["count_after"] > initial[1]["count_before"], "Fixture did not densify")
        report["lifecycle"] = dict(passed=True, initial_iterations=7, resumed_iterations=3,
            topology=[dict(iteration=r["iteration"], before=r["count_before"], after=r["count_after"])
                      for r in initial if r["count_before"] != r["count_after"]])
        model = out / "initial-model"
        files = [model / "point_cloud/iteration_7/point_cloud.ply", model / "exposure.json"]
        before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
        execute("render", [sys.executable, "-B", str(Path(__file__).resolve()), "--render-check",
            "--build-dir", str(args.build_dir.resolve()), "--native-source", str(args.native_source.resolve()),
            "--output", str(out)])
        from PIL import Image
        images = list((model / "test/ours_7/renders").glob("*.png"))
        require(len(images) == 1, "Expected one test-split render")
        for path in images:
            with Image.open(path) as image:
                image.verify()
        require(all(hashlib.sha256(p.read_bytes()).hexdigest() == before[str(p)] for p in files),
                "Rendering changed saved model files")
        report["render"] = dict(passed=True, views=len(images), held_out_during_training=False,
                                parameters_unchanged=read(out / "render-check.json")["passed"])
        execute("invalid-render-option", common + ["--mode", "render", "--activation-consume"], expected=2)
        if args.metrics:
            execute("metrics", common + ["--mode", "metrics", "--", "-m", str(model)])
            values = read(model / "results.json")
            require(bool(values) and all(math.isfinite(row[key]) for row in values.values()
                    for key in ("PSNR", "SSIM", "LPIPS")), "Invalid saved-model metrics")
            execute("missing-metrics", common + ["--mode", "metrics", "--", "-m", str(out / "missing")], expected=None)
            empty = out / "empty-model"
            (empty / "test").mkdir(parents=True)
            execute("empty-metrics", common + ["--mode", "metrics", "--", "-m", str(empty)], expected=None)
            report["metrics"] = dict(passed=True, values=values)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error=str(error))
        raise
    finally:
        write(record, report)
    print(record)


if __name__ == "__main__":
    main()
