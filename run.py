"""Train, render, or evaluate with the pinned trainer tools."""

from pathlib import Path
from contextlib import nullcontext
import argparse
import json
import math
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "render", "metrics"), default="train")
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--direct-inputs", action="store_true")
    parser.add_argument("--fused-adam", action="store_true",
                        help="select dense fused Gaussian Adam after setup and checkpoint restore")
    parser.add_argument("--quaternion-backward", action="store_true")
    parser.add_argument("--activation-consume", action="store_true")
    parser.add_argument(
        "--trace",
        type=Path,
        help="Small-fixture semantic checks; disables performance measurement",
    )
    parser.add_argument("--summary", type=Path,
                        help="write routing counts and fallback reasons without tensor readbacks")
    parser.add_argument("trainer_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.mode != "train" and any((
        args.direct_inputs, args.fused_adam, args.quaternion_backward,
        args.activation_consume, args.trace, args.summary,
    )):
        parser.error("Consumption, optimizer, trace and routing options require --mode train")
    summary = None
    if args.summary is not None:
        from routing import RoutingSummary

        summary = RoutingSummary(args.summary.resolve(), dict(
            fused_adam=args.fused_adam, direct_inputs=args.direct_inputs,
            quaternion_backward=args.quaternion_backward,
            activation_consume=args.activation_consume,
        ))
    with (summary if summary is not None else nullcontext()):
        launch(args, summary)


def launch(args, summary):
    build = args.build_dir.resolve()
    trainer = build / "trainer"
    from reproduction.prepare import verify_build
    from build import verify

    receipt = verify_build(build)
    verify(args.native_source.resolve(), receipt["inputs"]["native"])
    import torch  # Load Torch runtime libraries before its compiled extension.

    sys.path.insert(0, str(build))
    import current_native_sh_consumer

    sys.modules["diff_gaussian_rasterization._C"] = current_native_sh_consumer
    sys.path[:0] = [str(trainer), str(args.native_source.resolve())]
    if summary is not None:
        import clad_training

        if not hasattr(clad_training, "SUMMARY"):
            raise ValueError("Prepare a fresh trainer copy for --summary support")
        clad_training.SUMMARY = summary
    if args.fused_adam:
        from optimizer_setup import enable_fused_adam
        from scene.gaussian_model import GaussianModel

        enable_fused_adam(GaussianModel)
    if args.quaternion_backward or args.activation_consume:
        normalization = build / "normalization"
        if not normalization.is_dir():
            raise ValueError("Build with --normalization-object first")
        sys.path.insert(0, str(normalization))
        import clad_quaternion
        from clad_normalization import enable_quaternion_backward
        from scene.gaussian_model import GaussianModel

        if args.quaternion_backward:
            enable_quaternion_backward(GaussianModel, clad_quaternion)
        if args.activation_consume:
            if not hasattr(clad_quaternion, "consume_activations"):
                raise ValueError("Build the combined activation consumer first")
            import clad_training

            clad_training.ACTIVATION_EXTENSION = clad_quaternion
    if args.direct_inputs:
        if not hasattr(current_native_sh_consumer, "rasterize_gaussians_direct"):
            raise ValueError("Build the extension with --direct-inputs first")
        import clad_training

        clad_training.DIRECT_INPUTS = True
    if args.trace is not None:
        import clad_training

        trace = args.trace.resolve()
        if trace.exists():
            raise FileExistsError(trace)
        clad_training.TRACE = trace
    rest = args.trainer_args
    if rest and rest[0] == "--":
        rest = rest[1:]
    script = trainer / (args.mode + ".py")
    if not script.is_file():
        raise FileNotFoundError(f"Missing {script.name}; prepare a fresh build with current build.py")
    sys.argv = [str(script), *rest]
    namespace = runpy.run_path(str(script), run_name="__main__")
    if args.mode == "metrics":
        for model in namespace["args"].model_paths:
            results = json.loads((Path(model) / "results.json").read_text())
            if not results or any(
                not math.isfinite(row[key]) for row in results.values()
                for key in ("PSNR", "SSIM", "LPIPS")
            ):
                raise ValueError(f"No finite evaluation for {model}; render a nonempty test split first")


if __name__ == "__main__":
    main()
