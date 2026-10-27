"""Prepare a campaign directory. Does not launch training."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify_build(directory, *, direct_inputs=None, warp_accumulation=None):
    directory = directory.resolve()
    path = directory / "build.json"
    if not path.is_file():
        raise ValueError(f"Missing build receipt: {path}; rebuild with current build.py")
    record = json.loads(path.read_text())
    pins = json.loads((HERE.parent / "inputs.json").read_text())
    if record.get("version") != 2 or record.get("inputs") != pins:
        raise ValueError("Build receipt does not match the pinned renderer/trainer/Torch")
    options = record["configuration"]
    if ((direct_inputs is not None and options.get("direct_inputs") is not direct_inputs)
            or (warp_accumulation is not None and options.get("warp_accumulation") is not warp_accumulation)):
        raise ValueError("Wrong comparison build: native needs direct/warp off; candidate needs both on")
    libraries = record["libraries"]
    names = {"current_native_sh_consumer.so"}
    if options.get("normalization") is True:
        names.add("normalization/clad_quaternion.so")
    if set(libraries) != names:
        raise ValueError("Build receipt has an incomplete library set")
    if any(digest(directory / name) != value for name, value in libraries.items()):
        raise ValueError("Built module changed after its receipt was written")
    if digest(directory / "manifest.json") != record["manifest_sha256"]:
        raise ValueError("Build manifest changed after its receipt was written")
    manifest = json.loads((directory / "manifest.json").read_text())
    required = ("train.py", "render.py", "metrics.py", "gaussian_renderer/__init__.py",
                "scene/gaussian_model.py", "sh_consumer_adapter.py", "clad_training.py",
                "clad_activations.py", "clad_normalization.py")
    runtime = {directory / "trainer" / name for name in required}
    runtime.update((directory / "trainer").rglob("*.py"))
    if any(str(path) not in manifest for path in runtime):
        raise ValueError("Build manifest is missing prepared trainer sources; rebuild")
    for name, expected in manifest.items():
        path = Path(name)
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Build input/runtime changed or missing: {path}; rebuild")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--candidate-build", type=Path, required=True)
    parser.add_argument("--baseline-build", type=Path, required=True)
    parser.add_argument("--normalization-build", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=HERE / "plan.json")
    args = parser.parse_args()
    output = args.output.resolve()
    roots = [args.source, args.trainer_source, args.native_source,
             args.candidate_build, args.baseline_build, args.normalization_build]
    for path in roots:
        path = path.resolve()
        if output == path or path in output.parents:
            parser.error("Prepare the campaign outside its source and build directories")
    libraries = [args.candidate_build / "current_native_sh_consumer.so",
                 args.baseline_build / "current_native_sh_consumer.so",
                 args.normalization_build / "clad_quaternion.so"]
    for path in libraries:
        if not path.is_file():
            parser.error(f"Missing built module: {path}")
    if libraries[0].resolve() == libraries[1].resolve():
        parser.error("Candidate and native control must use separate builds")
    with libraries[0].open("rb") as stream:
        candidate_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    with libraries[1].open("rb") as stream:
        baseline_hash = hashlib.file_digest(stream, "sha256").hexdigest()
    if candidate_hash == baseline_hash:
        parser.error("Candidate and original-native control binaries must differ")
    candidate = verify_build(args.candidate_build, direct_inputs=True, warp_accumulation=True)
    baseline = verify_build(args.baseline_build, direct_inputs=False, warp_accumulation=False)
    if candidate["toolchain"] != baseline["toolchain"]:
        parser.error("Candidate and native control must use the same recorded toolchain/architecture")
    normalization_hash = candidate["libraries"].get("normalization/clad_quaternion.so")
    if normalization_hash is None or digest(libraries[2]) != normalization_hash:
        parser.error("Use the candidate build's recorded activation extension")
    sys.path.insert(0, str(HERE / "runner"))
    import campaign
    from training_state import file_digest

    arguments = SimpleNamespace(plan=args.plan.resolve(), source=args.source,
                                output=output, trainer_source=args.trainer_source,
                                native_source=args.native_source,
                                build_dir=args.candidate_build)
    path, config = campaign.prepare(arguments)
    config.update(native_build_dir=str(args.baseline_build.resolve()),
                  normalization_build_dir=str(args.normalization_build.resolve()),
                  feature_commit="9b4619dbf9db5404429f2dd9e72d3b882999feb9",
                  baseline="Separate original native renderer entry points",
                  candidate="Generated SH/activation consumers and explicit full-warp aggregation")
    receipts = [directory / name for directory in (args.candidate_build, args.baseline_build)
                for name in ("build.json", "manifest.json")]
    frozen = [*libraries, *receipts, args.plan.resolve(), Path(__file__),
              HERE / "run_pair.py", HERE / "quality.py", HERE / "PROTOCOL.md"]
    for item in frozen:
        config["files"][str(item.resolve())] = file_digest(item)
    config["metadata"]["campaign_sha256"] = campaign.config_digest(config)
    path.write_text(json.dumps(config, indent=2, allow_nan=False) + "\n")
    campaign.read_config(path)
    print(path)


if __name__ == "__main__":
    main()
