"""Build the adapter and a separate trainer copy."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shlex
import shutil
import subprocess
import sys
from prepare_native import prepare, prepare_warp_accumulation

HERE = Path(__file__).resolve().parent


def verify(root, hashes):
    for name, digest in hashes.items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(
                f"Unsupported source: {root / name}; expected pinned input"
            )


def toolchain_record(args):
    def identify(command):
        executable = shutil.which(str(command))
        if executable is None:
            raise FileNotFoundError(command)
        return dict(path=str(Path(executable).resolve()),
                    version=subprocess.check_output([executable, "--version"], text=True))

    return dict(
        python=sys.version, torch="2.5.1+cu121", arch=args.arch,
        cxx=identify(args.cxx), nvcc=identify(args.cuda_home / "bin/nvcc"),
        cuda_home=str(args.cuda_home.resolve()),
        cflags=["-O3"], cuda_cflags=["-O3", f"-ccbin={args.cxx}"],
        environment={key: os.environ.get(key) for key in (
            "CC", "NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "NVCC_CCBIN",
            "TORCH_DONT_CHECK_COMPILER_ABI",
        )},
    )


def write_build_record(output, pins, args, toolchain):
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()

    libraries = ["current_native_sh_consumer.so"]
    if args.normalization_object is not None:
        libraries.append("normalization/clad_quaternion.so")
    record = dict(
        version=2, inputs=pins,
        configuration=dict(direct_inputs=args.direct_inputs,
                           warp_accumulation=args.warp_accumulation,
                           normalization=args.normalization_object is not None),
        toolchain=toolchain,
        libraries={name: digest(output / name) for name in libraries},
        manifest_sha256=digest(output / "manifest.json"),
    )
    (output / "build.json").write_text(json.dumps(record, indent=2) + "\n")


def prepare_trainer(trainer, integrated):
    integrated.mkdir()
    for name in [
        "train.py",
        "render.py",
        "metrics.py",
        "lpipsPyTorch",
        "arguments",
        "gaussian_renderer",
        "scene",
        "utils",
        "LICENSE.md",
    ]:
        source = trainer / name
        if source.is_dir():
            shutil.copytree(
                source, integrated / name, ignore=shutil.ignore_patterns("__pycache__")
            )
        else:
            shutil.copy2(source, integrated / name)
    # metrics.py swallows errors and still exits 0. Re-raise them.
    metrics = integrated / "metrics.py"
    text = metrics.read_text()
    handler = '        except:\n            print("Unable to compute metrics for model", scene_dir)'
    if text.count(handler) != 1:
        raise ValueError("Unsupported native metric error handler")
    metrics.write_text(text.replace(handler, handler + "\n            raise"))
    subprocess.run(
        [
            "patch",
            "--batch",
            "-p1",
            "-d",
            str(integrated),
            "-i",
            str(HERE / "trainer.patch"),
        ],
        check=True,
    )
    shutil.copy2(HERE / "adapter.py", integrated / "sh_consumer_adapter.py")
    shutil.copy2(HERE / "integration.py", integrated / "clad_training.py")
    shutil.copy2(HERE / "normalization.py", integrated / "clad_normalization.py")
    shutil.copy2(HERE / "activation.py", integrated / "clad_activations.py")


def build_normalization(obj, output):
    from torch.utils.cpp_extension import load

    directory = output / "normalization"
    directory.mkdir()
    return load(
        name="clad_quaternion",
        sources=[str(HERE / "normalization.cpp")],
        extra_cflags=["-O3"],
        extra_ldflags=[shlex.quote(str(obj))],
        with_cuda=True,
        build_directory=str(directory),
        verbose=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-source", type=Path, required=True)
    parser.add_argument("--trainer-source", type=Path, required=True)
    parser.add_argument("--consumer-object", type=Path, required=True)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--arch", default="8.6")
    parser.add_argument("--direct-inputs", action="store_true")
    parser.add_argument("--warp-accumulation", action="store_true",
                        help="Combine full-warp pixel adjoints; requires --direct-inputs")
    parser.add_argument("--normalization-object", type=Path)
    args = parser.parse_args()
    if args.warp_accumulation and not args.direct_inputs:
        parser.error("--warp-accumulation requires --direct-inputs")
    native = args.native_source.resolve()
    trainer = args.trainer_source.resolve()
    obj = args.consumer_object.resolve()
    output = args.build_dir.resolve()
    if (
        trainer == output
        or trainer in output.parents
        or native == output
        or native in output.parents
    ):
        raise ValueError("Build outside the native and trainer source directories")
    pins = json.loads((HERE / "inputs.json").read_text())
    verify(native, pins["native"])
    verify(trainer, pins["trainer"])
    verify(trainer, pins["evaluation"])
    if not obj.is_file():
        raise FileNotFoundError(obj)
    normalization_object = args.normalization_object
    if normalization_object is not None:
        normalization_object = normalization_object.resolve()
        if not normalization_object.is_file():
            raise FileNotFoundError(normalization_object)
    if not (native / "third_party/glm/glm/glm.hpp").is_file():
        raise ValueError("Initialize the native renderer GLM submodule first")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Use an empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_HOME"] = str(args.cuda_home.resolve())
    os.environ["CXX"] = args.cxx
    os.environ["TORCH_CUDA_ARCH_LIST"] = args.arch
    os.environ.setdefault("MAX_JOBS", "2")
    import torch

    if torch.__version__ != pins["torch"]:
        raise ValueError(f'This optimizer example was validated with {pins["torch"]}')
    from torch.utils.cpp_extension import load

    toolchain = toolchain_record(args)
    prepare(native, output)
    compiled_native = native
    if args.direct_inputs:
        from direct_inputs import prepare as prepare_direct

        compiled_native = prepare_direct(native, output)
    if args.warp_accumulation:
        prepare_warp_accumulation(compiled_native, output)
    sources = [
        compiled_native / "cuda_rasterizer" / name
        for name in ["rasterizer_impl.cu", "forward.cu", "backward.cu"]
    ]
    sources += [
        compiled_native / "rasterize_points.cu",
        output / "split_backward.cu",
        output / "ext.cpp",
    ]
    if args.direct_inputs:
        sources.append(output / "direct_inputs.cu")
    module = load(
        name="current_native_sh_consumer",
        sources=list(map(str, sources)),
        extra_include_paths=list(
            map(str, [output, compiled_native, native / "third_party/glm"])
        ),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", f"-ccbin={args.cxx}"],
        extra_ldflags=[shlex.quote(str(obj))],
        build_directory=str(output),
        verbose=True,
    )
    integrated = output / "trainer"
    prepare_trainer(trainer, integrated)
    files = sources + [
        output / "build.ninja",
        obj,
        Path(module.__file__),
        HERE / "trainer.patch",
        HERE / "run.py",
        HERE / "build.py",
        HERE / "optimizer_setup.py",
        HERE / "routing.py",
        HERE / "reproduction/prepare.py",
        HERE / "inputs.json",
        integrated / "sh_consumer_adapter.py",
        integrated / "clad_training.py",
        integrated / "clad_activations.py",
    ]
    files += [path for path in integrated.rglob("*")
              if path.is_file() and "__pycache__" not in path.parts]
    if args.direct_inputs:
        files += [HERE / "direct_inputs.py"]
        files += list((compiled_native / "cuda_rasterizer").glob("*.h"))
    if args.warp_accumulation:
        files += [HERE / "WarpAccumulation.cuh", HERE / "prepare_native.py",
                  output / "WarpAccumulation.cuh"]
    if normalization_object is not None:
        normalization = build_normalization(normalization_object, output)
        files += [
            normalization_object,
            output / "normalization/build.ninja",
            Path(normalization.__file__),
            HERE / "normalization.cpp",
            HERE / "normalization.cu",
            integrated / "clad_normalization.py",
        ]
    for generated in (obj, normalization_object):
        if generated is None:
            continue
        receipt = generated.parent / "objects.json"
        if receipt.is_file():
            data = json.loads(receipt.read_text())
            recorded = data.get("objects", {}).get(generated.name)
            if recorded is not None:
                actual = hashlib.sha256(generated.read_bytes()).hexdigest()
                if recorded != actual or data.get("status") != "completed":
                    raise ValueError(f"Generated object does not match its receipt: {generated}")
                files.append(receipt)
    (output / "manifest.json").write_text(
        json.dumps(
            {
                str(p): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
                for p in files
            },
            indent=2,
        )
        + "\n"
    )
    write_build_record(output, pins, args, toolchain)
    print(f"Prepared trainer: {integrated}")


if __name__ == "__main__":
    main()
