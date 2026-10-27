"""Build the native adapter and prepare a separate instrumentable trainer copy."""

from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from prepare_native import prepare

HERE = Path(__file__).resolve().parent


def verify(root, hashes):
    for name, digest in hashes.items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(
                f"Unsupported source: {root / name}; expected pinned input"
            )


def prepare_trainer(trainer, integrated):
    integrated.mkdir()
    for name in [
        "train.py",
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
        extra_ldflags=[str(obj)],
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
    parser.add_argument("--normalization-object", type=Path)
    args = parser.parse_args()
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

    prepare(native, output)
    compiled_native = native
    if args.direct_inputs:
        from direct_inputs import prepare as prepare_direct

        compiled_native = prepare_direct(native, output)
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
        extra_ldflags=[str(obj)],
        build_directory=str(output),
        verbose=True,
    )
    integrated = output / "trainer"
    prepare_trainer(trainer, integrated)
    files = sources + [
        obj,
        Path(module.__file__),
        HERE / "trainer.patch",
        integrated / "sh_consumer_adapter.py",
        integrated / "clad_training.py",
        integrated / "clad_activations.py",
    ]
    if args.direct_inputs:
        files += [HERE / "direct_inputs.py"]
        files += list((compiled_native / "cuda_rasterizer").glob("*.h"))
    if normalization_object is not None:
        normalization = build_normalization(normalization_object, output)
        files += [
            normalization_object,
            Path(normalization.__file__),
            HERE / "normalization.cpp",
            HERE / "normalization.cu",
            integrated / "clad_normalization.py",
        ]
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
    print(f"Prepared trainer: {integrated}")


if __name__ == "__main__":
    main()
