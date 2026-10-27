"""Compile the SH and activation objects."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clang", required=True)
    parser.add_argument("--clad-source", type=Path, required=True)
    parser.add_argument("--clad-build", type=Path, required=True)
    parser.add_argument("--plugin", type=Path)
    parser.add_argument("--cuda-home", type=Path, required=True)
    parser.add_argument("--arch", default="sm_86")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    compiler = shutil.which(args.clang)
    if compiler is None:
        parser.error("--clang must name an available CUDA-enabled Clang")
    compiler = str(Path(compiler).resolve())
    source = args.clad_source.resolve()
    build = args.clad_build.resolve()
    plugin = (args.plugin or build / "lib/clad.so").resolve()
    cuda = args.cuda_home.resolve()
    output = args.output.resolve()
    sources = [HERE / "SHGradientConsumer.cu", HERE / "normalization.cu"]
    required = [plugin, source / "include/clad/Differentiator/Differentiator.h",
                cuda / "include/cuda_runtime.h", *sources]
    for path in required:
        if not path.is_file():
            parser.error(f"Missing build input: {path}")
    common = [compiler, "--driver-mode=g++", "-std=c++17", "-O3", "-fPIC",
              "--cuda-path=" + str(cuda),
              "--cuda-gpu-arch=" + args.arch, "-I" + str(source / "include"),
              "-I" + str(build / "include"), "-Xclang", "-load", "-Xclang",
              str(plugin), "-Xclang", "-add-plugin", "-Xclang", "clad",
              "-Xclang", "-plugin-arg-clad", "-Xclang", "-enable-va",
              "-Xclang", "-plugin-arg-clad", "-Xclang",
              "-fno-numerical-diff-fallback"]
    commands = [common + ["-DSH_CONSUMER_NO_MAIN", "-c", str(sources[0]),
                          "-o", str(output / "sh-consumer.o")],
                common + ["-ffp-contract=off", "-c", str(sources[1]),
                          "-o", str(output / "normalization.o")]]
    record = dict(status="planned", commands=commands,
                  inputs={str(path): digest(path) for path in required},
                  compiler=subprocess.check_output([compiler, "--version"], text=True))
    if args.dry_run:
        print(json.dumps(record, indent=2))
        return
    if output.exists():
        parser.error("Use a fresh object output directory")
    output.mkdir(parents=True)
    manifest = output / "objects.json"
    record["status"] = "building"
    manifest.write_text(json.dumps(record, indent=2) + "\n")
    try:
        for label, command in zip(("sh", "activation"), commands):
            with (output / (label + ".log")).open("w") as log:
                subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        record.update(status="completed", objects={
            name: digest(output / name)
            for name in ("sh-consumer.o", "normalization.o")})
    except Exception as error:
        record.update(status="failed", error=str(error))
        raise
    finally:
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(manifest)


if __name__ == "__main__":
    main()
