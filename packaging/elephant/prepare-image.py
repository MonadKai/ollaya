#!/usr/bin/env python3
"""Prepare a weight-free fork image context and an independent verifier archive.

This operation does not build, publish, deploy or change a running service.
Build the supplied binaries from the explicit source commit first.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE_IMAGE = "registry.100credit.cn/xdynamic/x86_64/nvidia/ollaya@sha256:af5e33cf6989959a2bfcc19b4dad61d03048f83b1e8baedfcc43ea0f70909544"


def sha(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def prepare(out, binary, parity, goldens, version, source_commit):
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit) or not re.fullmatch(r"[0-9A-Za-z][0-9A-Za-z.+-]{0,90}", version):
        raise ValueError("explicit full source commit and safe build version required")
    reference = json.loads((ROOT / "docs/validation/elephant-1-review-20261106.json").read_text())
    if sha(goldens) != reference["goldens_sha256"]:
        raise ValueError("verifier goldens differ from the qualified author reference")
    if subprocess.check_output(["git", "status", "--porcelain", "--", "Cargo.toml", "Cargo.lock", "crates"], cwd=ROOT):
        raise ValueError("native source worktree is dirty")
    subprocess.run(["git", "diff", "--exit-code", source_commit, "HEAD", "--", "Cargo.toml", "Cargo.lock", "crates"], cwd=ROOT, check=True)
    text = subprocess.check_output([str(Path(binary).resolve()), "--version"], text=True,
        stderr=subprocess.STDOUT, env={**os.environ, "OLLAYA_HOST": "127.0.0.1:0"}, timeout=30)
    if f"client version is {version}\n" not in text:
        raise ValueError("binary version differs from the requested image")
    for name in (binary, parity):
        with Path(name).open("rb") as source:
            header = source.read(20)
        if header[:5] != b"\x7fELF\x02" or header[18:20] != b"\x3e\x00":
            raise ValueError("Linux amd64 ELF artifacts required")
    out = Path(out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(binary, out / "ollaya")
    shutil.copyfile(parity, out / "parity_elephant")
    for name in ("ollaya", "parity_elephant"):
        (out / name).chmod(0o755)
    metadata = {"schema_version": "ollaya-elephant-build-v1", "version": version,
                "native_source_commit": source_commit,
                "packaging_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                "base_image": BASE_IMAGE, "platform": "linux/amd64", "layout": "elephant-1-v1",
                "weights_bundled": False, "python_runtime_required": False,
                "daemon_and_gpu_runner_sha256": sha(binary), "parity_sha256": sha(parity),
                "runtime": "Microsoft ORT 1.28.2 CUDA12; both native executables use cuda-dynamic",
                "qualified_scope": "Elephant text CUDA, 4096 tokens; no general K2/K3 release qualification"}
    (out / "build.json").write_text(json.dumps(metadata, indent=2) + "\n")
    dockerfile = Path(__file__).with_name("Dockerfile").read_text().replace("@SOURCE@", source_commit).replace("@VERSION@", version)
    (out / "Dockerfile").write_text(dockerfile)
    # The verification package is deliberately outside the runtime image context.
    verifier = package_verifier(out.parent / (out.name + "-verifier"), goldens, version)
    receipt = {"build": metadata, "context": str(out), **verifier}
    (out.parent / (out.name + "-receipt.json")).write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def package_verifier(bundle, goldens, version):
    reference = json.loads((ROOT / "docs/validation/elephant-1-review-20261106.json").read_text())
    if sha(goldens) != reference["goldens_sha256"]:
        raise ValueError("verifier goldens differ from the qualified author reference")
    bundle = Path(bundle).resolve()
    bundle.mkdir(exist_ok=False)
    package = bundle / "elephant_verify"
    package.mkdir()
    source = ROOT / "convert/ollaya_convert/families/elephant"
    for name in ("__init__.py", "daemon_smoke.py", "checks.py", "layout.py"):
        shutil.copyfile(source / name, package / name)
    shutil.copyfile(goldens, bundle / "goldens.json")
    manifest = {str(path.relative_to(bundle)): sha(path) for path in sorted(bundle.rglob("*")) if path.is_file()}
    (bundle / "manifest.json").write_text(json.dumps({"schema_version": "ollaya-elephant-verifier-v1", "version": version,
        "files": manifest, "prompt_limit": 4096, "request_limit": 16384,
        "invocation": "python -m elephant_verify.daemon_smoke --goldens goldens.json --base URL --model MODEL --out NEWFILE --evidence NEWDIR --expected-version " + version}, indent=2) + "\n")
    archive = bundle.parent / (bundle.name + ".tar.gz")
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                tar.add(path, arcname=str(path.relative_to(bundle)))
    return {"verifier_archive": str(archive), "verifier_sha256": sha(archive), "verifier_bytes": archive.stat().st_size}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verifier-only", action="store_true")
    for name in ("out", "goldens", "version"):
        parser.add_argument("--" + name, required=True)
    for name in ("binary", "parity", "source-commit"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    if args.verifier_only:
        print(json.dumps(package_verifier(args.out, args.goldens, args.version)))
    else:
        if not args.binary or not args.parity or not args.source_commit:
            parser.error("image preparation requires --binary, --parity and --source-commit")
        prepare(args.out, args.binary, args.parity, args.goldens, args.version, args.source_commit)
