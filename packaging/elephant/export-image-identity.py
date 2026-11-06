#!/usr/bin/env python3
"""Export source-built OCI manifest/config and an immutable publisher handoff.

Reads Docker's save stream; stores only verified metadata, never image layers.
No tag, push, container, model or service operation is performed.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile


def digest(raw):
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def export(image, target, out):
    local = json.loads(subprocess.check_output(["docker", "image", "inspect", image]))[0]
    desc = local["Descriptor"]
    manifest_sha = desc["digest"]
    config_sha = desc["annotations"]["config.digest"]
    wanted = {manifest_sha.split(":")[1]: "manifest", config_sha.split(":")[1]: "config"}
    found = {}
    proc = subprocess.Popen(["docker", "image", "save", image], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            for member in archive:
                key = Path(member.name).name.removesuffix(".json")
                if member.isfile() and key in wanted:
                    if member.size > 1024 * 1024:
                        raise ValueError("unexpected image metadata size")
                    found[wanted[key]] = archive.extractfile(member).read()
                    if len(found) == 2:
                        break
        if set(found) != {"manifest", "config"}:
            raise ValueError("Docker save stream lacks the exact OCI metadata")
    finally:
        proc.stdout.close()
        if proc.poll() is None:
            proc.terminate()
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        proc.stderr.close()
    if digest(found["manifest"]) != manifest_sha or digest(found["config"]) != config_sha:
        raise ValueError("exported image metadata SHA differs")
    manifest, config = (json.loads(found[name]) for name in ("manifest", "config"))
    if manifest["config"]["digest"] != config_sha or config["rootfs"]["diff_ids"] != local["RootFS"]["Layers"]:
        raise ValueError("exported config/layer identity differs from the local image")
    if config["architecture"] != "amd64" or config["os"] != "linux":
        raise ValueError("this producer requires linux/amd64")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    for name, raw in found.items():
        (out / ("image." + name + ".json")).write_bytes(raw)
    version = config["config"]["Labels"]["org.opencontainers.image.version"]
    handoff = {"schema_version": 1, "created_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "transport": "local-daemon", "source": {"requested_ref": image, "resolved_ref": image,
            "content_digest": config_sha, "manifest_digest": manifest_sha},
        "image": {"engine_name": "ollaya", "engine_version": version + "-cuda12", "size_bytes": local["Size"]},
        "platform": {"type": "x86_64", "oci": "linux/amd64", "provider": "nvidia"}, "archive": None,
        "target": {"registry": "registry.100credit.cn", "image": target}}
    (out / "handoff.json").write_text(json.dumps(handoff, indent=2) + "\n")
    print(json.dumps(handoff), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("image", "target", "out"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    export(args.image, args.target, args.out)
