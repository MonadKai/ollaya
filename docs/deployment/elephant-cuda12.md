# Fork Elephant-1 CUDA12 delivery

This producer builds `0.18.0-elephant1-r1`, a scoped internal fork image for
`elephant-1-v1` text, FP32 computation and 4096-token prompts. It inherits the
immutable, previously verified Ollaya CUDA12 image and its Microsoft ORT 1.28.2
and NVIDIA libraries. Both daemon and GPU runner are replaced together; the
new binary uses dynamic ORT for CPU too. CPU results from the upstream static
build and other model/platform qualification do not transfer to this image.

Build in the owning source checkout with its pinned toolchain/lock:

```sh
OLLAYA_BUILD_VERSION=0.18.0-elephant1-r1 cargo build --release --locked -p ollaya --features ollaya-runner/cuda-dynamic
OLLAYA_BUILD_VERSION=0.18.0-elephant1-r1 cargo build --release --locked -p ollaya-runner --features cuda-dynamic --example parity_elephant
python3 packaging/elephant/prepare-image.py --out /new/context \
  --binary target/release/ollaya --parity target/release/examples/parity_elephant \
  --goldens convert/out/goldens-elephant-1.json --version 0.18.0-elephant1-r1 \
  --source-commit FULL_NATIVE_SOURCE_COMMIT
docker build --platform linux/amd64 --provenance=false -t LOCAL_FORK_TAG /new/context
```

The context contains two ELF files, their identity, and the fixed Dockerfile.
It never includes `convert/`, Python, a graph, tokenizer or weights. A separate
verifier tar contains only standard-library Python and the author's fixed
12-row goldens. Run it from its extracted directory with a selected consumer
interpreter; verify `manifest.json` and archive SHA first. `--evidence NEWDIR`
saves raw requests, responses and HTTP status. `--leave-loaded --keep-alive 30m`
permits the consumer to perform independent residency/GPU observation; failures
still unload. Restarts remain separate Compose actions.

Runtime contract remains UID/GID 1000 by default, `/usr/bin/ollaya serve`,
loopback chosen by `OLLAYA_HOST`, read-only `/models`, writable
`/home/ollaya/.ollaya`, NVIDIA whole-device UUID, and native healthcheck.
The image sets ORT and library paths to its inherited `cuda_v12` pack. Use an
explicit cached model name and immutable manifest; no public Elephant catalog
or model-weight rehosting is implied. Public schema requires at least two
choice options and non-null score descriptions. Images and longer contexts
are rejected. Performance, general K2/K3 release gates and production/business
quality are outside this internal functional qualification.
