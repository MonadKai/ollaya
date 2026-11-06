---
name: ollaya-elephant-release
description: Build and hand off the fork's Elephant-1 CUDA12 image and matching verification package. Use for producer image delivery; site deployment consumes the published artifacts separately.
---

# Deliver the Elephant fork image

Read [the producer contract](../../docs/deployment/elephant-cuda12.md). Preserve the
text-only 4096-token scope and existing parity thresholds. Runtime Python and model
weights remain outside the image.

1. Bind the native source commit, lock and build version. Build both the daemon
   and `parity_elephant` with `cuda-dynamic`; run the relevant tests/parity.
2. Run `packaging/elephant/prepare-image.py` with explicit artifacts, source commit,
   original goldens and a new output directory. It only prepares artifacts. Check
   build hashes, source identity, archive manifest and absence of model weights.
3. Build the isolated context with Docker. Verify both daemon and GPU runner
   version/hash, the inherited CUDA12 pack and the raw-logit gate on target hardware.
   Treat a healthy daemon as readiness only.
4. Publish the built image through the consumer publisher's immutable handoff,
   using a new fork tag. Re-read registry manifest/config/platform and pin the
   resulting digest. Keep the matching verifier archive and its SHA with the release.
5. Hand image and verifier identities, CLI/env/mount/UID contract and qualification
   limits to the consuming deployment skill. Site placement, lifecycle and current
   service facts remain with that consumer and its infrastructure registry.

Keep publication, deployment, inference checks and restart separate operations.
Existing session authorization remains valid. A failed candidate retains evidence;
it does not replace an older service or qualify untested models/backends.
