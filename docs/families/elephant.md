# Elephant-1 (fork implementation)

Base Labs' [Elephant-1](https://huggingface.co/baselabs/elephant-1) is a Qwen3.5-9B
decision model. This fork adds the **text-only** `elephant-1-v1` ONNX layout.
Full-weight CPU/CUDA parity and native daemon integration pass on y108p62.
The implementation lives on `feat/elephant-1`, with a private `elephant:1-text`
development tag. A scoped internal CUDA12 fork image and standalone verifier are
now available through the [producer delivery contract](../deployment/elephant-cuda12.md).
Public catalog and upstream release publication remain outside this fork delivery.

The model and reference are pinned to HF revision
`b3f8a93462fb1051a9de031e94a7ff380e254072` and OneJev
`81ce62f1597c91e46767d4d02d6ac2e18534fe94` with the model's `serve.patch`.
The author model card declares Apache-2.0. Weights remain the author's original
`model.safetensors`; no model files are committed or re-hosted.

## Execution contract

Each question renders one author chat prompt. The system message and state are
the first block; the question, options and assistant prompt are the second.
The eight full-attention layers see every token in their own block and earlier
blocks. The 24 Gated DeltaNet layers remain causal. Padding is excluded from
full attention's keys. Each question currently recomputes the prefix.

The graph reads the answer position and the LM head's 256 author slot rows.
It returns raw logits, without generation or sampling. The runner changes
the author's yes/no slot order into Ollaya's false/true order; choices and
scores keep their original order. Calibration is temperature 1.0, as in this
checkpoint. The shared daemon renders answers.

Two existing wire conventions differ from the author's OneJev service:
Ollaya rounds to four decimals (OneJev: six), and its score confidence is
normalized peak probability, whereas OneJev uses distance from the modal
level. These are deterministic API transformations, not model calibration.
Raw slot logits, option probabilities, choice and expected score are the
model parity targets. Do not compare these two confidence formulas as if
they were a numerical error, or change other families' wire behavior.

The layout accepts 1–255 named choices and 1–10 score levels, structured state,
and the author's structured instructions/criteria entries. Ollaya's public API
still applies its shared request validation and missing-instruction defaults:
public choices require at least two options and public score descriptions
cannot be null. Use explicit instructions when comparing HTTP results with
OneJev. A null score description can be replaced by an empty string with an
identical rendered prompt. The direct runner supports the author's single
option, null descriptions and omitted-instruction defaults.

| Tensor | Type | Shape |
| --- | --- | --- |
| `input_ids` | int64 | `[rows, seq]`, right-padded to a multiple of 64 |
| `last_pos` | int64 | `[rows]`, last real token |
| `prefix_lengths` | int64 | `[rows]`, boundary between the two blocks |
| `cand_logits` | float32 | `[rows, 256]`, author slot order |

The Rust runner processes one row per ORT call to bound memory. Its decision
configuration supplies the prompt limit (4096, validated with real weights
on CPU and CUDA); it refuses longer inputs and images. The
training limit of 16384 is an upper bound, not a verified runtime promise.
The total request budget counts the prefix once plus all question suffixes
and is capped at 16384 tokens.

## Files

- `convert/ollaya_convert/families/elephant/`: independent layout, pinned author
  reference wrapper, full-sequence graph, export, goldens and synthetic smoke builder.
- `crates/ollaya-decision/src/elephant.rs`: author rendering and common token boundary.
- `crates/ollaya-runner/src/elephant.rs`: tokenizer/slot checks and ORT execution.
- `crates/ollaya-runner/examples/parity_elephant.rs`: tokens and raw-logit gate.
- `convert/tests/fixtures/elephant-layout.json`: author-generated token fixtures.

## Reproduce

Use `uv sync --extra elephant` in `convert/`. The extra adds only the pinned
Pydantic schema dependency; Python remains build-time only. Obtain the fixed
OneJev archive from
`https://codeload.github.com/OmniJev/OneJev/tar.gz/81ce62f1597c91e46767d4d02d6ac2e18534fe94`,
extract it in a build directory, and apply the fixed model snapshot's
`serve.patch` using `git apply`. Set `ELEPHANT_ONEJEV` to that directory.
The reference wrapper verifies hashes of the patched engine and its imported
OneJev source modules, and rejects a previously imported `qev` from another tree.

The author's Transformers pin is 5.19.0; the rest of Ollaya's conversion stack
stays on 5.17. Keep the author version in a separate build-time target and put
it ahead of `convert` on `PYTHONPATH` for full-weight reference/export runs:

```sh
uv pip install --python .venv/bin/python --target /path/to/reference-deps --no-deps transformers==5.19.0
export PYTHONPATH=/path/to/reference-deps:.
export ELEPHANT_ONEJEV=/path/to/pinned-patched/OneJev
MODEL=/absolute/path/to/fixed/huggingface/snapshot
.venv/bin/python -m ollaya_convert.families.elephant.export --model "$MODEL" --out out/elephant-1-wl
.venv/bin/python -m ollaya_convert.families.elephant.goldens --model "$MODEL" --out out/goldens-elephant-1.json
```

Reference loading checks the complete weight SHA-256 and the pinned config,
tokenizer, chat template and readout files before loading. Unexpected calibration,
readout-head or adapter overrides are rejected. The reference
then compares **every loaded text parameter** with the source checkpoint to
detect missing-key/random-initialization failures when loading a multimodal
checkpoint into a text-only class. It requires noncausal attention, slot-LM
readout and no external calibration. Export stops if the fp32 eager graph
differs from the author's sequential raw logits by more than `1e-3`.

From the repository root, with the local compatible dynamic ORT environment:

```sh
export OLLAYA_THREADS=8
cargo run -p ollaya-runner --features cuda-dynamic --example parity_elephant -- \
  convert/out/elephant-1-wl convert/out/goldens-elephant-1.json cpu
CUDA_VISIBLE_DEVICES=4 cargo run -p ollaya-runner --features cuda-dynamic --example parity_elephant -- \
  convert/out/elephant-1-wl convert/out/goldens-elephant-1.json cuda
```

`ORT_DYLIB_PATH` and the CUDA library path must name a compatible local ORT GPU
installation. No Python package is used by the runtime.

## Full-weight evidence (y108p62, 2026-11-06)

The verified original weight file is 18,819,722,392 bytes, SHA-256
`682f02348dfe4617dd21bba2915c12e805de8cf16327a61e895a831ea681d0ba`.
All 427 loaded text parameters match the checkpoint exactly in their loaded
dtype; the 333 vision tensors are deliberately unused. The weightless graph
is 12,530,518 bytes and references original BF16 bytes with FP32 computation.
Only mask, rotary and slot/configuration constants are inline, not learned
weights. These are development correctness results, not quality or speed
benchmarks and not a public K2/K3 release qualification.

| Gate | Cases | Maximum raw-logit error | Maximum probability error |
| --- | --- | --- | --- |
| Eager graph vs author sequential cache-fork | export sample | `4.7683716e-6` | — |
| Rust/ORT CPU | 9 requests / 12 rows | `1.1920929e-5` | `5.2638805e-7` |
| Rust/ORT CUDA | same 12 rows | `1.5258789e-5` | `1.1922098e-6` |
| Isolated runner HTTP CUDA | same 12 rows | `1.5258789e-5` | — |

Every decision matches. Token IDs, prefix boundaries and option counts match
exactly. Inputs include structured/Unicode/control-token state, multiple
questions, 1/26/27/255 choices and exact 512/2048/4096-token prompts. The
unchanged gate is logits `1e-3`, probabilities `1e-4`, identical decisions.
CPU uses eight threads; CUDA uses physical GPU 4 (logical `cuda:0`, RTX 5090).
The same 12 rows passed again after the three review rounds; the gate now uses
the loaded model's actual layout instead of the fixture's layout.

The private registry pull, `/api/show`, `/api/decide`, `/v1/systemone`, `/api/ps`
and CLI all pass with the actual packaged graph and original weights.
Public HTTP checks cover 11 schema-compatible golden rows plus two repeated
rows through `/v1/systemone`; the author-only schema case is explicitly
rejected, and its compatible score/noul subset keeps identical prompts.
Maximum wire error is `4.9671153e-5`, consistent with four-decimal rounding.
Images, empty score criteria, overlong text and prefix-marker collisions
return 422. The isolated runner returns 400 for those same cases.

After the private registry was stopped, cached inference still passed.
The native runner's loaded library map contains no Python, Torch or
`convert/` paths. A private native library prefix holds ORT 1.28.2 CUDA13
and its dependencies. No global runtime installation was changed.
The model was unloaded and test listeners stopped after verification.

Machine-readable receipts: [initial development validation](../validation/elephant-1-y108p62.json)
and [three-round review](../validation/elephant-1-review-20261106.json).
The [review report](../validation/elephant-1-review-20261106.md) records the fixes,
negative controls, fresh validation and deployment-skill assessment.
The original full goldens and logs remain at
`/opt/kai.liu/var/ollaya-elephant` and `convert/out/` on this host.

## Local development tag

On the prepared host, start the isolated daemon in one terminal:

```sh
/opt/kai.liu/var/ollaya-elephant/ollaya serve
```

In another terminal:

```sh
/opt/kai.liu/var/ollaya-elephant/ollaya run elephant:1-text \
  'The customer was charged twice and requests a refund.' \
  --ask 'Does the customer want a refund?' --format json --keepalive 0
```

The wrapper uses localhost port 11439, a private model store, physical GPU 4
and eight CPU threads. It defaults to unloading after each request. Override
`ELEPHANT_GPU` only after checking that the chosen GPU is available.
Stop this daemon with the same wrapper's `stop` command. The cached tag needs
no registry server, Python environment or further weight download.

To create a fresh private registry after exporting and passing parity:

```sh
PYTHONPATH=convert convert/.venv/bin/python -m ollaya_convert.families.elephant.package_local \
  --model convert/out/elephant-1-wl --registry /path/to/private-registry \
  --origin http://127.0.0.1:18899
```

Serve that directory locally while pulling with `OLLAYA_REGISTRY` set to its
origin. The manifest names the author's pinned weight/tokenizer URLs; the
registry contains only graph/configuration artifacts. On this host the verified
HF blob was hardlinked into the private store to avoid downloading it twice.
Use `daemon_smoke --goldens ... --out ...` after the raw-logit gates to repeat
the public API checks. Both HTTP checkers reject empty/misaligned/non-finite
goldens before starting work and write PASS only after successful cleanup.
The private tag requires FP32 computation; an experimental `--fp16` export
cannot be packaged as `elephant:1-text` and needs its own tag and qualification.
Private packaging deliberately does not modify the
public catalog, website, release tags or upstream repository.

## Earlier structural checks

- Author layout: 6 requests / 9 rows, including Unicode, structured state,
  control tokens and 26/27/255 options; Python and Rust token sequences,
  prefix boundaries and option counts agree exactly.
- A tiny random four-layer hybrid model matches HF eager with an explicit
  block mask; ONNX dynamic lengths 64 and 192 match eager within `1e-5`.
- The synthetic model, real tokenizer and all 9 author-layout rows pass the
  Rust/ORT CUDA path on GPU 4: maximum raw-logit error `1.3411045e-7` against
  HF eager, identical decisions. This proves wiring, **not Elephant-1 accuracy**.
- The same synthetic model passes localhost runner HTTP inference and four
  rejection cases (images, empty score criteria, overlong text and author
  prefix-marker collision).
- On Transformers 5.19, the pinned author's actual noncausal mask installer
  and sequential cache-fork code agree with the complete-sequence graph on
  all 12 random-model rows, including exact 512/2048/4096-token inputs:
  maximum raw-logit error `8.9406967e-8`. The full-weight golden generator
  includes these three boundary inputs automatically.
- Full Rust workspace tests pass with `cuda-dynamic`; Clippy passes with
  `-D warnings`; all 45 Python conversion tests pass, including seven
  reference identity, packaging and HTTP-gate regression tests.
- Public quality/speed benchmarks, other hardware/OS validation, image
  reference parity and prompts above 4096 remain future work. Shared-prefix
  caching is a performance follow-up behind the same Engine interface.

See [model contract](../elephant-1-model-contract.md) and
[design research](../elephant-1-feasibility.md) for the public image-reference
gap and the cache/long-context follow-up. Public catalog publication still
requires the repository's release/quality gates; the synthetic artifact never
becomes a catalog model.
