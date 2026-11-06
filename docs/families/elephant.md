# Elephant-1 (fork implementation)

Base Labs' [Elephant-1](https://huggingface.co/baselabs/elephant-1) is a Qwen3.5-9B
decision model. This fork adds the **text-only** `elephant-1-v1` ONNX layout.
Full-weight parity is pending the pinned checkpoint download; it is not yet a
validated catalog tag. The implementation lives on `feat/elephant-1`.

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
still applies its shared request validation and missing-instruction defaults;
use explicit instructions when comparing HTTP results with OneJev. The direct
runner also supports the author's omitted-instruction defaults.

| Tensor | Type | Shape |
| --- | --- | --- |
| `input_ids` | int64 | `[rows, seq]`, right-padded to a multiple of 64 |
| `last_pos` | int64 | `[rows]`, last real token |
| `prefix_lengths` | int64 | `[rows]`, boundary between the two blocks |
| `cand_logits` | float32 | `[rows, 256]`, author slot order |

The Rust runner processes one row per ORT call to bound memory. Its decision
configuration supplies the prompt limit (export defaults to 4096, pending
full-weight resource validation); it refuses longer inputs and images. The
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
The reference wrapper verifies hashes of the patched engine, prompt and readout.

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

The export checks the complete weight SHA-256 before loading. The reference
then compares **every loaded text parameter** with the source checkpoint to
detect missing-key/random-initialization failures when loading a multimodal
checkpoint into a text-only class. It requires noncausal attention, slot-LM
readout and no external calibration. Export stops if the fp32 eager graph
differs from the author's sequential raw logits by more than `1e-3`.

From the repository root, with the local compatible dynamic ORT environment:

```sh
cargo run -p ollaya-runner --features cuda-dynamic --example parity_elephant -- \
  convert/out/elephant-1-wl convert/out/goldens-elephant-1.json cpu
CUDA_VISIBLE_DEVICES=4 cargo run -p ollaya-runner --features cuda-dynamic --example parity_elephant -- \
  convert/out/elephant-1-wl convert/out/goldens-elephant-1.json cuda
```

`ORT_DYLIB_PATH` and the CUDA library path must name a compatible local ORT GPU
installation. No Python package is used by the runtime.

## Evidence so far (y108p62, 2026-11-06)

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
  `-D warnings`; all 38 Python conversion tests pass.
- Full Elephant-1 weight loading, export, CPU/CUDA parity, long-input memory,
  HTTP integration and quality measurements are still required.

See [model contract](../elephant-1-model-contract.md) and
[design research](../elephant-1-feasibility.md) for the public image-reference
gap and the cache/long-context follow-up. Catalog publication is gated on
full-weight results; the synthetic artifact never becomes a catalog model.
