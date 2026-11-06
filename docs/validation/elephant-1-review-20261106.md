# Elephant-1: three review and repair rounds

Scope: fork branch `feat/elephant-1`, fixed base
`9b8698e39f11a0f0bcd08ad686de51cd1b4f6fa8`; initial delivery `d32dba4`.
User authorized three review/fix rounds, then a push to the fork branch when
sound. Each round reviewed the complete diff independently on Standards and
Spec axes. This is development validation, not a release or public benchmark.

The first-principles questions were: what bytes and configuration define the
model; whether its prompt, two attention blocks, causal recurrence and answer
slots reproduce the author; whether the runtime executes that same contract;
and whether a passing check actually proves a nonempty, correctly aligned
comparison on the artifact that will serve requests.

## Standards

Round 1 found two material issues, independently reproduced:

1. The parity example rendered rows using the goldens' layout and passed them
   straight to `option_logits`. A broken exported template and prompt limit
   could pass. The gate now obtains rows from the loaded model's `decision.json`
   and tokenizer, then compares all tokens and boundaries with the goldens.
   Empty cases and invalid device arguments are rejected.
2. The private packager always labeled its graph FP32, including experimental
   FP16 exports. It now refuses non-FP32 exports before packaging; reduced
   precision requires a different tag and independent gate (ADR 0008).

Round 2 found no additional Standards violation. Round 3 found no additional
Standards violation and independently ran all seven new tool regressions.
No style-only restructuring was requested or applied.

## Spec

Round 1 confirmed the same two issues and found one additional identity gap:
matching all loaded parameters did not fix the configuration/tokenizer that
interprets them. A tiny-model counterexample kept every parameter identical
but changed RoPE theta and changed logits by approximately 0.02194. Reference
and export could otherwise inherit the same drift and still agree.

`ref.load` now checks the original full weight SHA-256 plus the pinned semantic
files before initialization. The seven file hashes were checked against the
fixed HF tree's Git/LFS object IDs. It refuses unexpected calibration/readout-head/
adapter overrides, requires the author's Transformers 5.19.0, verifies the
OneJev source closure and rejects cached imports from another source tree.
The five newly pinned, unpatched transitive modules were independently compared
with their original files at OneJev `81ce62f`.

Round 2 found one additional verification gap: empty cases or missing logits
could skip HTTP numeric comparisons and still write PASS. A no-GPU mock
reproduced `golden_rows=0, passed=true`. Both HTTP validators now admit only
nonempty, aligned question/row/logit sets with correct widths, metadata and
finite values before any process/network work. PASS is written after confirmed
unload/cleanup. Export/reference checks also reject non-finite results.

Round 3 found no additional material Spec defect. Prompt/split, block visibility,
DeltaNet causality, padding, 256-slot readout, canonical noul order, temperature,
resource limits, packaging and the complete validation chain were rechecked.

## Repairs and fresh verification

Code fixes are in `f0807a4` (round 1) and `66eae06` (round 2). Round 3 reviewed
`66eae06`; the final documentation commit changes no executed code. The native
serving binary remains the previously checked `36b2020` engine: no runtime
implementation changed during these rounds. The parity example and Python
admission/verification tooling changed; the graph and original goldens did not.

- Broken template, one-token prompt limit, empty goldens and misspelled device
  all fail; the unchanged tiny artifact still passes.
- Seven new tool regressions cover semantic-file drift, foreign reference
  imports, injected calibration, precision admission, damaged/empty goldens,
  positive admission and failed unload without a PASS receipt.
- Complete conversion suite: 45/45 pass. Rust fmt, Clippy `-D warnings` and
  full workspace tests with `cuda-dynamic` pass.
- Fresh original-weight/semantic/source validation and loading of all 427 text
  parameters pass. The checked sample matches the existing author golden
  exactly (maximum raw-logit error 0); goldens were not regenerated.
- Full real CPU/CUDA gate: the same 9 requests / 12 rows, identical tokens,
  boundaries and decisions. CPU maximum logit error `1.1920929e-5`, probability
  error `5.2638805e-7`; CUDA `1.5258789e-5` and `1.1922098e-6`.
- Updated isolated runner HTTP: all 12 rows and four rejection cases pass.
  Updated public daemon checker: 11 compatible golden rows plus two repeated
  System One rows, one explicit author-only schema rejection and four 422
  rejection cases pass. Maximum wire-rounding error `4.9671153e-5`.
- Fresh FP32 packaging produces exactly the previous private manifest digest
  `8c8b1aab5340b812d8f099a827833eb0a5a30823a4d990da6ce14945b9a17e35`.

The thresholds remain logits `1e-3`, probabilities `1e-4`, identical decisions.
No golden was removed, rewritten or substituted. CPU runs use eight threads;
CUDA uses available physical GPU 4. Test listeners were stopped and GPU 4
returned to 2 MiB. See the [machine-readable receipt](elephant-1-review-20261106.json)
for source, artifact and log hashes.

## Deployment-skill assessment

The inspected `xd-model-deploy-skills` checkout was clean at `2e4caaa`. No skill
update is needed for this development-only delivery:

- `deploy-ollaya-docker` consumes a fixed published CUDA12 image and requires a
  separate compatibility contract for another image/version/platform/model.
- `verify-ollaya-service` already requires explicit model identities, matching
  substantive inference checks and separate CUDA/residency evidence. It does
  not certify a new model from HTTP 200 alone.
- `deploy-ollaya-decisions` intentionally remains the fixed Clef/Strands asset
  profile. Adding Elephant to it would change that profile's identity.

This fork's private native CUDA13 development prefix is not the published
CUDA12 image; an unchanged version string does not establish compatibility.
When a deployment is requested, the producer first needs a versioned image
digest/platform contract and a matching, distributable Elephant verifier.
Then update the generic consumer snapshot and model expectations, using an
independent asset profile if necessary. Keep builds/native packaging in Ollaya
and actual site facts in xd-infra. Do not duplicate this source/export workflow
inside deployment skills or declare Elephant supported by the older image.

Current capability remains text-only, 4096 tokens per prompt, total request
budget 16384 with the shared prefix counted once. Images, longer prompts,
shared-prefix optimization, other platforms and public quality/performance
release qualification remain unverified.
