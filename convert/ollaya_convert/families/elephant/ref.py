"""Author runtime wrapper. Set ELEPHANT_ONEJEV to the pinned, serve.patch-applied source tree.

The source is build-time only; never a dependency of the Rust runner.
"""
import os
import sys
import hashlib
from functools import lru_cache
from pathlib import Path

REPO = "baselabs/elephant-1"
REVISION = "b3f8a93462fb1051a9de031e94a7ff380e254072"
ONEJEV = "81ce62f1597c91e46767d4d02d6ac2e18534fe94"
WEIGHT_SHA256 = "682f02348dfe4617dd21bba2915c12e805de8cf16327a61e895a831ea681d0ba"


@lru_cache(maxsize=1)
def upstream():
    path = Path(os.environ["ELEPHANT_ONEJEV"]).resolve()
    expected = {
        "engine.py":"dc3eb95e0118cb056357bd3e68c0ed9f18b4656492b02087e9ca53773af8d3f0",
        "prompt.py":"f0763b5f747039c2919bc2a6d51e4ed88308fe3497e53498e6f298286b6169b2",
        "readout.py":"e084edead750ddeb39384b9f659182df52720d96294d9a95c9259a8001e3c20d",
    }
    for filename,digest in expected.items():
        if hashlib.sha256((path/"qev"/filename).read_bytes()).hexdigest() != digest:
            raise ValueError(f"reference is not the pinned OneJev + serve.patch: {filename}")
    sys.path.insert(0, str(path))
    from qev.engine import DecisionEngine
    from qev.prompt import render_question
    from qev.schema import SystemOneRequest
    return DecisionEngine, render_question, SystemOneRequest


def load(model_dir, device="cpu", dtype="float32"):
    cls, _, _ = upstream()
    engine = cls(str(Path(model_dir).resolve()), device=device, dtype=dtype,
                 fork_mode="sequential", max_branch_tokens=16384, max_request_tokens=16384)
    assert engine.info.extra["attention"] == "noncausal"
    assert engine.info.extra["readout"] == "slot_lm"
    assert engine.info.extra["calibration"] == "none"
    assert len(engine.slot_ids) == 256
    # The checkpoint is multimodal, the reference class text-only. Prove the
    # loader's prefix conversion loaded every text parameter, not random init.
    import torch
    from safetensors import safe_open
    with safe_open(str(Path(model_dir)/"model.safetensors"),framework="pt",device="cpu") as weights:
        for name, param in engine.model.named_parameters():
            key = "model.language_model."+name[len("model."):] if name.startswith("model.") else name
            original=weights.get_tensor(key).to(dtype=param.dtype)
            if not torch.equal(param.detach().cpu(),original):
                raise ValueError(f"loaded parameter differs from author checkpoint: {name}")
    return engine


def prepare(engine, state, questions):
    _, render, schema = upstream()
    req = schema(state=state, questions=questions)
    rendered = [render(k, q, style=engine.style) for k, q in req.questions.items()]
    prefix, suffixes, _ = engine._split(state, rendered)
    if len(prefix)+max(map(len,suffixes)) > engine.max_branch_tokens:
        raise ValueError("author branch token limit exceeded")
    if len(prefix)+sum(map(len,suffixes)) > engine.max_request_tokens:
        raise ValueError("author request token limit exceeded")
    return rendered, prefix, suffixes


def forward(engine, state, questions):
    rendered, _, _ = prepare(engine, state, questions)
    return [b.logits for b in engine._run_sequential(state, rendered)]
