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
SNAPSHOT_SHA256 = {
    "config.json": "f26324589d15932e2742dbb13dbc7c561865ce4f80990af3b7dff05efe4dadab",
    "readout.json": "96ea8a5c6ecfc30c20cdd369762db5997c5f4fb9821ce0a0318941343eb5e8ba",
    "tokenizer.json": "a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f",
    "tokenizer_config.json": "2984c03425a28d242744e234a5003786fd289b3713db79fe0958c65eb59d85b2",
    "chat_template.jinja": "a4aee8afcf2e0711942cf848899be66016f8d14a889ff9ede07bca099c28f715",
    "generation_config.json": "290866a6d7d4a4442fe83ba2434a31517446f40a01de500ff97b4354e7e5d082",
    "serve.patch": "f234c125f1d38d82104b9e1f24e75bde51e4401928acfba2fbf5c59ef2fb46b8",
}
REFERENCE_SHA256 = {
    "__init__.py": "6446cd0006940630b28180d275efe39972b871c2fb2ae502c848607c940e8376",
    "client.py": "560ca00f7ecffee72e6490afdd69a34cf4f55a6e1765a0493d4beebf8b6b4278",
    "engine.py": "dc3eb95e0118cb056357bd3e68c0ed9f18b4656492b02087e9ca53773af8d3f0",
    "prompt.py": "f0763b5f747039c2919bc2a6d51e4ed88308fe3497e53498e6f298286b6169b2",
    "readout.py": "e084edead750ddeb39384b9f659182df52720d96294d9a95c9259a8001e3c20d",
    "answers.py": "09b06dba258c80a6a9afdaa5aacfd0f7523f9d2a087a7b6402ca03553ea6fba6",
    "calibrate.py": "3fa6013a81c7e79a702afcdcc0ebd089bee7047590c18cc44708850f5aeda1ee",
    "schema.py": "bce4b7d353ac115a9e50aef284975ddeed9a3bb452e3fd928a556668705bc2f3",
}


def verify_files(root, expected):
    for filename, digest in expected.items():
        with (Path(root) / filename).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"pinned file SHA256 mismatch: {filename}")


def verify_snapshot(model_dir, weights=False):
    """Weights, configuration and prompt artifacts jointly define this checkpoint."""
    verify_files(model_dir, SNAPSHOT_SHA256)
    for name in ("calibration.json", "readout_head.pt", "adapter_config.json"):
        if (Path(model_dir) / name).exists():
            raise ValueError(f"unexpected checkpoint override: {name}")
    if weights:
        verify_files(model_dir, {"model.safetensors": WEIGHT_SHA256})


def verify_reference(path):
    root = Path(path).resolve() / "qev"
    verify_files(root, REFERENCE_SHA256)
    # sys.path does not replace previously imported modules. Refuse a cached
    # package from another checkout before it can silently supply the oracle.
    for name, module in tuple(sys.modules.items()):
        if name == "qev" or name.startswith("qev."):
            origin = getattr(module, "__file__", None)
            if origin is None or not Path(origin).resolve().is_relative_to(root):
                raise ValueError(f"reference module was imported from another tree: {name}")


@lru_cache(maxsize=1)
def upstream():
    path = Path(os.environ["ELEPHANT_ONEJEV"]).resolve()
    verify_reference(path)
    sys.path.insert(0, str(path))
    from qev.engine import DecisionEngine
    from qev.prompt import render_question
    from qev.schema import SystemOneRequest
    verify_reference(path)
    return DecisionEngine, render_question, SystemOneRequest


def load(model_dir, device="cpu", dtype="float32"):
    verify_snapshot(model_dir, weights=True)
    import transformers
    if transformers.__version__ != "5.19.0":
        raise ValueError("author reference requires Transformers 5.19.0 in its isolated build environment")
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
