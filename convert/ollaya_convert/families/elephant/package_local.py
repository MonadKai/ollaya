"""Package the fork's Elephant text artifact into a private test registry.

python -m ollaya_convert.families.elephant.package_local --model out/elephant-1-wl \
    --registry /path/to/test-registry --origin http://127.0.0.1:8899

Run author/runtime parity first. This does not publish a public catalog tag or
upload anything; weights and tokenizer URLs remain the author's pinned files.
"""
import argparse
import json
from pathlib import Path

from . import ref


def build(model_dir, registry, origin):
    from ... import package
    from ...catalog import LICENSE_APACHE, _wl

    model_dir=Path(model_dir).resolve()
    decision=json.loads((model_dir/"decision.json").read_text())
    if (decision.get("engine"),decision.get("layout"))!=("onnx","elephant-1-v1"):
        raise ValueError("expected an Elephant-1 ONNX export")
    source=decision.get("upstream",{})
    if (source.get("repo"),source.get("revision"))!=(ref.REPO,ref.REVISION):
        raise ValueError("artifact does not name the pinned author checkpoint")
    package.REGISTRY=str(Path(registry).resolve())
    spec={"namespace":"library","model":"elephant","family":"elephant","author":"Base Labs",
          "license":"Apache-2.0",
          "license_text":"Elephant-1 by Base Labs (https://huggingface.co/baselabs/elephant-1).\n\n"+LICENSE_APACHE,
          "tags":{"1-text":_wl("elephant-1",ref.REPO,ref.REVISION,
              "Elephant-1 text-only decision model on Qwen3.5-9B. Original author weights; "
              "block-bidirectional full attention and causal Gated DeltaNet. Fork development tag.",
              "9B",decision["max_prompt_tokens"],["en"],wl_dir=str(model_dir))},"aliases":{}}
    package.package_model(spec,package.Blobs(origin))
    print("private registry:",package.REGISTRY,"model: elephant:1-text",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",required=True);ap.add_argument("--registry",required=True)
    ap.add_argument("--origin",required=True)
    args=ap.parse_args();build(args.model,args.registry,args.origin)
