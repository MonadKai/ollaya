"""Export Elephant-1 text using unmodified safetensors; no Python runtime dependency.

ELEPHANT_ONEJEV=/path/to/pinned/patched/OneJev python -m ollaya_convert.families.elephant.export \
    --model /absolute/snapshot --out out/elephant-1
"""
import argparse
import gc
import json
import os
import shutil

import torch
from ...weightless_sharded import safetensors_source
from ..llm_common import onnx_export as ox
from ..llm_common.qwen35 import CHUNK, half_trunk
from . import ref
from .graphs import ElephantGraph
from .layout import ElephantLayout, SLOTS, SYSTEM

SAMPLE_STATE = "The customer was charged twice and requests a refund."
SAMPLE_QUESTIONS = {
    "refund": {"type": "noul", "instructions": "Does the customer want a refund?"},
    "team": {"type": "choice", "instructions": "Which team?", "criteria": {"billing": "payments", "tech": "bugs"}},
}


def rename(name):
    if name.startswith("trunk.m."):
        return ["model.language_model." + name[len("trunk.m."):]]
    return [name]


def export(model_dir, out_dir, fp16=False, max_tokens=4096):
    if not 1 <= max_tokens <= 16384:
        raise ValueError("max_tokens must be within training limit 16384")
    engine = ref.load(model_dir)
    tok = engine.tokenizer
    marker = "OLLAYA_ELEPHANT_CONTENT_MARKER"
    template = tok.apply_chat_template([{"role":"system", "content":SYSTEM}, {"role":"user", "content":marker}],
                                      tokenize=False, add_generation_prompt=True, enable_thinking=False)
    pre, post = template.split(marker)
    cfg = {"engine":"onnx", "family":"elephant", "layout":"elephant-1-v1",
           "template":{"pre":pre,"post":post}, "max_prompt_tokens":max_tokens,"max_request_tokens":16384,
           "upstream":{"repo":ref.REPO,"revision":ref.REVISION,"onejev":ref.ONEJEV},
           "contract":{"seq_multiple":CHUNK,"attention":"block_bidirectional full; causal linear", "text_only":True},
           "special_tokens":{"pad":engine.pad_id},
           "codes":{"strings":SLOTS,"ids":engine.slot_ids},
           "precision":ox.FP16_PRECISION if fp16 else "fp32"}
    import tokenizers
    lay = ElephantLayout(tokenizers.Tokenizer.from_file(os.path.join(model_dir,"tokenizer.json")),cfg)
    rows = lay.encode(SAMPLE_STATE,SAMPLE_QUESTIONS)
    _, prefix, suffixes = ref.prepare(engine,SAMPLE_STATE,SAMPLE_QUESTIONS)
    assert [r["ids"] for r in rows] == [prefix+s for s in suffixes]
    assert all(r["prefix_length"] == len(prefix) for r in rows)
    want = ref.forward(engine,SAMPLE_STATE,SAMPLE_QUESTIONS)
    graph = ElephantGraph(engine.model.model,engine.model.lm_head,engine.slot_ids).eval()
    t = max(len(r["ids"]) for r in rows)
    t = (t+CHUNK-1)//CHUNK*CHUNK
    ids = torch.zeros((2,t),dtype=torch.long)
    for i,r in enumerate(rows): ids[i,:len(r["ids"])] = torch.tensor(r["ids"])
    args = (ids,torch.tensor([len(r["ids"])-1 for r in rows]),torch.tensor([len(prefix)]*2))
    with torch.inference_mode(): got = graph(*args)
    err = max(float((got[i,:len(w)]-torch.tensor(w)).abs().max()) for i,w in enumerate(want))
    print("fp32 eager vs author sequential raw logits:",err,flush=True)
    if err > 1e-3: raise ValueError("eager graph failed author parity")
    if fp16: half_trunk(graph.trunk)
    r = torch.export.Dim("rows",min=1,max=4096)
    n = torch.export.Dim("chunks",min=1,max=256)
    names = ["input_ids","last_pos","prefix_lengths"]
    dyn = {"input_ids":{0:r,1:CHUNK*n},"last_pos":{0:r},"prefix_lengths":{0:r}}
    tmp = ox.scratch_dir("elephant-export-")
    ox.export_graph(graph,args,names,["cand_logits"],dyn,os.path.join(tmp,"model.onnx"))
    del graph,engine,got
    gc.collect()
    weight = os.path.join(model_dir,"model.safetensors")
    source = safetensors_source("model.safetensors",weight,repo=ref.REPO,revision=ref.REVISION,filename="model.safetensors")
    report = ox.weightless(tmp,out_dir,[source],rename)
    cfg["weights_in_memory"] = ox.weights_in_memory(report)
    shutil.copy(os.path.join(model_dir,"tokenizer.json"),out_dir)
    ox.write_json(os.path.join(out_dir,"decision.json"),cfg)
    ox.write_json(os.path.join(out_dir,"calibration.json"),{"temperature":[1.0]*3,"temperature_by_options":{}})
    layers = [{"role":"graph","path":"model.onnx","hosted_by":"ollaya","sha256":ox.sha256_file(os.path.join(out_dir,"model.onnx")),"bytes":os.path.getsize(os.path.join(out_dir,"model.onnx"))},
              ox.file_entry("weights",ref.REPO,ref.REVISION,"model.safetensors",weight,location="model.safetensors"),
              ox.file_entry("tokenizer",ref.REPO,ref.REVISION,"tokenizer.json",os.path.join(model_dir,"tokenizer.json")),
              {"role":"decision","path":"decision.json","hosted_by":"ollaya"},
              {"role":"calibration","path":"calibration.json","hosted_by":"ollaya"}]
    ox.write_json(os.path.join(out_dir,"files.json"),{"model":"elephant-1","layers":layers,"weightless":{k:v for k,v in report.items() if k!="unused"}})
    ox.cleanup(tmp)


if __name__ == "__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",required=True)
    ap.add_argument("--out",required=True)
    ap.add_argument("--fp16",action="store_true")
    ap.add_argument("--max-tokens",type=int,default=4096)
    a=ap.parse_args()
    export(a.model,a.out,a.fp16,a.max_tokens)
