"""Check author block-mask installation and cached fork on a random tiny model.

Requires the pinned source via ELEPHANT_ONEJEV, and the real tokenizer only.
This tests reference mechanics independently of the 18.82 GB checkpoint.
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3_5ForCausalLM, Qwen3_5TextConfig
from . import ref
from .goldens import cases
from .graphs import ElephantGraph
from .layout import SLOTS


def check(snapshot):
    torch.set_num_threads(4);torch.manual_seed(42)
    cls,_,_=ref.upstream()
    from qev import readout
    from qev.prompt import DEFAULT_STYLE
    e=cls.__new__(cls)
    e.tokenizer=AutoTokenizer.from_pretrained(snapshot,local_files_only=True)
    cfg=Qwen3_5TextConfig(vocab_size=len(e.tokenizer),hidden_size=64,intermediate_size=128,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_key_head_dim=16,linear_value_head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=4,
        rope_parameters={"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.5,"mrope_section":[1,1,2]})
    cfg._attn_implementation="eager"
    e.model=Qwen3_5ForCausalLM(cfg).eval()
    e.slot_ids=[e.tokenizer.encode(s,add_special_tokens=False)[0] for s in SLOTS]
    e.device=torch.device("cpu");e.style=DEFAULT_STYLE;e.template_kwargs={"enable_thinking":False}
    e.max_branch_tokens=16384;e.max_request_tokens=16384
    e.readout=readout.Readout.attach(e.model,readout.ReadoutConfig.load(snapshot),slot_ids=e.slot_ids)
    graph=ElephantGraph(e.model.model,e.model.lm_head,e.slot_ids).eval()
    maximum=0.0;count=0
    for state,questions in cases():
        want=ref.forward(e,state,questions)
        _,prefix,suffixes=ref.prepare(e,state,questions)
        for values,suffix in zip(want,suffixes):
            ids=prefix+suffix;t=(len(ids)+63)//64*64
            x=torch.zeros((1,t),dtype=torch.long);x[0,:len(ids)]=torch.tensor(ids)
            with torch.inference_mode(): got=graph(x,torch.tensor([len(ids)-1]),torch.tensor([len(prefix)]))[0,:len(values)]
            maximum=max(maximum,float((got-torch.tensor(values)).abs().max()));count+=1
    record={"reference":"pinned author sequential fork on random tiny model","rows":count,"max_logit_error":maximum}
    print(json.dumps(record),flush=True)
    if maximum>1e-5: raise ValueError("author cached fork differs from full-sequence graph")


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--model",required=True)
    check(ap.parse_args().model)
