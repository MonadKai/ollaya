"""Build a tiny random-weight artifact to verify Rust/ORT wiring, never model accuracy.

python -m ollaya_convert.families.elephant.smoke --model SNAPSHOT --out /tmp/elephant-smoke
"""
import argparse
import json
from pathlib import Path
import shutil

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig
from ..llm_common.onnx_export import export_graph
from .graphs import ElephantGraph, block_bias
from .layout import SLOTS


def build(snapshot, out):
    import tokenizers
    torch.manual_seed(42)
    torch.set_num_threads(4)
    tok=tokenizers.Tokenizer.from_file(str(Path(snapshot)/"tokenizer.json"))
    slots=[tok.encode(s,add_special_tokens=False).ids[0] for s in SLOTS]
    cfg=Qwen3_5TextConfig(vocab_size=tok.get_vocab_size(),hidden_size=64,intermediate_size=128,num_hidden_layers=4,
        num_attention_heads=4,num_key_value_heads=2,head_dim=16,linear_key_head_dim=16,linear_value_head_dim=16,
        linear_num_key_heads=2,linear_num_value_heads=4,
        rope_parameters={"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.5,"mrope_section":[1,1,2]})
    cfg._attn_implementation="eager"
    model=Qwen3_5ForCausalLM(cfg).eval()
    graph=ElephantGraph(model.model,model.lm_head,slots).eval()
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    ids=torch.randint(0,cfg.vocab_size,(2,128));last=torch.tensor([100,110]);prefix=torch.tensor([40,50])
    r=torch.export.Dim("rows",min=1,max=4096);n=torch.export.Dim("chunks",min=1,max=256)
    dyn={"input_ids":{0:r,1:64*n},"last_pos":{0:r},"prefix_lengths":{0:r}}
    export_graph(graph,(ids,last,prefix),list(dyn),["cand_logits"],dyn,str(out/"model.onnx"))
    fixture=Path(__file__).parents[3]/"tests/fixtures/elephant-layout.json"
    data=json.loads(fixture.read_text())
    decision={**data["layout"],"engine":"onnx","family":"elephant","layout":"elephant-1-v1",
              "contract":{"seq_multiple":64},"special_tokens":{"pad":0},"codes":{"strings":SLOTS,"ids":slots}}
    (out/"decision.json").write_text(json.dumps(decision))
    (out/"calibration.json").write_text(json.dumps({"temperature":[1.0]*3,"temperature_by_options":{}}))
    shutil.copy(Path(snapshot)/"tokenizer.json",out/"tokenizer.json")
    for case in data["cases"]:
        logits=[]
        for row in case["rows"]:
            ids=torch.tensor([row["ids"]]);last=torch.tensor([ids.shape[1]-1]);prefix=torch.tensor([row["prefix_length"]])
            with torch.inference_mode():
                result=model(input_ids=ids,attention_mask={"full_attention":block_bias(ids,last,prefix),"linear_attention":None},
                             use_cache=False,logits_to_keep=1).logits[0,0,slots[:row["options"]]].tolist()
            logits.append(list(reversed(result)) if row["noul"] else result)
        case["logits"]=logits
    data["reference"]="SYNTHETIC RANDOM TINY MODEL: HF eager block mask, not Elephant-1 weights"
    (out/"synthetic-goldens.json").write_text(json.dumps(data))
    print("synthetic artifact:",out,flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--model",required=True);ap.add_argument("--out",required=True)
    a=ap.parse_args();build(a.model,a.out)
