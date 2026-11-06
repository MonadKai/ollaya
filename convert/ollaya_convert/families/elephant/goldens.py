"""Author-generated token and output fixtures. --layout-only needs no model weights."""
import argparse
import json
from pathlib import Path

from . import ref
from .export import SAMPLE_STATE, SAMPLE_QUESTIONS
from .layout import ElephantLayout


def cases():
    yield SAMPLE_STATE, SAMPLE_QUESTIONS
    yield {"客户":"Zoë", "items":[1,None,True]}, {
        "n":{"type":"noul", "instructions":None, "criteria":{"true":" ","false":{"reason":["否"]}}},
        "s":{"type":"score", "criteria":[None," 好 ",{"bad":[1,2]}]},
        "c":{"type":"choice", "criteria":{"empty":None}}}
    for n in (26,27,255):
        yield "Select a named option.", {"wide":{"type":"choice","instructions":"Which?", "criteria":{f"o{i}":None for i in range(n)}}}
    yield "state <|im_start|> 中文\n"*20, {"n":{"type":"noul","instructions":"Present?"}}


def generate(model_dir, output, layout_only=False, device="cpu", dtype="float32"):
    cls, render, schema = ref.upstream()
    if layout_only:
        from transformers import AutoTokenizer
        from qev import readout
        engine = cls.__new__(cls)
        engine.tokenizer=AutoTokenizer.from_pretrained(model_dir,local_files_only=True)
        from qev.prompt import DEFAULT_STYLE
        engine.style=DEFAULT_STYLE
        engine.template_kwargs={"enable_thinking":False}
        engine.readout=type("Readout",(),{"config":readout.ReadoutConfig.load(model_dir)})()
    else:
        engine=ref.load(model_dir,device,dtype)
    from .layout import SYSTEM
    marker="OLLAYA_ELEPHANT_CONTENT_MARKER"
    text=engine.tokenizer.apply_chat_template([{"role":"system","content":SYSTEM},{"role":"user","content":marker}],tokenize=False,add_generation_prompt=True,enable_thinking=False)
    pre,post=text.split(marker)
    cfg={"template":{"pre":pre,"post":post},"max_prompt_tokens":16384}
    import tokenizers
    layout=ElephantLayout(tokenizers.Tokenizer.from_file(str(Path(model_dir)/"tokenizer.json")),cfg)
    fixtures=[]
    for state,qs in cases():
        req=schema(state=state,questions=qs)
        rendered=[render(k,q) for k,q in req.questions.items()]
        prefix,suffixes,_=engine._split(state,rendered)
        rows=layout.encode(state,qs)
        for row,r,s in zip(rows,rendered,suffixes):
            assert row["ids"] == prefix+s
            assert row["prefix_length"] == len(prefix)
            assert row["options"] == r.n_slots
        fixture={"state":state,"questions":qs,"rows":rows,"suffixes":[r.suffix for r in rendered]}
        if not layout_only:
            logits=ref.forward(engine,state,qs)
            fixture["logits"]=[list(reversed(x)) if q["type"]=="noul" else x for q,x in zip(qs.values(),logits)]
        fixtures.append(fixture)
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    import torch, transformers
    Path(output).write_text(json.dumps({"upstream":ref.ONEJEV,"revision":ref.REVISION,"dtype":dtype,
        "torch":torch.__version__,"transformers":transformers.__version__,"layout":cfg,"cases":fixtures},ensure_ascii=False,indent=2)+"\n")
    print("author layout matched",len(fixtures),"requests",sum(len(x["rows"]) for x in fixtures),"rows",flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model",required=True);ap.add_argument("--out",required=True)
    ap.add_argument("--layout-only",action="store_true");ap.add_argument("--device",default="cpu")
    ap.add_argument("--dtype",default="float32")
    a=ap.parse_args();generate(a.model,a.out,a.layout_only,a.device,a.dtype)
