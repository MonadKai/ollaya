"""Verify the isolated runner HTTP protocol against a supplied golden artifact."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import urllib.request
import urllib.error

from .checks import load_goldens


def check(binary, model, goldens, device):
    model=Path(model).resolve()
    data=load_goldens(goldens)
    log=model/"http-runner.log"
    with log.open("a") as stderr:
        proc=subprocess.Popen([str(Path(binary).resolve()),"runner","--graph-fp32",str(model/"model.onnx"),
            "--tokenizer",str(model/"tokenizer.json"),"--decision",str(model/"decision.json"),
            "--device",device,"--threads","4"],stdout=subprocess.PIPE,stderr=stderr,text=True)
        try:
            line=proc.stdout.readline()
            if not line: raise RuntimeError(f"runner failed readiness; inspect {log}")
            loaded=json.loads(line)
            base=f"http://127.0.0.1:{loaded['port']}"
            def post(body):
                req=urllib.request.Request(base+"/decide",data=json.dumps(body).encode(),headers={"Content-Type":"application/json"})
                try:
                    with urllib.request.urlopen(req,timeout=600) as r: return r.status,json.load(r)
                except urllib.error.HTTPError as e: return e.code,json.load(e)
            maximum=0.0;count=0
            for case in data["cases"]:
                status,result=post({"state":case["state"],"questions":case["questions"]})
                assert status==200,(status,result)
                assert len(result["questions"])==len(case["logits"])
                for row,want in zip(result["questions"],case["logits"]):
                    assert len(row["logits"])==len(want)
                    assert all(math.isfinite(value) for value in row["logits"]), row
                    maximum=max(maximum,max(abs(a-b) for a,b in zip(row["logits"],want)))
                    count+=1
                assert not result["state_truncated"]
                expected=sum(len(r["ids"])-r["prefix_length"] for r in case["rows"])+case["rows"][0]["prefix_length"]
                assert result["input_tokens"]==expected
            assert maximum<=1e-3,maximum
            valid={"q":{"type":"noul","instructions":"Is anything stated?"}}
            for body in [
                {"state":"x","questions":valid,"images":["AA=="]},
                {"state":"x","questions":{"q":{"type":"score","criteria":[]}}},
                {"state":"word "*20000,"questions":valid},
                {"state":"QEV_PREFIX_MARK","questions":valid},
            ]:
                status,result=post(body)
                assert status==400,(status,result)
                assert "error" in result,result
            record={"reference":data.get("reference","author Elephant-1"),"device":loaded["device"],
                    "rows":count,"max_logit_error":maximum,"rejections":4,"passed":True}
        finally:
            proc.terminate()
            try: proc.wait(timeout=20)
            except subprocess.TimeoutExpired: proc.kill();proc.wait()
    (model/"http-result.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(record),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--binary",required=True);ap.add_argument("--model",required=True)
    ap.add_argument("--goldens",required=True);ap.add_argument("--device",default="cpu")
    a=ap.parse_args();check(a.binary,a.model,a.goldens,a.device)
