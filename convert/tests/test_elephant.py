"""Mask and hybrid-trunk tests use random tiny weights, never downloaded model outputs."""
import json
from pathlib import Path
import unittest

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from ollaya_convert.families.elephant.graphs import ElephantGraph, block_bias
from ollaya_convert.families.elephant.layout import suffix


class ElephantTests(unittest.TestCase):
    def test_author_layout_fixtures(self):
        data = json.loads((Path(__file__).parent/"fixtures/elephant-layout.json").read_text())
        for case in data["cases"]:
            self.assertEqual([suffix(q)[0] for q in case["questions"].values()],case["suffixes"])

    def test_visibility_excludes_suffix_from_prefix_and_padding_from_all(self):
        bias=block_bias(torch.zeros(2,64,dtype=torch.long),torch.tensor([9,19]),torch.tensor([4,8]))
        self.assertTrue(torch.isfinite(bias[0,0,:4,:4]).all())
        self.assertTrue(torch.isneginf(bias[0,0,:4,4:]).all())
        self.assertTrue(torch.isfinite(bias[0,0,4:10,:10]).all())
        self.assertTrue(torch.isneginf(bias[0,0,:,10:]).all())
        self.assertTrue(torch.isneginf(bias[1,0,:,20:]).all())

    def test_hybrid_graph_against_hf_explicit_block_mask(self):
        torch.manual_seed(42)
        cfg=Qwen3_5TextConfig(vocab_size=512,hidden_size=64,intermediate_size=128,num_hidden_layers=4,
            num_attention_heads=4,num_key_value_heads=2,head_dim=16,
            linear_key_head_dim=16,linear_value_head_dim=16,linear_num_key_heads=2,linear_num_value_heads=4,
            rope_parameters={"rope_type":"default","rope_theta":10000.0,"partial_rotary_factor":0.5,"mrope_section":[1,1,2]})
        cfg._attn_implementation="eager"
        model=Qwen3_5ForCausalLM(cfg).eval()
        graph=ElephantGraph(model.model,model.lm_head,list(range(256))).eval()
        ids=torch.randint(0,512,(2,128));last=torch.tensor([95,110]);prefix=torch.tensor([40,64])
        bias=block_bias(ids,last,prefix)
        # HF accepts the full_attention mask independently of its linear mask.
        with torch.inference_mode():
            want=model(input_ids=ids,attention_mask={"full_attention":bias,"linear_attention":None},use_cache=False).logits
            got=graph(ids,last,prefix)
        want=want[torch.arange(2),last,:256]
        torch.testing.assert_close(got,want,atol=1e-5,rtol=1e-5)
        # Right padding cannot change any valid readout.
        ids2=ids.clone();ids2[0,96:]=511;ids2[1,111:]=511
        with torch.inference_mode(): got2=graph(ids2,last,prefix)
        torch.testing.assert_close(got,got2,atol=1e-6,rtol=1e-6)

        # Exercise the real exporter and ORT at another sequence length: catches
        # traced masks or Scan trip counts that an eager-only check cannot find.
        import tempfile
        import onnxruntime as ort
        from ollaya_convert.families.llm_common.onnx_export import export_graph
        with tempfile.TemporaryDirectory() as tmp:
            r=torch.export.Dim("rows",min=1,max=4)
            n=torch.export.Dim("chunks",min=1,max=8)
            dyn={"input_ids":{0:r,1:64*n},"last_pos":{0:r},"prefix_lengths":{0:r}}
            path=str(Path(tmp)/"model.onnx")
            export_graph(graph,(ids,last,prefix),list(dyn),["cand_logits"],dyn,path)
            options=ort.SessionOptions();options.intra_op_num_threads=2;options.enable_mem_reuse=False
            session=ort.InferenceSession(path,options,providers=["CPUExecutionProvider"])
            for t in (64,192):
                x=torch.randint(0,512,(1,t));l=torch.tensor([t-7]);p=torch.tensor([t//2])
                with torch.inference_mode(): expected=graph(x,l,p).numpy()
                actual=session.run(None,{"input_ids":x.numpy(),"last_pos":l.numpy(),"prefix_lengths":p.numpy()})[0]
                import numpy as np
                np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)


if __name__=="__main__": unittest.main()
