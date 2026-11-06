"""Private-tag admission must reject unqualified precision before packaging."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch

from ollaya_convert.families.elephant import package_local, ref
from ollaya_convert.families.elephant.checks import load_goldens
from ollaya_convert.families.elephant import daemon_smoke, http_smoke


class ElephantPackagingTests(unittest.TestCase):
    def test_reduced_or_unknown_precision_cannot_be_packaged_as_fp32(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for precision in ("fp16 compute", "fp16", None):
                decision = {"engine": "onnx", "layout": "elephant-1-v1",
                            "precision": precision, "max_prompt_tokens": 4096,
                            "upstream": {"repo": ref.REPO, "revision": ref.REVISION}}
                (root / "decision.json").write_text(json.dumps(decision))
                with patch("ollaya_convert.package.package_model") as package:
                    with self.assertRaisesRegex(ValueError, "requires fp32"):
                        package_local.build(root, root / "registry", "http://127.0.0.1:18899")
                    package.assert_not_called()


class ElephantReferenceIdentityTests(unittest.TestCase):
    def test_semantic_drift_fails_before_loading_the_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = root / "config.json"
            config.write_bytes(b'{"rope_theta":10000}')
            expected = {"config.json": hashlib.sha256(config.read_bytes()).hexdigest()}
            with patch.dict(ref.SNAPSHOT_SHA256, expected, clear=True):
                ref.verify_snapshot(root)
                config.write_bytes(b'{"rope_theta":10}')
                with self.assertRaisesRegex(ValueError, "config.json"):
                    ref.verify_snapshot(root)

    def test_an_injected_calibration_is_not_the_pinned_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "calibration.json").write_text('{"temperature":2}')
            with patch.dict(ref.SNAPSHOT_SHA256, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "unexpected checkpoint override"):
                    ref.verify_snapshot(root)

    def test_preimported_foreign_reference_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            module = types.ModuleType("qev")
            module.__file__ = "/another-checkout/qev/__init__.py"
            with patch.dict(ref.REFERENCE_SHA256, {}, clear=True), patch.dict(ref.sys.modules, {"qev": module}):
                with self.assertRaisesRegex(ValueError, "another tree"):
                    ref.verify_reference(tmp)


class ElephantGoldenAdmissionTests(unittest.TestCase):
    def valid(self):
        return {"cases": [{"state": "test", "questions": {"q": {"type": "noul"}},
                           "rows": [{"qid": "q", "ids": [1, 2], "prefix_length": 1, "options": 2, "noul": True}],
                           "logits": [[0.0, 1.0]]}]}

    def test_complete_goldens_are_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "goldens.json"
            path.write_text(json.dumps(self.valid()))
            self.assertEqual(load_goldens(path), self.valid())

    def test_corrupt_or_empty_goldens_fail_before_runtime_work(self):
        corrupt = [{"cases": []}, {"cases": [None]}]
        for field, value in [("logits", []), ("rows", []), ("questions", {}),
                             ("logits", [[0.0]]), ("logits", [[float("nan"), 1.0]]),
                             ("logits", [[float("inf"), 1.0]])]:
            data = self.valid()
            data["cases"][0][field] = value
            corrupt.append(data)
        data = self.valid()
        data["cases"][0]["rows"][0]["qid"] = "another-question"
        corrupt.append(data)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "goldens.json"
            for data in corrupt:
                with self.subTest(data=data):
                    path.write_text(json.dumps(data))
                    with patch("urllib.request.urlopen") as network, patch("subprocess.Popen") as runner:
                        with self.assertRaises(ValueError):
                            daemon_smoke.check("http://127.0.0.1:1", "elephant:1-text", path, root / "result.json")
                        with self.assertRaises(ValueError):
                            http_smoke.check("/no/binary", root, path, "cpu")
                        network.assert_not_called()
                        runner.assert_not_called()
                        self.assertFalse((root / "result.json").exists())

    def test_failed_unload_cannot_publish_a_pass(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "goldens.json"
            path.write_text(json.dumps(self.valid()))
            def response(request, timeout):
                result = {"details": {"family": "elephant"}, "model_info": {"elephant.layout": "elephant-1-v1"},
                          "capabilities": ["choice", "score", "noul"]}
                if request.full_url.endswith("/api/decide"):
                    body = json.loads(request.data)
                    if "state" not in body:
                        result = {"done_reason": "load"}  # unload must be acknowledged
                    elif body["state"] == "test":
                        result = {"state_truncated": False, "usage": {"input_tokens": 2},
                                  "answers": {"q": {"type": "noul", "noul": 0.7311}}}
                    else:
                        import io
                        import urllib.error
                        raise urllib.error.HTTPError(request.full_url, 422, "refused", {}, io.BytesIO(b'{"error":"refused"}'))
                import io
                from unittest.mock import MagicMock
                value = MagicMock()
                value.__enter__.return_value = io.StringIO(json.dumps(result))
                value.__enter__.return_value.status = 200
                return value
            with patch("urllib.request.urlopen", side_effect=response):
                with self.assertRaises(AssertionError):
                    daemon_smoke.check("http://127.0.0.1:1", "elephant:1-text", path, root / "result.json")
            self.assertFalse((root / "result.json").exists())


if __name__ == "__main__":
    unittest.main()
