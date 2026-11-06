"""Private-tag admission must reject unqualified precision before packaging."""
import json
import hashlib
from pathlib import Path
import tempfile
import unittest
import types
from unittest.mock import patch

from ollaya_convert.families.elephant import package_local, ref


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


if __name__ == "__main__":
    unittest.main()
