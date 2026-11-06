"""Check a locally packaged Elephant tag through the native daemon's public APIs.

Run the strict raw-logit parity gate first. This checks the daemon's existing
four-decimal wire transformations, not a different model tolerance.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import urllib.error
import urllib.request

from .layout import suffix
from .checks import load_goldens


DEFAULTS = {
    "noul": "Is the statement true, or is the answer to the question yes?",
    "choice": "Which option applies to the state?",
    "score": "Which level describes the state?",
}


def check(base, model, goldens, out, leave_loaded=False, evidence=None, keep_alive="5m", expected_version=None):
    trace = None if evidence is None else Path(evidence)
    if trace is not None:
        trace.mkdir(parents=True, exist_ok=False)
    transaction = 0
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def call(path, body=None):
        nonlocal transaction
        transaction += 1
        req = urllib.request.Request(
            base.rstrip("/") + path,
            data=None if body is None else json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with opener.open(req, timeout=600) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as error:
            status, raw = error.code, error.read()
        if trace is not None:
            (trace / f"{transaction:03d}-request.json").write_text(json.dumps({"path": path, "body": body}, ensure_ascii=False, indent=2) + "\n")
            (trace / f"{transaction:03d}-response.json").write_bytes(raw)
            (trace / f"{transaction:03d}-status.json").write_text(json.dumps({"status": status}) + "\n")
        return status, json.loads(raw)

    data = load_goldens(goldens)
    maximum = 0.0
    rows = 0
    rejections = []
    schema_rejections = 0
    record = None

    def compare(answers, case):
        nonlocal maximum, rows
        assert list(answers) == list(case["questions"]), answers
        for (qid, question), logits in zip(case["questions"].items(), case["logits"]):
            exp = [math.exp(x - max(logits)) for x in logits]
            probs = [x / sum(exp) for x in exp]
            answer = answers[qid]
            assert answer["type"] == question["type"], answer
            if question["type"] == "noul":
                error = abs(answer["noul"] - probs[1])
            else:
                keys = list(question["criteria"]) if question["type"] == "choice" else [str(i) for i in range(len(probs))]
                assert list(answer["probabilities"]) == keys, answer
                error = max(abs(answer["probabilities"][key] - p) for key, p in zip(keys, probs))
                confidence = 1.0 if len(probs) == 1 else (len(probs) * max(probs) - 1) / (len(probs) - 1)
                error = max(error, abs(answer["confidence"] - confidence))
                if question["type"] == "choice":
                    assert answer["choice"] == keys[max(range(len(probs)), key=probs.__getitem__)], answer
                else:
                    error = max(error, abs(answer["score"] - sum(i * p for i, p in enumerate(probs))))
            # 0.00005 is decimal rounding; raw probability error has its own strict gate.
            assert error <= 0.00006, (qid, error, answer)
            maximum = max(maximum, error)
            rows += 1

    try:
        if expected_version is not None:
            status, version = call("/api/version")
            assert status == 200 and version["version"] == expected_version, (status, version)
        status, show = call("/api/show", {"model": model})
        assert status == 200 and show["details"]["family"] == "elephant", show
        assert show["model_info"]["elephant.layout"] == "elephant-1-v1", show
        assert set(show["capabilities"]) == {"choice", "score", "noul"}, show
        for case in data["cases"]:
            case = copy.deepcopy(case)
            questions = copy.deepcopy(case["questions"])
            if any(q["type"] == "choice" and len(q["criteria"]) == 1 for q in questions.values()):
                # Author/runner accepts one option and null score descriptions;
                # the existing public TypeSafe schema intentionally does not.
                status, result = call("/api/decide", {"model": model, "state": case["state"],
                                                      "questions": questions, "stream": False})
                assert status == 422 and result["code"] == "INVALID_REQUEST", (status, result)
                schema_rejections += 1
                keep = [i for i, q in enumerate(questions.values()) if not (q["type"] == "choice" and len(q["criteria"]) == 1)]
                case["questions"] = {key: q for i, (key, q) in enumerate(questions.items()) if i in keep}
                case["rows"] = [case["rows"][i] for i in keep]
                case["logits"] = [case["logits"][i] for i in keep]
                questions = copy.deepcopy(case["questions"])
            # Generic API defaults missing instructions to qid. Supply the author's
            # wording explicitly to keep these requests identical to the goldens.
            for question in questions.values():
                value = question.get("instructions")
                if value is None or isinstance(value, str) and not value.strip():
                    question["instructions"] = DEFAULTS[question["type"]]
                if question["type"] == "score":
                    question["criteria"] = ["" if v is None else v for v in question["criteria"]]
            for key, question in questions.items():
                assert suffix(question) == suffix(case["questions"][key]), key
            body = {"model": model, "state": case["state"], "questions": questions}
            status, result = call("/api/decide", {**body, "stream": False, "keep_alive": keep_alive})
            assert status == 200 and not result["state_truncated"], (status, result)
            expected = sum(len(row["ids"]) - row["prefix_length"] for row in case["rows"]) + case["rows"][0]["prefix_length"]
            assert result["usage"]["input_tokens"] == expected, result
            compare(result["answers"], case)
            if any(q["type"] == "score" for q in questions.values()):
                status, result = call("/v1/systemone", body)
                assert status == 200, (status, result)
                compare(result["answers"], case)

        valid = {"q": {"type": "noul", "instructions": "Is anything stated?"}}
        for body in [
            {"state": "x", "questions": valid, "images": ["AA=="]},
            {"state": "x", "questions": {"q": {"type": "score", "criteria": []}}},
            {"state": "word " * 20000, "questions": valid},
            {"state": "QEV_PREFIX_MARK", "questions": valid},
        ]:
            status, result = call("/api/decide", {"model": model, "stream": False, "keep_alive": keep_alive, **body})
            assert status == 422 and "error" in result, (status, result)
            rejections.append(status)
        record = {"model": model, "golden_rows": sum(len(c["rows"]) for c in data["cases"]),
                  "wire_rows_checked": rows, "max_wire_rounding_error": maximum,
                  "rejections": rejections, "author_only_schema_rejections": schema_rejections,
                  "model_left_loaded": leave_loaded, "expected_version": expected_version, "passed": True}
    finally:
        if not leave_loaded or record is None:
            status, result = call("/api/decide", {"model": model, "keep_alive": 0})
            assert status == 200 and result["done_reason"] == "unload", (status, result)
    Path(out).write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="http://127.0.0.1:11439")
    parser.add_argument("--model", default="elephant:1-text")
    parser.add_argument("--goldens", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--leave-loaded", action="store_true", help="retain the verified model for separate residency observation")
    parser.add_argument("--evidence", help="new directory for raw requests/responses/status codes")
    parser.add_argument("--keep-alive", default="5m")
    parser.add_argument("--expected-version")
    args = parser.parse_args()
    check(args.base, args.model, args.goldens, args.out, args.leave_loaded, args.evidence, args.keep_alive, args.expected_version)
