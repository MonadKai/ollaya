"""Admission for full-output Elephant goldens, before starting runtime work."""
import json
import math
from pathlib import Path


def load_goldens(path):
    data = json.loads(Path(path).read_text())
    cases = data.get("cases") if isinstance(data, dict) else None
    if not isinstance(cases, list) or not cases:
        raise ValueError("goldens require nonempty cases")
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise ValueError(f"golden case {index} must be an object")
        questions, rows, logits = (case.get(key) for key in ("questions", "rows", "logits"))
        if not isinstance(questions, dict) or not questions:
            raise ValueError(f"golden case {index} requires questions")
        if not isinstance(rows, list) or not isinstance(logits, list) or not len(questions) == len(rows) == len(logits):
            raise ValueError(f"golden case {index} questions/rows/logits counts differ")
        for (qid, question), row, values in zip(questions.items(), rows, logits):
            if not isinstance(question, dict) or not isinstance(row, dict):
                raise ValueError(f"golden row {qid} must be an object")
            kind = question.get("type")
            criteria = question.get("criteria")
            if kind == "noul":
                count = 2
            elif kind == "choice" and isinstance(criteria, dict) and 1 <= len(criteria) <= 255:
                count = len(criteria)
            elif kind == "score" and isinstance(criteria, list) and 1 <= len(criteria) <= 10:
                count = len(criteria)
            else:
                raise ValueError(f"invalid golden question {qid}")
            ids, prefix = row.get("ids"), row.get("prefix_length")
            if not isinstance(ids, list) or not ids or any(type(x) is not int or x < 0 for x in ids):
                raise ValueError(f"invalid token IDs in golden row {qid}")
            if type(prefix) is not int or not 0 <= prefix <= len(ids):
                raise ValueError(f"invalid prefix length in golden row {qid}")
            if row.get("qid") != qid or row.get("options") != count or row.get("noul") is not (kind == "noul"):
                raise ValueError(f"golden row {qid} metadata disagrees with its question")
            if not isinstance(values, list) or len(values) != count:
                raise ValueError(f"golden row {qid} logit width differs from its options")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in values):
                raise ValueError(f"golden row {qid} has non-finite or nonnumeric logits")
    return data
