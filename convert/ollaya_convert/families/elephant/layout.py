"""Independent port of OneJev 81ce62f + Elephant serve.patch's default prompt."""
import json

SYSTEM = ("Apply the question to the state. Choose exactly one of the listed options. "
          "Respond with only its uppercase letter, with no explanation or reasoning.")
SKIP = set("BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ".split())
LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SLOTS = list(LETTERS) + [a+b for a in LETTERS for b in LETTERS if a+b not in SKIP][:230]


def entry(v):
    if v is None:
        return ""
    if isinstance(v, str):
        return v.strip()
    if not isinstance(v, (dict, list)):
        raise ValueError("instructions and criteria entries must be strings, objects, arrays or null")
    return json.dumps(v, ensure_ascii=False, indent=2)


def suffix(q):
    ins = entry(q.get("instructions"))
    c = q.get("criteria")
    kind = q.get("type")
    if kind == "noul":
        if c is not None and not isinstance(c, dict):
            raise ValueError("noul criteria must be an object or null")
        c = c or {}
        ins = ins or "Is the statement true, or is the answer to the question yes?"
        names = ["yes", "no"]
        descs = [entry(c["true"]) if c.get("true") is not None else "the statement is true / the answer is yes",
                 entry(c["false"]) if c.get("false") is not None else "the statement is false / the answer is no"]
    elif kind == "choice":
        if not isinstance(c, dict) or not 1 <= len(c) <= 255:
            raise ValueError("choice requires 1..255 named options")
        ins = ins or "Which option applies to the state?"
        names, descs = list(c), [entry(x) for x in c.values()]
    elif kind == "score":
        if not isinstance(c, list) or not 1 <= len(c) <= 10:
            raise ValueError("score requires 1..10 levels")
        ins = ins or "Which level describes the state?"
        names, descs = [f"level {i}" for i in range(len(c))], [entry(x) for x in c]
    else:
        raise ValueError("unknown question type")
    opts = "\n".join(f"{label}. {name}" + (f": {d}" if d else "")
                     for label, name, d in zip(SLOTS, names, descs))
    header = "Rate the state:" if kind == "score" else "Question:"
    answer = "letter" if len(names) <= 26 else "label"
    return f"{header} {ins}\n\nOptions:\n{opts}\n\nAnswer with one {answer}: {', '.join(SLOTS[:len(names)])}.", len(names)


class ElephantLayout:
    def __init__(self, tokenizer, decision):
        self.tok, self.cfg = tokenizer, decision

    def encode(self, state, questions):
        if not isinstance(questions, dict) or not questions:
            raise ValueError("at least one question is required")
        text = state if isinstance(state, str) else ("" if state is None else json.dumps(state, ensure_ascii=False, indent=2))
        pre = self.cfg["template"]["pre"] + f"<state>\n{text}\n</state>\n\n"
        if "QEV_PREFIX_MARK" in pre:
            raise ValueError("state collides with the author's prefix marker")
        post = self.cfg["template"]["post"]
        enc = lambda s: self.tok.encode(s, add_special_tokens=False).ids
        prefix = enc(pre)
        rows = []
        for key, q in questions.items():
            s, n = suffix(q)
            ids = enc(pre+s+post)
            if len(ids) > self.cfg["max_prompt_tokens"]:
                raise ValueError("prompt exceeds max_prompt_tokens; never truncated")
            rows.append({"qid": key, "ids": ids, "options": n, "noul": q["type"] == "noul"})
        p = len(prefix)
        while p and any(r["ids"][:p] != prefix[:p] for r in rows):
            p -= 1
        for r in rows:
            r["prefix_length"] = p
        if p + sum(len(r["ids"])-p for r in rows) > self.cfg.get("max_request_tokens",16384):
            raise ValueError("request exceeds shared-prefix token budget")
        return rows
