//! Elephant-1's author prompt and two attention blocks (OneJev 81ce62f + serve.patch).
use serde::Deserialize;
use serde_json::Value;

use crate::{Error, TokenEncoder, pyjson};

#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    pub pre: String,
    pub post: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ElephantLayout {
    pub template: Template,
    pub max_prompt_tokens: usize,
    #[serde(default = "request_limit")]
    pub max_request_tokens: usize,
}

fn request_limit() -> usize {
    16384
}

#[derive(Debug, Clone)]
pub struct ElephantRow {
    pub ids: Vec<u32>,
    pub prefix_length: usize,
    pub options: usize,
    pub noul: bool,
}

fn strip(s: &str) -> &str {
    s.trim_matches(|c: char| c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c))
}

fn entry(v: &Value) -> Result<String, Error> {
    match v {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(strip(s).to_owned()),
        Value::Array(_) | Value::Object(_) => Ok(pyjson::dumps_indent(v, false, 2)),
        _ => Err(Error::invalid(
            "instructions and criteria entries must be strings, objects, arrays or null",
        )),
    }
}

pub fn slot_labels() -> Vec<String> {
    let skip = "BQ BZ CJ CQ CZ DQ DZ EJ EY FJ FQ FV FZ GJ GK GQ GZ HJ IY JF JG JH JL JN JQ";
    let mut out: Vec<String> = ('A'..='Z').map(String::from).collect();
    for a in 'A'..='Z' {
        for b in 'A'..='Z' {
            let s = format!("{a}{b}");
            if !skip.split_whitespace().any(|x| x == s) && out.len() < 256 {
                out.push(s);
            }
        }
    }
    out
}

/// Returns the author suffix, option count and whether slot order is yes/no.
pub fn suffix(q: &Value) -> Result<(String, usize, bool), Error> {
    if !q.is_object() {
        return Err(Error::invalid("question definition must be an object"));
    }
    let mut ins = entry(&q["instructions"])?;
    let c = &q["criteria"];
    let kind = q["type"].as_str().unwrap_or("");
    let (names, descs): (Vec<String>, Vec<String>) = match kind {
        "noul" => {
            if !c.is_null() && !c.is_object() {
                return Err(Error::invalid("noul criteria must be an object or null"));
            }
            if ins.is_empty() {
                ins = "Is the statement true, or is the answer to the question yes?".into();
            }
            let desc = |key: &str, default: &str| {
                if c[key].is_null() {
                    Ok(default.to_owned())
                } else {
                    entry(&c[key])
                }
            };
            (
                vec!["yes".into(), "no".into()],
                vec![
                    desc("true", "the statement is true / the answer is yes")?,
                    desc("false", "the statement is false / the answer is no")?,
                ],
            )
        }
        "choice" => {
            let c = c
                .as_object()
                .filter(|c| !c.is_empty() && c.len() <= 255)
                .ok_or_else(|| Error::invalid("choice requires 1..255 named options"))?;
            if ins.is_empty() {
                ins = "Which option applies to the state?".into();
            }
            (
                c.keys().cloned().collect(),
                c.values().map(entry).collect::<Result<_, _>>()?,
            )
        }
        "score" => {
            let c = c
                .as_array()
                .filter(|c| !c.is_empty() && c.len() <= 10)
                .ok_or_else(|| Error::invalid("score requires 1..10 levels"))?;
            if ins.is_empty() {
                ins = "Which level describes the state?".into();
            }
            (
                (0..c.len()).map(|i| format!("level {i}")).collect(),
                c.iter().map(entry).collect::<Result<_, _>>()?,
            )
        }
        _ => return Err(Error::invalid("unknown question type")),
    };
    let labels = slot_labels();
    let options = names.len();
    let opts = labels
        .iter()
        .zip(names)
        .zip(descs)
        .map(|((label, name), desc)| {
            if desc.is_empty() {
                format!("{label}. {name}")
            } else {
                format!("{label}. {name}: {desc}")
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    let header = if kind == "score" {
        "Rate the state:"
    } else {
        "Question:"
    };
    let answer = if options <= 26 { "letter" } else { "label" };
    Ok((
        format!(
            "{header} {ins}\n\nOptions:\n{opts}\n\nAnswer with one {answer}: {}.",
            labels[..options].join(", ")
        ),
        options,
        kind == "noul",
    ))
}

impl ElephantLayout {
    pub fn rows<T: TokenEncoder>(
        &self,
        tok: &T,
        state: &Value,
        defs: &[(&str, &Value)],
    ) -> Result<Vec<ElephantRow>, Error> {
        if defs.is_empty() {
            return Err(Error::invalid("at least one question is required"));
        }
        let state = match state {
            Value::Null => String::new(),
            Value::String(s) => s.clone(),
            v => pyjson::dumps_indent(v, false, 2),
        };
        let pre = format!("{}<state>\n{state}\n</state>\n\n", self.template.pre);
        if pre.contains("QEV_PREFIX_MARK") {
            return Err(Error::invalid(
                "state collides with the author's prefix marker",
            ));
        }
        let prefix = tok.encode(&pre)?;
        let mut rows = Vec::new();
        for (_, q) in defs {
            let (s, options, noul) = suffix(q)?;
            let ids = tok.encode(&format!("{pre}{s}{}", self.template.post))?;
            if ids.len() > self.max_prompt_tokens {
                return Err(Error::invalid(format!(
                    "prompt has {} tokens; limit is {}; never truncated",
                    ids.len(),
                    self.max_prompt_tokens
                )));
            }
            rows.push(ElephantRow {
                ids,
                prefix_length: 0,
                options,
                noul,
            });
        }
        let mut p = prefix.len();
        while p > 0 && rows.iter().any(|r| r.ids.get(..p) != Some(&prefix[..p])) {
            p -= 1;
        }
        if p == 0 || rows.iter().any(|r| p == r.ids.len()) {
            return Err(Error::invalid("empty prefix or suffix block"));
        }
        for r in &mut rows {
            r.prefix_length = p;
        }
        let total = p + rows.iter().map(|r| r.ids.len() - p).sum::<usize>();
        if total > self.max_request_tokens {
            return Err(Error::invalid(format!(
                "request has {total} shared-prefix tokens; limit is {}",
                self.max_request_tokens
            )));
        }
        Ok(rows)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn author_noul_order_and_defaults() {
        let (s, n, noul) = suffix(&json!({"type":"noul"})).unwrap();
        assert_eq!(n, 2);
        assert!(noul);
        assert!(s.contains("A. yes: the statement is true / the answer is yes\nB. no:"));
        assert!(s.ends_with("Answer with one letter: A, B."));
    }

    #[test]
    fn wide_choices_skip_non_token_labels() {
        let labels = slot_labels();
        assert_eq!(labels.len(), 256);
        assert!(!labels.contains(&"BQ".into()));
        let c: serde_json::Map<String, Value> =
            (0..70).map(|i| (format!("x{i}"), Value::Null)).collect();
        let (s, n, _) = suffix(&json!({"type":"choice", "criteria":c})).unwrap();
        assert_eq!(n, 70);
        assert!(s.contains("Answer with one label:"));
        assert!(!s.contains("BQ."));
    }

    #[test]
    fn author_rejections_and_structured_rendering() {
        for q in [
            json!({"type":"choice","criteria":[]}),
            json!({"type":"score","criteria":[]}),
            json!({"type":"noul","instructions":true}),
        ] {
            assert!(suffix(&q).is_err());
        }
        let (s, _, _) =
            suffix(&json!({"type":"score","criteria":[{"é":[1,null]}," fine "]})).unwrap();
        assert!(s.contains("A. level 0: {\n  \"é\": [\n    1,\n    null\n  ]\n}"));
        assert!(s.contains("B. level 1: fine"));
    }
}
