//! Author token fixtures, optionally followed by the full model's raw-logit gate.
use std::path::Path;

use ollaya_decision::{TokenEncoder, elephant::ElephantLayout};
use ollaya_runner::{elephant::ElephantModel, onnx::Device};
use serde_json::Value;

struct Tok(tokenizers::Tokenizer);
impl TokenEncoder for Tok {
    fn encode(&self, s: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(s, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        return Err("usage: parity_elephant MODEL_DIR GOLDENS [cpu|cuda], or --layout-only TOKENIZER GOLDENS".into());
    }
    let layout_only = args[1] == "--layout-only";
    let tokenizer = if layout_only {
        std::path::PathBuf::from(args.get(2).ok_or("missing tokenizer")?)
    } else {
        Path::new(&args[1]).join("tokenizer.json")
    };
    let golden_path = args
        .get(if layout_only { 3 } else { 2 })
        .ok_or("missing goldens")?;
    let tok = Tok(tokenizers::Tokenizer::from_file(tokenizer).map_err(|e| e.to_string())?);
    let data: Value = serde_json::from_str(&std::fs::read_to_string(golden_path)?)?;
    let layout: ElephantLayout = serde_json::from_value(data["layout"].clone())?;
    let threads = std::env::var("OLLAYA_THREADS")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?;
    if threads == Some(0) {
        return Err("OLLAYA_THREADS must be positive".into());
    }
    let model = if !layout_only {
        let device = if args.get(3).is_some_and(|s| s == "cuda") {
            Device::Cuda(0)
        } else {
            Device::Cpu
        };
        Some(ElephantModel::load(Path::new(&args[1]), device, threads)?)
    } else {
        None
    };
    let mut rows_checked = 0;
    let mut max_error = 0.0_f64;
    let mut max_probability_error = 0.0_f64;
    for case in data["cases"].as_array().ok_or("no cases")? {
        let defs: Vec<_> = case["questions"]
            .as_object()
            .ok_or("no questions")?
            .iter()
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        let rows = layout.rows(&tok, &case["state"], &defs)?;
        assert_eq!(rows.len(), case["rows"].as_array().ok_or("no rows")?.len());
        for (row, want) in rows.iter().zip(case["rows"].as_array().ok_or("no rows")?) {
            assert_eq!(
                row.ids,
                serde_json::from_value::<Vec<u32>>(want["ids"].clone())?
            );
            assert_eq!(
                row.prefix_length,
                want["prefix_length"].as_u64().unwrap() as usize
            );
            assert_eq!(row.options, want["options"].as_u64().unwrap() as usize);
            assert_eq!(row.noul, want["noul"].as_bool().unwrap());
            rows_checked += 1;
        }
        if let Some(m) = &model {
            let logits = m.option_logits(&rows)?;
            let want: Vec<Vec<f32>> = serde_json::from_value(case["logits"].clone())?;
            assert_eq!(logits.len(), want.len());
            for (got, want) in logits.iter().zip(&want) {
                assert_eq!(got.len(), want.len());
                for (a, b) in got.iter().zip(want) {
                    assert!(a.is_finite() && b.is_finite());
                    max_error = max_error.max(f64::from((a - b).abs()));
                }
                let probabilities = |v: &[f32]| {
                    ollaya_decision::calibration::softmax(v.iter().map(|&x| f64::from(x)))
                };
                for (a, b) in probabilities(got).iter().zip(probabilities(want)) {
                    max_probability_error = max_probability_error.max((a - b).abs());
                }
                let argmax = |v: &[f32]| {
                    v.iter()
                        .enumerate()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(&a.0)))
                        .unwrap()
                        .0
                };
                assert_eq!(argmax(got), argmax(want), "decision mismatch");
            }
        }
    }
    println!(
        "record: {}",
        serde_json::json!({"family":"elephant","rows":rows_checked,"model_checked":model.is_some(),"threads":threads,"max_logit_error":max_error,"max_probability_error":max_probability_error})
    );
    assert!(
        max_error <= 1e-3,
        "raw logits failed 1e-3 gate: {max_error}"
    );
    assert!(
        max_probability_error <= 1e-4,
        "option probabilities failed 1e-4 gate: {max_probability_error}"
    );
    Ok(())
}
