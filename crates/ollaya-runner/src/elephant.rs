//! Elephant-1 text ONNX runner: separate prefix/suffix blocks, causal DeltaNet.
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::elephant::{ElephantLayout, ElephantRow};
use ollaya_decision::{Calibration, CalibrationFile, Questions, TokenEncoder};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

/// One row per call bounds the first port's quadratic attention memory.
const MAX_ROWS: usize = 1;
/// Shrink CUDA's arena after a long row; the configured prompt limit controls admission.
const TOKEN_BUDGET: usize = 2048;
const INPUTS: [&str; 3] = ["input_ids", "last_pos", "prefix_lengths"];
const OUTPUT: &str = "cand_logits";

#[derive(Debug, Clone, Deserialize)]
struct Contract {
    /// Rows are padded to a multiple of this (one `Scan` step of the DeltaNet layers).
    seq_multiple: usize,
}

#[derive(Debug, Clone, Deserialize)]
struct SpecialTokens {
    pad: u32,
}

#[derive(Debug, Clone, Deserialize)]
struct Codes {
    strings: Vec<String>,
    ids: Vec<u32>,
}

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    contract: Contract,
    special_tokens: SpecialTokens,
    codes: Codes,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    elephant: ElephantLayout,
}

pub struct ElephantModel {
    session: Mutex<Session>,
    tokenizer: Tokenizer,
    seq_multiple: usize,
    pad: u32,
    pub layout: ElephantLayout,
    pub calibration: Calibration,
    pub device: Device,
}

struct Tokenizer(tokenizers::Tokenizer);

impl TokenEncoder for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ollaya_decision::Error> {
        self.0
            .encode_fast(text, false)
            .map(|e| e.get_ids().to_vec())
            .map_err(|e| ollaya_decision::Error::Tokenizer(e.to_string()))
    }
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

impl Engine for ElephantModel {
    fn run_json(&self, state: &Value, questions: &Value) -> Result<Output, Error> {
        let defs = questions
            .as_object()
            .ok_or_else(|| ollaya_decision::Error::invalid("questions must be an object"))?;
        self.answer(
            state,
            &defs
                .iter()
                .map(|(k, v)| (k.as_str(), v))
                .collect::<Vec<_>>(),
        )
    }

    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(qid, q)| (qid.as_str(), &q.definition))
            .collect();
        self.answer(state, &defs)
    }
}

impl ElephantModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx" || config.layout != "elephant-1-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/elephant-1-v1",
                config.engine, config.layout
            )));
        }
        let bad = |e: String| Error::Model(format!("{}: {e}", files.decision.display()));
        if !(1..=16384).contains(&config.elephant.max_prompt_tokens) {
            return Err(bad("max_prompt_tokens must be in 1..=16384".into()));
        }
        if !(1..=16384).contains(&config.elephant.max_request_tokens) {
            return Err(bad("max_request_tokens must be in 1..=16384".into()));
        }
        if config.contract.seq_multiple != 64 {
            return Err(bad("contract.seq_multiple must be 64".into()));
        }
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let labels = ollaya_decision::elephant::slot_labels();
        if config.codes.strings != labels || config.codes.ids.len() != 256 {
            return Err(bad("codes must be the author's 256 slot labels".into()));
        }
        for (label, id) in labels.iter().zip(&config.codes.ids) {
            let encoded = tokenizer
                .encode_fast(label.as_str(), false)
                .map_err(|e| bad(e.to_string()))?;
            if encoded.get_ids() != [*id] {
                return Err(bad(format!("slot {label} does not encode to {id}")));
            }
        }
        let weights = config.weights_in_memory;
        let session = session_for(
            &files.graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(crate::decider::configure(b, device)?),
        )?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the elephant contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(ElephantModel {
            session: Mutex::new(session),
            tokenizer: Tokenizer(tokenizer),
            seq_multiple: config.contract.seq_multiple,
            pad: config.special_tokens.pad,
            layout: config.elephant,
            calibration,
            device,
        })
    }

    /// Every question's row, in request order.
    pub fn rows(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Vec<ElephantRow>, Error> {
        Ok(self.layout.rows(&self.tokenizer, state, defs)?)
    }

    /// Answer the questions: one logit per option.
    pub fn answer(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Output, Error> {
        let rows = self.rows(state, defs)?;
        let logits = self.option_logits(&rows)?;
        Ok(Output {
            questions: logits
                .into_iter()
                .map(|logits| QuestionOutput {
                    logits,
                    act_logits: None,
                })
                .collect(),
            input_tokens: rows
                .iter()
                .map(|r| r.ids.len() - r.prefix_length)
                .sum::<usize>()
                + rows.first().map_or(0, |r| r.prefix_length),
            state_tokens: rows.first().map_or(0, |r| r.prefix_length),
            state_truncated: false,
        })
    }

    /// Each row's option logits (its first `options` code logits). Rows run shortest first, in
    /// batches padded to a multiple of `seq_multiple`.
    pub fn option_logits(&self, rows: &[ElephantRow]) -> Result<Vec<Vec<f32>>, Error> {
        let width = 256;
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let padded: Vec<usize> = order
            .iter()
            .map(|&i| rows[i].ids.len().div_ceil(self.seq_multiple) * self.seq_multiple)
            .collect();
        let pad = i64::from(self.pad);
        let mut logits = vec![Vec::new(); rows.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&padded, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            let seq = padded[range].iter().copied().max().unwrap_or(0);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), pad);
            let mut last_pos = Array1::<i64>::zeros(batch.len());
            let mut prefix_lengths = Array1::<i64>::zeros(batch.len());
            for (r, &i) in batch.iter().enumerate() {
                for (c, &id) in rows[i].ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                last_pos[r] = rows[i].ids.len() as i64 - 1;
                prefix_lengths[r] = rows[i].prefix_length as i64;
            }
            let options = crate::engine::run_options(self.device, batch.len() * seq, TOKEN_BUDGET)?;
            let outputs = session.run_with_options(
                ort::inputs![
                    "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                    "last_pos" => ort::value::Tensor::from_array(last_pos)?,
                    "prefix_lengths" => ort::value::Tensor::from_array(prefix_lengths)?,
                ],
                &options,
            )?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != batch.len() || out.ncols() != width {
                return Err(Error::Model(format!(
                    "{OUTPUT} has shape {:?} for {} rows of {width} codes",
                    out.shape(),
                    batch.len()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                logits[i] = row.iter().take(rows[i].options).copied().collect();
                if logits[i].iter().any(|v| !v.is_finite()) {
                    return Err(Error::Model(
                        "Elephant-1 returned non-finite option logits".into(),
                    ));
                }
                // Author slots are yes/no; Ollaya's canonical noul logits are false/true.
                if rows[i].noul {
                    logits[i].swap(0, 1);
                }
            }
        }
        Ok(logits)
    }
}
