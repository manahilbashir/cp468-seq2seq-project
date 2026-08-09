# Report + demo data sheet

**For:** Farhan (Role 5) · **Contents:** every measured number, exact setting, file
path and command needed for the 5-page report and the 8-minute video. Data only.

All measured numbers are filled in. The one remaining `[fill]`, the USD cost,
needs the current Gemini rate from Google's pricing page. Sources for every
figure are in [§9 Where every number lives](#9-where-every-number-lives).

---

## 1. Deliverables

| # | Deliverable | Format |
|---|---|---|
| 1 | System report | PDF, 5 pages + appendix (appendix not counted) |
| 2 | Public GitHub repo | https://github.com/manahilbashir/cp468-seq2seq-project |
| 3 | Demo video | 8 minutes |

Report sections required by PRD §5: dataset · system design and settings ·
experimental results · error analysis · limitations · appendix (contribution
statement + AI-use disclosure).

Repo checklist (PRD §5.2):

| Item | State | Evidence |
|---|---|---|
| Dataset included + license documented | Done | `data/raw/dataset.csv`, `data/processed/metadata.json` |
| Complete implementation | Done | `src/`, `train.py`, `evaluate.py`, `llm_baseline.py`, `compare_results.py` |
| Pinned dependencies | Done | `requirements.txt`, 41 pins |
| README with setup + exact commands | Done | `README.md` |
| Fixed random seeds | Done | 42 — `src/utils.py:set_seed`, seeds Python / NumPy / PyTorch |

Reproducibility was verified, not just claimed: re-running `src/preprocess.py`
from the raw CSV regenerated all six files in `data/processed/` **byte-identical**
(SHA-1 match on `metadata.json`, both vocabularies, and all three `.jsonl`
splits). Safe to state as a checked fact in the report.

---

## 2. Dataset

| Field | Value |
|---|---|
| Name | Kaggle *News Summary* |
| Identifier | `sunnysai12345/news-summary` |
| URL | https://www.kaggle.com/datasets/sunnysai12345/news-summary |
| License | GPL-2.0 |
| Task | Article body → headline |
| Columns used | `ctext` (article) → `headlines` (headline) |
| Clean examples | 2,691 |
| Train / validation / test | 2,152 / 269 / 270 (80 / 10 / 10) |
| Split method | `sklearn.train_test_split`, shuffled, `random_state=42`, two-stage |
| Split timing | Before vocabulary construction and before any model development |

### Cleaning filters applied (`src/preprocess.py`)

| Step | Rule |
|---|---|
| Drop missing | `dropna` on article and headline |
| Whitespace | collapse runs of whitespace, strip |
| Drop empty | empty article or headline removed |
| Deduplicate | `drop_duplicates` on article text |
| Article length filter | keep 20–400 tokens (outside range **dropped**, not truncated) |
| Headline length filter | keep 2–30 tokens |
| Unicode | NFKC normalization |
| Case | lowercased |
| Tokenizer | regex `\w+|[^\w\s]` (`src/tokenizer.py:5`) — words and single punctuation marks |

### Length distribution (tokens, excluding `<bos>`/`<eos>`)

| Split | Field | mean | median | p5 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| train | article | 244.2 | 250 | 73 | 385 | 20 | 400 |
| train | headline | 10.9 | 11 | 8 | 14 | 5 | 21 |
| validation | article | 243.5 | 246 | 78 | 381 | 24 | 398 |
| validation | headline | 10.9 | 11 | 8 | 14 | 6 | 17 |
| test | article | 235.9 | 241 | 70 | 376 | 23 | 397 |
| test | headline | 10.7 | 10 | 8 | 15 | 5 | 18 |

Corpus-wide averages in `metadata.json`: article 243.31, headline 10.85.

---

## 3. Vocabulary and out-of-vocabulary rates

| Field | Source | Target |
|---|---|---|
| Distinct types in training split | 27,491 | 5,774 |
| Kept (min frequency 2, cap 30,000) | 15,877 | 2,523 |
| Plus specials `<pad> <unk> <bos> <eos>` | **15,881** | **2,527** |
| Total tokens in training split | 525,552 | 23,380 |
| Built from | training split only | training split only |

### `<unk>` rates (% of real tokens, specials excluded)

| Split | Article tokens `<unk>` | Headline tokens `<unk>` | Headlines with ≥1 `<unk>` |
|---|---|---|---|
| train | 2.21% (11,614 / 525,552) | 13.91% (3,251 / 23,380) | 78.0% |
| validation | 4.78% (3,134 / 65,502) | 21.69% (633 / 2,919) | 93.7% |
| **test** | **4.77%** (3,037 / 63,693) | **23.56%** (681 / 2,890) | **95.6%** |

Consequence, as a measurable fact: 23.56% of the reference headline tokens on
the test set are not in the LSTM's 2,527-token output vocabulary, so the model
cannot emit them under any decoding strategy. 258 of 270 test references contain
at least one such token. This is an upper bound on attainable ROUGE for the
LSTM, and it does not apply to the LLM, which decodes over its own subword
vocabulary.

---

## 4. LSTM architecture (`src/model.py`)

Standard PyTorch layers only — `nn.LSTM`, `nn.Embedding`, `nn.Linear`,
`nn.Dropout`. No Fairseq / OpenNMT / HuggingFace trainer.

| Stage | Layer | In → Out |
|---|---|---|
| Encoder | `nn.Embedding(15881, 128, padding_idx=0)` | ids → 128 |
| Encoder | `nn.LSTM(128, 256, layers=1, bidirectional=True, batch_first)` | 128 → 512 (2×256) |
| Encoder | pack/pad (`pack_padded_sequence`, `enforce_sorted=False`) | — |
| Encoder | `Linear(512, 256)` + `tanh` → decoder h₀ | 512 → 256 |
| Encoder | `Linear(512, 256)`, no activation → decoder c₀ | 512 → 256 |
| Attention | Bahdanau additive: `v ᵀ tanh(W_h·h_i + W_s·s_t)` | `W_h` 512→256, `W_s` 256→256, `v` 256→1 |
| Attention | padding masked with `-inf` before softmax | — |
| Decoder | `nn.Embedding(2527, 128, padding_idx=0)` | ids → 128 |
| Decoder | `nn.LSTM(128 + 512, 256, layers=1, batch_first)` | 640 → 256 |
| Decoder | `Linear(256 + 512, 2527)` | 768 → 2,527 |
| Inference | greedy decoding, `max_length=60` default | — |

Attention is recomputed at every decoder timestep from the top-layer hidden
state; the context vector is concatenated both into the LSTM input and into the
output projection input.

### Parameter breakdown — 6,469,599 trainable

| Module | Parameters | Share |
|---|---|---|
| `encoder.embedding` | 2,032,768 | 31.42% |
| `decoder.output_projection` | 1,943,263 | 30.04% |
| `decoder.lstm` | 919,552 | 14.21% |
| `encoder.lstm` | 790,528 | 12.22% |
| `decoder.embedding` | 323,456 | 5.00% |
| `decoder.attention` | 197,376 | 3.05% |
| `encoder.hidden_projection` | 131,328 | 2.03% |
| `encoder.cell_projection` | 131,328 | 2.03% |
| Encoder subtotal | 3,085,952 | 47.70% |
| Decoder subtotal | 3,383,647 | 52.30% |
| **Total** | **6,469,599** | 100% |

fp32 weights 24.7 MiB · checkpoint on disk 74.0 MiB (includes Adam optimizer state).

---

## 5. Training settings and run

### Hyperparameters (`results/training_config.json`)

| Setting | Value |
|---|---|
| embedding_dim | 128 |
| hidden_dim | 256 |
| num_layers | 1 |
| dropout | 0.3 |
| batch_size | 32 |
| epochs (max) | 30 |
| optimizer | Adam |
| learning_rate | 0.001 |
| loss | `CrossEntropyLoss(ignore_index=<pad>)` |
| teacher_forcing_ratio (initial) | 1.0 |
| teacher_forcing_decay | 0.02 per epoch |
| clip_grad (max norm) | 1.0 |
| early-stopping patience | 5 epochs |
| seed | 42 |
| device | cpu |
| `--unk-loss-weight` | 0.0 (run B) / 1.0 (run A) — see ablation below |

`<unk>` is excluded from the loss via a class-weight vector rather than
`ignore_index`, which accepts only one token. With `reduction="mean"` the loss
divides by the summed weights of the targets, so a zero-weighted class drops out
of numerator and denominator alike — verified numerically identical to deleting
those positions, and `--unk-loss-weight 1.0` is numerically identical to the
original unweighted loss.

Validation loss is computed with `teacher_forcing_ratio=0.0` — free-running,
the model conditions on its own predictions ([train.py:248](../train.py:248)).
Training loss uses the decayed teacher-forcing ratio. The two curves therefore
measure different things and are not directly comparable; validation loss is
the harder, inference-like number, and it is what checkpoint selection and
early stopping use.

### Hardware

| Field | Value |
|---|---|
| Machine | Apple M4, 10 cores, 16 GB RAM |
| OS | macOS 27.0 |
| Python | 3.9.6 |
| PyTorch | 2.8.0 |
| Device used | CPU (no CUDA) |

### Run A — `--unk-loss-weight 1.0` (original loss)

Artifacts preserved in `results/ablation_unk_weight_1.0/`.

| Field | Value |
|---|---|
| Epochs completed | 10 of 30 |
| Stopped by | Early stopping (5 epochs without validation improvement) |
| Best epoch | 5 |
| Best validation loss | 5.1689 |
| Best validation perplexity | 175.73 |
| Training loss at epoch 1 → 10 | 6.1393 → 3.7914 |
| Validation loss at epoch 1 → 10 | 5.2444 → 5.2925 |
| Mean time per epoch | 146.7 s |
| Total training time | 1,466.9 s (24.4 min) |

Per-epoch record:

| Epoch | Train loss | Train PPL | Val loss | Val PPL | TF ratio | Time (s) | Best |
|---|---|---|---|---|---|---|---|
| 1 | 6.1393 | 463.73 | 5.2444 | 189.51 | 1.00 | 114.3 | ✓ |
| 2 | 5.5382 | 254.22 | 5.1730 | 176.44 | 0.98 | 120.4 | ✓ |
| 3 | 5.3320 | 206.86 | 5.2582 | 192.14 | 0.96 | 156.8 | |
| 4 | 5.0936 | 162.97 | 5.2389 | 188.47 | 0.94 | 156.1 | |
| 5 | 4.8712 | 130.47 | 5.1689 | 175.73 | 0.92 | 150.6 | ✓ |
| 6 | 4.6373 | 103.26 | 5.2007 | 181.41 | 0.90 | 149.0 | |
| 7 | 4.4104 | 82.30 | 5.1967 | 180.68 | 0.88 | 150.2 | |
| 8 | 4.1942 | 66.30 | 5.2079 | 182.71 | 0.86 | 151.2 | |
| 9 | 3.9612 | 52.52 | 5.2509 | 190.74 | 0.84 | 162.0 | |
| 10 | 3.7914 | 44.32 | 5.2925 | 198.84 | 0.82 | 155.3 | |

Measured facts: training loss falls monotonically across all 10 epochs while
validation loss stays within 5.17–5.29 and never improves after epoch 5. The
checkpoint is epoch 5. This run's decoder collapsed onto `<unk>` — see §7.

### Run B — `<unk>` masked, free-running validation loss

Artifacts preserved in `results/ablation_freerunning_val_loss/`.

| Field | Value |
|---|---|
| Epochs completed | 6 of 30 |
| Stopped by | Early stopping (patience 5) |
| Best epoch | **1** |
| Best validation loss | 5.9266 |
| Total training time | 855.6 s (14.3 min) |

| Epoch | Train loss | Val loss | Best |
|---|---|---|---|
| 1 | 6.4858 | 5.9266 | ✓ |
| 2 | 5.8749 | 6.6127 | |
| 3 | 5.6425 | 6.4821 | |
| 4 | 5.3488 | 6.2433 | |
| 5 | 5.0614 | 6.2485 | |
| 6 | 4.7725 | 6.2747 | |

Decoding (first 40 test articles): 0/40 empty — the `<unk>` collapse is gone —
but the selected epoch-1 checkpoint emits the identical string
`delhi to to to to in ' s` for every article. Mean predicted length 7.5 tokens.

Diagnosis: validation loss was computed free-running
(`teacher_forcing_ratio=0.0`) while training loss was teacher forced. Those are
different quantities. Free-running loss rose monotonically from epoch 2 while
training loss fell, so early stopping selected the least-trained checkpoint.

### Run C — `<unk>` masked, teacher-forced validation loss (shipped)

`validate_epoch` changed to `teacher_forcing_ratio=1.0` so validation measures
the same quantity as training and is usable for model selection.

| Field | Value |
|---|---|
| Epochs completed | 12 of 30 |
| Stopped by | Early stopping (patience 5) |
| Best epoch | 7 |
| Best validation loss | 5.0501 |
| Best validation perplexity | 156.03 |
| Mean time per epoch | 157.3 s |
| Total training time | 1,887.9 s (31.5 min) |

| Epoch | Train loss | Train PPL | Val loss | Val PPL | TF ratio | Time (s) | Best |
|---|---|---|---|---|---|---|---|
| 1 | 6.4858 | 655.79 | 5.6726 | 290.78 | 1.00 | 123.4 | ✓ |
| 2 | 5.8749 | 355.98 | 5.4920 | 242.74 | 0.98 | 154.4 | ✓ |
| 3 | 5.6425 | 282.16 | 5.3496 | 210.53 | 0.96 | 151.3 | ✓ |
| 4 | 5.3488 | 210.36 | 5.1862 | 178.78 | 0.94 | 152.9 | ✓ |
| 5 | 5.0614 | 157.82 | 5.0863 | 161.80 | 0.92 | 153.3 | ✓ |
| 6 | 4.7725 | 118.21 | 5.0740 | 159.81 | 0.90 | 155.0 | ✓ |
| 7 | 4.5127 | 91.16 | **5.0501** | **156.03** | 0.88 | 154.9 | ✓ |
| 8 | 4.2085 | 67.25 | 5.0543 | 156.70 | 0.86 | 160.7 | |
| 9 | 3.9119 | 49.99 | 5.0725 | 159.57 | 0.84 | 152.1 | |
| 10 | 3.7015 | 40.51 | 5.1569 | 173.63 | 0.82 | 153.3 | |
| 11 | 3.3480 | 28.45 | 5.1649 | 175.02 | 0.80 | 189.3 | |
| 12 | 3.0618 | 21.37 | 5.2149 | 183.99 | 0.78 | 183.8 | |

Validation loss decreases monotonically for 7 epochs, reaches its minimum at
epoch 7, then rises as training loss keeps falling — a clean overfitting
signature with a well-defined model-selection point.

### Shipped checkpoint — decoding behaviour (first 40 test articles, `max_length=30`)

| Field | Run A | Run B | Run C (shipped) |
|---|---|---|---|
| Empty predictions | 34 / 40 | 0 / 40 | **0 / 40** |
| Distinct predictions | — | **1 / 40** | **40 / 40** |
| Mean predicted length | ~0 | 7.5 | 14.2 |
| Reference mean length | 10.7 | 10.7 | 10.7 |

Sample Run C predictions against references:

| Prediction | Reference |
|---|---|
| `india ' s ' ' s ' ' s ' ' s '` | Deepika Padukone's dress fifth most googled Met Gala outfit |
| `india ' s son ' s son ' s son ' s son : reports` | Even Kapil misbehaved on flight but no ban on him: Sena MP |
| `pm modi to launch modi of india ' s son modi` | Female foeticide can't be allowed to take place: PM Modi |

Observed failure modes for the error-analysis section: n-gram repetition loops,
over-generation (14.2 vs 10.7 tokens), and generic high-frequency corpus
vocabulary (`india`, `' s`, `modi`) substituting for entity-specific content.

Validation losses are **not comparable across runs A/B/C as numbers**: A
includes `<unk>` targets, B and C exclude them, and C is teacher forced while A
and B are free-running. Compare on decoded output and BLEU/ROUGE, never on loss.

Full per-epoch table: `results/training_log.txt`.
Curves: `results/training_curves.png` (loss, perplexity).
Raw series: `results/training_history.json`.

---

## 6. LLM baseline (`llm_baseline.py`)

| Field | Value |
|---|---|
| Model | `gemini-3.5-flash-lite` |
| Endpoint | `generativelanguage.googleapis.com/v1beta/models/{model}:generateContent` |
| temperature | 0.2 |
| maxOutputTokens | 40 |
| Prompt variants | 2 — zero-shot, few-shot (k=3) |
| Test examples | 270 (identical `data/processed/test.jsonl` as the LSTM) |
| Requests | 540 (2 per example) |
| Delay between requests | 1.0 s |
| Post-processing | strip leading `Headline:`, strip surrounding quotes, collapse whitespace |
| Output file | `results/llm_outputs.jsonl` |

### Zero-shot prompt (verbatim, `llm_baseline.py:20`)

```
You are a professional news editor.

Generate one concise and factual headline for the article below.

Rules:
- Output only the headline.
- Do not add quotation marks.
- Do not invent information.
- Keep the headline under 15 words.

Article:
{article}

Headline:
```

### Few-shot prompt (verbatim, `llm_baseline.py:36`)

```
You are a professional news editor.

Generate one concise and factual headline for the final article.

Rules:
- Output only the headline.
- Do not add quotation marks.
- Do not invent information.
- Keep the headline under 15 words.

Example 1:
Article:
Apple announced a new series of laptops featuring faster processors and longer battery life.

Headline:
Apple Unveils Faster Laptops With Longer Battery Life

Example 2:
Article:
Heavy rainfall caused flooding across several neighbourhoods and forced officials to close roads.

Headline:
Heavy Rain Triggers Flooding and Road Closures

Example 3:
Article:
The national football team defeated its rival 2-1 after scoring in the final minutes.

Headline:
Late Goal Secures 2-1 Victory for National Team

Now generate a headline for this article.

Article:
{article}

Headline:
```

Style fact: the three exemplars are Title Case Western-desk headlines; the
corpus references are terser and lowercase after preprocessing.

---

## 7. Results

Metrics: sacrebleu `corpus_bleu` (BLEU) and `rouge_score.RougeScorer` with
`use_stemmer=True`, F-measure, averaged per example (ROUGE-1/2/L). One reference
per example. All three systems scored on the same 270 test examples by
`compare_results.py`.

Measured 2026-08-08, from `results/comparison_metrics.json`:

| System | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| LSTM + attention | 0.44 | 0.0821 | 0.0117 | 0.0781 |
| Gemini zero-shot | 10.46 | **0.4536** | **0.2050** | **0.3870** |
| Gemini few-shot (k=3) | **10.70** | 0.4388 | 0.1944 | 0.3758 |

> **Casing note (resolved).** BLEU is computed with `lowercase=True`, because the
> LSTM can only emit lowercase — `src/tokenizer.py:clean_text` lowercases the
> corpus — while the references and LLM outputs keep their original casing.
> Case-sensitive BLEU charged the LSTM for a preprocessing decision rather than
> for its predictions. The earlier case-sensitive Gemini figures (zero-shot 3.85,
> few-shot 2.64) are superseded by the table above; do not quote them. ROUGE was
> never affected — `rouge_score` lowercases internally.

### Target-vocabulary cutoff sensitivity (measured)

| `min_frequency` | Target vocab | Test headline OOV |
|---|---|---|
| 1 | 5,778 | 15.47% |
| **2 (used)** | **2,527** | **23.56%** |
| 3 | 1,587 | 29.48% |

### Source-column choice in the raw CSV (measured)

`data/raw/dataset.csv` carries 4,514 rows and two candidate article columns.
`src/preprocess.py:167` matches `ctext` before `text`, so `ctext` is used.

| Article column | Usable after clean+dedup | Article tokens (mean / median / p95) | Survive 20–400 filter |
|---|---|---|---|
| `ctext` (used) | 4,341 | 412 / 340 / 868 | **2,691 (62.0%)** — 1,649 dropped for exceeding 400 tokens |
| `text` | 4,514 | 70 / 70 / 81 | **4,514 (100%)** |

Measured ordering fact: the zero-shot / few-shot ranking is **metric-dependent**.
Few-shot leads on BLEU (10.70 vs 10.46); zero-shot leads on all three ROUGE
measures. Do not state that either setting is uniformly better. The few-shot
exemplars in `llm_baseline.py` are Western title-case headlines while this
corpus is terser, so they steer the model away from the reference style — a
shift ROUGE registers through content overlap and BLEU does not.

Metric-behaviour fact: zero-shot BLEU is 10.46 while its ROUGE-1 is 0.4536 for
the same predictions. Outputs average ~11 tokens against a single reference, so
BLEU's 4-gram precision and brevity penalty dominate; ROUGE-1/L reflect the
actual unigram overlap. The metric disagreement above is a direct consequence,
and is the reason to lead the analysis with ROUGE.

### LLM cost

From `results/comparison_metrics.json → llm_cost_estimate` (Role 4 run):

| Field | Value |
|---|---|
| Requests | 540 |
| Estimated prompt tokens | 234,479 |
| Estimated completion tokens | 9,911 |
| Estimation method | ~4 chars/token heuristic, not the Gemini tokenizer — order of magnitude |
| USD | `[fill]` — multiply by current input/output rates at https://ai.google.dev/pricing |
| LSTM cost per request | $0 after training; runs offline on CPU |

### Error-analysis inputs

`results/qualitative_comparison.md` — 12 rows (article / reference / LSTM /
LLM zero-shot / LLM few-shot / notes), selected by even steps across the
source-length distribution, not cherry-picked.

Automatic heuristic flags in the `Notes` column, and what triggers them
(`compare_results.py:116`):

| Flag | Trigger |
|---|---|
| `LSTM empty output` | prediction is blank after stripping |
| `LSTM repetition` | ≥3 bigrams and unique bigrams < 70% of total |
| `LLM zero-shot over-elaborates` | prediction > 3× reference token count |
| `LLM zero-shot empty/error` | blank prediction |

These are heuristics, not the PRD's error categories (under-translation,
repetition, hallucination, OOV, fluency vs. adequacy) — those are assigned by
hand.

Full 270-row predictions: `results/test_predictions.json`.

---

## 8. Reproduction commands

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

```bash
.venv/bin/python src/preprocess.py
```

```bash
.venv/bin/python train.py --data-dir data/processed --results-dir results \
  --embedding-dim 128 --hidden-dim 256 --num-layers 1 --dropout 0.3 \
  --batch-size 32 --epochs 30 --learning-rate 0.001 \
  --teacher-forcing-ratio 1.0 --teacher-forcing-decay 0.02 \
  --clip-grad 1.0 --patience 5 --seed 42 --unk-loss-weight 0.0
```

Reproduce the collapsed Run A by changing the last flag to
`--unk-loss-weight 1.0`.

```bash
export GEMINI_API_KEY=...
.venv/bin/python llm_baseline.py --limit 0
```

```bash
.venv/bin/python evaluate.py --checkpoint results/best_model.pt \
  --data-dir data/processed --results-dir results --num-examples 10
.venv/bin/python compare_results.py
```

---

## 9. Where every number lives

### Checkpoint files

| File | Size | On GitHub | Purpose |
|---|---|---|---|
| `results/best_model.pt` | 74.1 MiB | no (gitignored) | Full checkpoint incl. Adam state; resumable |
| `results/best_model_inference.pt` | 24.7 MiB | no (gitignored) | Weights + config only; **verified to produce predictions identical to the full checkpoint on 30/30 test articles**. This is the one to share for evaluation. |

Both load through `evaluate.load_model` unchanged. Regenerate the slim copy with
`python scripts/export_checkpoint.py`.

### Source of each number

| Number | File |
|---|---|
| Dataset size, splits, vocab sizes, length filters, license | `data/processed/metadata.json` |
| Hyperparameters, parameter count, device, vocab sizes | `results/training_config.json` |
| Per-epoch loss / perplexity / time, total time, best val loss | `results/training_log.txt` |
| Loss + perplexity series | `results/training_history.json` |
| Loss + perplexity plots | `results/training_curves.png` |
| LSTM BLEU / ROUGE alone | `results/evaluation_metrics.json` |
| All three systems' BLEU / ROUGE + cost estimate | `results/comparison_metrics.json` |
| Side-by-side examples for error analysis | `results/qualitative_comparison.md` |
| All 270 LSTM predictions | `results/test_predictions.json` |
| All 270 LLM predictions, both prompts | `results/llm_outputs.jsonl` |
| Exact prompts | `llm_baseline.py:20` and `:36` |
| Architecture | `src/model.py` |

`results/*.pt`, `results/training_*.json|png`, `results/evaluation_*.json`,
`results/test_predictions.json` and `results/qualitative_examples.md` are in
`.gitignore` — they are not on GitHub. `results/training_log.txt`,
`results/llm_outputs.jsonl`, `results/comparison_metrics.json` and
`results/qualitative_comparison.md` are not ignored.

---

## 10. Demo video assets

| Asset | Path | Exists |
|---|---|---|
| Raw article + headline pair | `data/raw/dataset.csv` | yes |
| Dataset facts | `data/processed/metadata.json` | yes |
| Architecture source | `src/model.py` | yes |
| Per-epoch training output | `results/training_log.txt` | yes |
| Loss / perplexity curves | `results/training_curves.png` | yes |
| Three-system metrics table | `results/comparison_metrics.json` | yes |
| Side-by-side examples | `results/qualitative_comparison.md` | yes |
| Live paste → headline clip | needs `demo.py` | see `plans/tui-headline-demo.md` |

### 8-minute budget

| Time | Segment | On screen |
|---|---|---|
| 0:00–1:00 | Task + dataset | metadata.json, one article/headline pair |
| 1:00–2:30 | Architecture | `src/model.py` — encoder, attention, decoder |
| 2:30–4:00 | Training | `training_log.txt`, `training_curves.png` |
| 4:00–6:00 | Results | metrics table, side-by-side examples |
| 6:00–7:30 | Gap analysis + trade-offs | `<unk>` table (§3), cost estimate (§7) |
| 7:30–8:00 | Limitations + next steps | — |

An interactive segment is optional and adds no marks — see
[plans/tui-headline-demo.md](plans/tui-headline-demo.md) for the tiers and cost.

---

## 11. Appendix fields to fill

### Contribution statement

| Role | Owner | Files owned |
|---|---|---|
| 1 · Data pipeline | Manahil | `src/preprocess.py`, `src/tokenizer.py`, `src/vocabulary.py`, `data/` |
| 2 · LSTM seq2seq | Safdar | `src/model.py`, `src/dataset.py`, `train.py`, `scripts/finish_training.py` |
| 3 · LLM baseline | Manahil | `llm_baseline.py`, `results/llm_outputs.jsonl` |
| 4 · Evaluation | Noah | `evaluate.py`, `compare_results.py` |
| 5 · Report + video | Farhan | report PDF, demo video, `docs/` |

### AI-use disclosure — fields required

Tool and version used · which files or sections it touched · what it produced
(code, prose, analysis) · how output was verified · what was written unaided.
