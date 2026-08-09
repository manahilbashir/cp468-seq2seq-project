# CP468: LSTM vs LLM for Headline Generation

**Course:** CP468 — Artificial Intelligence · Wilfrid Laurier University · Spring 2026
**Team:** Safdar · Farhan · Noah · Manahil · Morad

---

## What this project is

The task: given a **news article**, generate a **one-line headline**.

We build two systems for that task and compare them on the same test set:

1. **LSTM seq2seq, written from scratch** — bidirectional LSTM encoder → Bahdanau attention → LSTM decoder, trained only on our dataset
2. **LLM baseline** — Gemini prompted to write headlines for the same articles

The objective is not to outperform the LLM but to quantify the gap between the two and account for it: model capacity, pretraining, and the trade-offs in cost, latency, control and offline use.

**Deliverables:** public GitHub repo · 5-page report · 8-minute demo video

---

## Team and status

| Role | Owner | Status | Deliverable |
|------|-------|--------|-------------|
| 1 · Data pipeline | Manahil | Complete | `src/preprocess.py`, `data/processed/` |
| 2 · LSTM seq2seq | Safdar | Complete | `src/model.py`, `train.py`, trained checkpoint |
| 3 · LLM baseline | Manahil | Complete | `llm_baseline.py`, `results/llm_outputs.jsonl` |
| 4 · Evaluation | Noah | Complete | `evaluate.py`, `compare_results.py`, metrics below |
| 5 · Report & video | Farhan | In progress | 5-page PDF report, 8-minute demo video |

All code is written and has been run. Every figure the report needs is in `results/`.

### Training run

12 epochs with early stopping; best validation loss **5.0501** at epoch 7; 31.5 minutes on an Apple M4 CPU. Curves, logs and configuration are in `results/`.

**The model trains successfully but generates low-quality headlines.** Across the first 40 test articles it produced 40 distinct, non-empty predictions, confirming that it conditions on its input rather than emitting a fixed string. The output is nonetheless weak and prone to repetition loops. This is the expected outcome at this data scale: with 2,691 examples, validation loss reached its minimum at epoch 7 and rose thereafter while training loss continued to fall — textbook overfitting. It forms the basis of the report's error analysis.

### Two defects found and fixed during training

Each is preserved as a reproducible ablation under `results/`, with its own README.

| Ablation | Symptom | Resolution |
|---|---|---|
| `ablation_unk_weight_1.0/` | Decoder emitted `<unk>` almost exclusively; 34 of 40 predictions were empty once special tokens were stripped | `--unk-loss-weight 0.0`. `<unk>` accounted for 13.91% of training headline tokens — roughly 4× the most frequent real word — making a constant `<unk>` prediction the lowest-loss policy available |
| `ablation_freerunning_val_loss/` | Early stopping selected the epoch-1 checkpoint, which produced one identical headline for every article | Validation loss is now computed with teacher forcing, matching training conditions, so it ranks checkpoints meaningfully |

### Checkpoint availability

`results/best_model_inference.pt` is committed to the repository — no separate download is required. It is the epoch-7 model with the Adam optimizer state removed (74.1 MiB → 24.7 MiB) and is verified to produce predictions identical to the full checkpoint.

The full `results/best_model.pt` is excluded from version control. Regenerate the slim copy after any retrain with `python scripts/export_checkpoint.py`.

---

## Final results (270 test examples)

All three systems are scored on the same 270 test examples under the same settings. Source: `results/comparison_metrics.json`.

| System | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| LSTM + attention | 0.44 | 0.0821 | 0.0117 | 0.0781 |
| Gemini zero-shot | 10.46 | **0.4536** | **0.2050** | **0.3870** |
| Gemini few-shot (k=3) | **10.70** | 0.4388 | 0.1944 | 0.3758 |

The gap is large and consistent: the LLM scores roughly **5.5× higher on ROUGE-1** and **24× higher on BLEU**. Measuring and explaining that gap is the object of the project.

**Note on BLEU casing.** BLEU is computed with `lowercase=True` (`evaluate.py:147`). Preprocessing lowercases the corpus (`src/tokenizer.py`) to hold the target vocabulary at 2,527 words — without it, `India` and `india` occupy separate entries and already-sparse counts are split further. The LSTM can therefore only emit lowercase, while the references and Gemini outputs retain normal capitalization, so case-sensitive BLEU penalized the LSTM for a preprocessing decision rather than for prediction quality. ROUGE is unaffected, as `rouge_score` lowercases internally.

### Findings for the report

- **The zero-shot / few-shot ranking depends on the metric.** Few-shot leads narrowly on BLEU (10.70 vs 10.46), while zero-shot leads on all three ROUGE measures. The few-shot exemplars in `llm_baseline.py` use Western title-case phrasing, whereas this corpus is terser, so they steer the model away from the reference style — a shift ROUGE detects through content overlap and BLEU does not. Neither setting is uniformly better.
- **BLEU is a weak metric for this task.** On ~10-token headlines with a single reference, the brevity penalty and 4-gram precision make it unstable, which is precisely why the two Gemini settings rank differently across metrics. ROUGE should lead the analysis, with this limitation stated.
- **The LSTM's dominant failure mode is repetition.** For example, from `results/qualitative_comparison.md`: `india ' s son ' s son ' s son ' s son`. Greedy decoding combined with an underfit model settles into high-frequency loops; roughly half the sampled examples exhibit it.

**LLM cost.** 540 requests, approximately 234K prompt tokens and 9.9K completion tokens (`llm_cost_estimate` in the metrics JSON). This uses a 4-characters-per-token heuristic rather than Gemini's tokenizer; multiply by the current rates on [Google's pricing page](https://ai.google.dev/pricing) and present the result as an order-of-magnitude estimate.

---

## LSTM architecture (Role 2)

Standard PyTorch layers only (`nn.LSTM`, `nn.Embedding`, `nn.Linear`) — no Fairseq / OpenNMT / HuggingFace `Seq2SeqTrainer`.

- **Encoder:** bidirectional LSTM (+ pack/pad)
- **Attention:** Bahdanau (additive), padding masked
- **Decoder:** unidirectional LSTM, attends each step
- **Training:** teacher forcing (optional decay), grad clipping, early stopping, checkpointing
- **Inference:** greedy decoding

---

## Setup

```bash
git clone https://github.com/manahilbashir/cp468-seq2seq-project.git
cd cp468-seq2seq-project

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pins everything, including `sacrebleu`, `rouge_score` and `certifi`. Verified on Python 3.9 / macOS.

---

## How to run

Run from the repository root, in order. These four steps reproduce every number reported above.

All four have already been run and their outputs are committed, so this section is for reproduction rather than for completing the project. Step 1 requires the Kaggle dataset and step 3 a Gemini API key; step 2 takes about 31 minutes. Step 4 takes roughly 8 seconds and can be re-run at any time.

### 1. Preprocess *(Role 1)*

```bash
python src/preprocess.py
```

Reads `data/raw/dataset.csv`, writes `train/validation/test.jsonl`, `source_vocab.json`, `target_vocab.json` and `metadata.json` into `data/processed/`. Vocabularies come from the **training split only**, so no val/test leakage. Seed 42.

### 2. Train the LSTM *(Role 2)*

```bash
python train.py \
  --data-dir data/processed \
  --results-dir results \
  --embedding-dim 128 \
  --hidden-dim 256 \
  --num-layers 1 \
  --dropout 0.3 \
  --batch-size 32 \
  --epochs 30 \
  --learning-rate 0.001 \
  --teacher-forcing-ratio 1.0 \
  --teacher-forcing-decay 0.02 \
  --clip-grad 1.0 \
  --patience 5 \
  --seed 42 \
  --unk-loss-weight 0.0
```

Writes `results/best_model.pt`, `training_curves.png`, `training_history.json`, `training_config.json`. ~31.5 min on an Apple M4 CPU (~157 s/epoch). Append `| tee results/training_log.txt` to keep the per-epoch log the report needs.

`--unk-loss-weight` defaults to `0.0` (removes `<unk>` from the loss). Pass `1.0` to reproduce the collapse in `results/ablation_unk_weight_1.0/`.

Optional hyperparameter search — trains a grid, promotes the best run by validation loss:

```bash
python scripts/finish_training.py --quick   # 2 configs
```

### 3. LLM baseline *(Role 3)*

Gemini on the **same** `data/processed/test.jsonl`, two prompt settings (zero-shot and 3-example few-shot) per PRD §4.2. Both prompts are at the top of `llm_baseline.py`.

```bash
export GEMINI_API_KEY=your_key_here

python llm_baseline.py --limit 0     # 0 = full test set
```

Writes `results/llm_outputs.jsonl` — one record per test example with the article, reference headline, and both generated headlines.

### 4. Evaluate and compare *(Role 4)*

Approximately 8 seconds total on CPU.

```bash
python evaluate.py --checkpoint results/best_model_inference.pt --data-dir data/processed --results-dir results --num-examples 10
```

```bash
python compare_results.py
```

`evaluate.py` writes `results/evaluation_metrics.json`, `qualitative_examples.md` and `test_predictions.json` (all 270 LSTM predictions).

`compare_results.py` writes `results/comparison_metrics.json` (BLEU/ROUGE for all three systems plus the LLM token-cost estimate) and `results/qualitative_comparison.md` (side-by-side examples spread across the article-length distribution, per PRD §4.3).

---

## Repository layout

```
data/raw/                     Raw CSV (Role 1)
data/processed/               Splits + vocabularies + metadata (Role 1)
src/preprocess.py             Cleaning, splitting, vocab building (Role 1)
src/model.py                  LSTM encoder–decoder + attention (Role 2)
train.py                      Single training run (Role 2)
scripts/finish_training.py    Train + hyperparameter grid (Role 2)
scripts/export_checkpoint.py  Slim inference-only checkpoint (Role 2)
llm_baseline.py               Gemini zero-shot + few-shot (Role 3)
evaluate.py                   BLEU / ROUGE + examples (Role 4)
compare_results.py            LSTM vs LLM comparison + cost (Role 4)
results/                      Checkpoint, curves, metrics, predictions
requirements.txt              Pinned dependencies
```

---

## Reproducibility

- Seed fixed at **42** in `src/preprocess.py`, `train.py` and `scripts/finish_training.py` (`src/utils.py:set_seed` seeds Python, NumPy and PyTorch)
- Splits created before any model development; vocabularies built from the training split only
- All dependencies pinned in `requirements.txt`
- Model size, training time and hardware are printed at the end of every training run and stored in `results/training_config.json`

---

## License and attribution

**Dataset.** Kaggle *News Summary* (`sunnysai12345/news-summary`), redistributed under **GPL-2.0** as stated by its publisher. 4,514 rows of Indian and UK news from April–May 2017; copyright in the underlying article text stays with the originating publishers (mainly India Today, Hindustan Times, The Guardian). Non-commercial academic coursework only.

Full provenance, column descriptions, licensing notes and the raw → processed transformation are in **[data/README.md](data/README.md)**. Machine-readable version in `data/processed/metadata.json`.

```
sunnysai12345. "News Summary." Kaggle, 2018.
https://www.kaggle.com/datasets/sunnysai12345/news-summary
Licensed GPL-2.0. Accessed 2026-08-08.
```

**Code.** Academic use for CP468.
