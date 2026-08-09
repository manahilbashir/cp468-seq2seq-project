# CP468: LSTM vs LLM for Headline Generation

**Course:** CP468 - Artificial Intelligence · Wilfrid Laurier University · Spring 2026
**Team:** Safdar · Farhan · Noah · Manahil · Morad

---

## What this project is

Given a news article, generate a one-line headline.

We built two systems for that task and compared them on the same test set:

1. **LSTM seq2seq, written from scratch** - bidirectional LSTM encoder → Bahdanau attention → LSTM decoder, trained only on our dataset
2. **LLM baseline** - Gemini prompted to write headlines for the same articles

We did not expect to beat the LLM. The goal was to measure the gap and explain it: model size, pretraining, and the trade-offs in cost, latency, control and offline use.

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

All the code has been written and run, and the figures the report needs are in `results/`.

### Training run

12 epochs with early stopping; best validation loss 5.0501 at epoch 7; 31.5 minutes on an Apple M4 CPU. Curves, logs and config are in `results/`.

The model trains but generates poor headlines. Across the first 40 test articles it produced 40 different non-empty predictions, so it is conditioning on its input, but the output is weak and often repetitive. With only 2,691 examples, validation loss bottomed out at epoch 7 and rose after that while training loss kept falling - standard overfitting.

### Two bugs we found and fixed during training

Each one is kept as a reproducible ablation under `results/`, with its own README.

| Ablation | Symptom | Fix |
|---|---|---|
| `ablation_unk_weight_1.0/` | Decoder emitted `<unk>` almost exclusively; 34 of 40 predictions were empty once special tokens were stripped | `--unk-loss-weight 0.0`. `<unk>` was 13.91% of training headline tokens - about 4x the most frequent real word - so always predicting `<unk>` was the lowest-loss option |
| `ablation_freerunning_val_loss/` | Early stopping picked the epoch-1 checkpoint, which produced the same headline for every article | Validation loss is now computed with teacher forcing so it matches training conditions and ranks checkpoints usefully |

---

## Final results (270 test examples)

All three systems are scored on the same 270 test examples with the same settings. Source: `results/comparison_metrics.json`.

| System | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| LSTM + attention | 0.44 | 0.0821 | 0.0117 | 0.0781 |
| Gemini zero-shot | 10.46 | 0.4536 | 0.2050 | 0.3870 |
| Gemini few-shot (k=3) | 10.70 | 0.4388 | 0.1944 | 0.3758 |

The gap is large and consistent: the LLM scores about 5.5x higher on ROUGE-1 and 24x higher on BLEU.

### Ablations, same 270 examples

| Run | What changed | BLEU | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|
| Shipped | biLSTM + Bahdanau attention | 0.44 | 0.0821 | 0.0781 |
| A · `ablation_unk_weight_1.0/` | `<unk>` left in the loss | 0.00 | 0.0067 | 0.0067 |
| B · `ablation_freerunning_val_loss/` | free-running validation loss | no checkpoint saved - training-curve finding only | | |
| C · attention removed | `--no-attention` | not yet run | | |

### Does the gap depend on article length?

Equal-count terciles of the test split, ROUGE-1:

| Bucket | Articles | Source tokens | LSTM | Zero-shot | Few-shot |
|---|---|---|---|---|---|
| 1 of 3 | 90 | 19-154 | 0.0851 | 0.4899 | 0.4736 |
| 2 of 3 | 90 | 154-251 | 0.0846 | 0.4318 | 0.4176 |
| 3 of 3 | 90 | 252-347 | 0.0767 | 0.4392 | 0.4252 |

No - the gap is 5.8x on the shortest bucket and 5.7x on the longest. Both systems drop off a little with length and neither collapses. The LSTM is not specifically bad on long articles, it is bad everywhere.

### Failure modes (all 270, `scripts/error_analysis.py`)

| Category | LSTM | Zero-shot | Few-shot |
|---|---|---|---|
| Repetition loop | 41.5% | 0.0% | 0.0% |
| Over the prompt's 15-word limit | 24.4% | 1.1% | 0.0% |
| Content >=50% absent from the article | 84.8% | 4.1% | 5.2% |
| Mean length (words; references 9.19) | 14.42 | 11.29 | 10.60 |

The LSTM's repetition and over-generation show up clearly. The LLM failures we expected mostly did not: Gemini hallucinates at 4-5% by this proxy and broke the format rule 3 times zero-shot, never few-shot. Per-category sample IDs are in `results/error_analysis.md`.

**Note on BLEU casing.** BLEU uses `lowercase=True` (`evaluate.py:147`) because preprocessing lowercases the corpus (`src/tokenizer.py`). The LSTM can only produce lowercase while the references and Gemini outputs keep capitalization, so case-sensitive BLEU would penalize a preprocessing choice rather than prediction quality. ROUGE is unaffected - `rouge_score` lowercases internally.

### Findings for the report

- **Zero-shot vs few-shot depends on the metric.** Few-shot wins narrowly on BLEU (10.70 vs 10.46), but zero-shot wins on all three ROUGE scores. The few-shot examples in `llm_baseline.py` use Western title-case phrasing while this corpus is terser, so they push the model away from the reference style - ROUGE picks that up through content overlap and BLEU does not. Neither setting is better across the board.
- **BLEU is a weak metric here.** On ~10-token headlines with one reference, the brevity penalty and 4-gram precision make it unstable, which is why the two Gemini settings rank differently across metrics. ROUGE should lead the analysis and we should say so.
- **Repetition is the LSTM's main failure.** From `results/qualitative_comparison.md`: `india ' s son ' s son ' s son ' s son`. Greedy decoding plus an underfit model settles into high-frequency loops. Across all 270 test examples, 41.5% of predictions have under 70% unique bigrams (`results/error_analysis.md`).

**LLM cost: $0.0951 total** - 540 requests at the `gemini-3.5-flash-lite` rates of $0.30/1M input and $2.50/1M output ([pricing](https://ai.google.dev/gemini-api/docs/pricing), retrieved 2026-08-08), or $0.176 per 1,000 requests. Token counts are estimated at 4 characters per token, so treat it as a ballpark.

The whole baseline cost about ten cents, so cost is not a good argument for the small model - offline use, privacy, latency and control are.

---

## LSTM architecture (Role 2)

Standard PyTorch layers only (`nn.LSTM`, `nn.Embedding`, `nn.Linear`) - no Fairseq, OpenNMT or HuggingFace `Seq2SeqTrainer`.

- **Encoder:** bidirectional LSTM (+ pack/pad)
- **Attention:** Bahdanau (additive), padding masked
- **Decoder:** unidirectional LSTM, attends each step
- **Training:** teacher forcing (optional decay), grad clipping, early stopping, checkpointing
- **Inference:** greedy decoding
- **Ablation switch:** `--no-attention` replaces the per-step attention context with the encoder's final bidirectional state held constant across timesteps - the classic recurrent bottleneck. Drops 197,376 parameters (6,469,599 → 6,272,223) and changes nothing else.

6,469,599 trainable parameters · 31.5 min on an Apple M4 CPU.

---

## Setup

```bash
git clone https://github.com/manahilbashir/cp468-seq2seq-project.git
cd cp468-seq2seq-project

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Tested on Python 3.9 / macOS.

---

## How to run

Run from the repository root, in order. All outputs are already committed, so this is for reproduction only. Step 1 needs the Kaggle dataset, step 3 a Gemini API key.

### 1. Preprocess *(Role 1)*

```bash
python src/preprocess.py
```

Writes the three splits, both vocabularies and `metadata.json` into `data/processed/`. Vocabularies are built from the training split only. Seed 42.

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

Writes `results/best_model.pt`, `training_curves.png`, `training_history.json`, `training_config.json`. ~31.5 min on an Apple M4 CPU. Add `| tee results/training_log.txt` to save the per-epoch log. Pass `--unk-loss-weight 1.0` to reproduce the collapse in `results/ablation_unk_weight_1.0/`.

Optional hyperparameter grid:

```bash
python scripts/finish_training.py --quick   # 2 configs
```

### 3. LLM baseline *(Role 3)*

Gemini on the same test split, zero-shot and 3-example few-shot per PRD §4.2. Both prompts are at the top of `llm_baseline.py`.

```bash
export GEMINI_API_KEY=your_key_here

python llm_baseline.py --limit 0     # 0 = full test set
```

Writes `results/llm_outputs.jsonl`.

### 4. Evaluate and compare *(Role 4)*

```bash
python evaluate.py --checkpoint results/best_model_inference.pt --data-dir data/processed --results-dir results --num-examples 10
```

```bash
python compare_results.py
```

`evaluate.py` writes `evaluation_metrics.json`, `qualitative_examples.md` and `test_predictions.json`. `compare_results.py` writes `comparison_metrics.json` (all three systems, length buckets, USD cost) and `qualitative_comparison.md`.

### 5. Error analysis

```bash
python scripts/error_analysis.py
```

Writes `results/error_analysis.md` and `.json`. Needs no checkpoint - it reads the committed prediction files.

### Ablations

Score the `<unk>`-collapse ablation, which ships with a checkpoint:

```bash
python evaluate.py --checkpoint results/ablation_unk_weight_1.0/best_model_inference.pt --data-dir data/processed --results-dir results/ablation_unk_weight_1.0
```

Train the attention ablation (~30 min):

```bash
python train.py --data-dir data/processed --results-dir results/ablation_no_attention --embedding-dim 128 --hidden-dim 256 --num-layers 1 --dropout 0.3 --batch-size 32 --epochs 30 --learning-rate 0.001 --teacher-forcing-ratio 1.0 --teacher-forcing-decay 0.02 --clip-grad 1.0 --patience 5 --seed 42 --unk-loss-weight 0.0 --no-attention
```

---

## Repository layout

```
data/raw/                     Raw CSV (Role 1)
data/processed/               Splits + vocabularies + metadata (Role 1)
src/preprocess.py             Cleaning, splitting, vocab building (Role 1)
src/model.py                  LSTM encoder - decoder + attention (Role 2)
train.py                      Single training run (Role 2)
scripts/finish_training.py    Train + hyperparameter grid (Role 2)
scripts/export_checkpoint.py  Slim inference-only checkpoint (Role 2)
llm_baseline.py               Gemini zero-shot + few-shot (Role 3)
evaluate.py                   BLEU / ROUGE + examples (Role 4)
compare_results.py            LSTM vs LLM + length buckets + USD cost (Role 4)
scripts/error_analysis.py     Failure modes, all 270 (Role 4)
results/                      Checkpoint, curves, metrics, predictions
requirements.txt              Pinned dependencies
```

---

## Reproducibility

- Seed fixed at 42 in `src/preprocess.py`, `train.py` and `scripts/finish_training.py` (`src/utils.py:set_seed` seeds Python, NumPy and PyTorch)
- Splits created before any model development; vocabularies built from the training split only
- All dependencies pinned in `requirements.txt`
- Model size, training time and hardware are printed at the end of every training run and stored in `results/training_config.json`

---

## License

Dataset: Kaggle [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) (`sunnysai12345`), GPL-2.0, non-commercial academic use - details in [data/README.md](data/README.md). Code: academic use for CP468.
