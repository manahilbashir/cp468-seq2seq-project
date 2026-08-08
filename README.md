# CP468: LSTM vs LLM for Headline Generation

**Course:** CP468 — Artificial Intelligence · Wilfrid Laurier University · Spring 2026  
**Team:** Safdar · Farhan · Noah · Manahil · Morad

---

## What this project is

We take a **news article** and generate a **short headline**.

| | |
|---|---|
| **Input** | Article body (many sentences) |
| **Output** | One concise headline |

We build **two systems** on the same task and compare them:

1. **LSTM seq2seq (from scratch)** — bidirectional LSTM encoder → Bahdanau attention → LSTM decoder, trained only on our dataset  
2. **LLM baseline** — a modern language model (API or local) prompted to write headlines on the **same test set**

The goal is not to beat the LLM. It is to measure the gap, understand why it exists (capacity, pretraining, attention vs recurrence), and discuss trade-offs (cost, latency, control, offline use).

**Course deliverables:** public GitHub repo · 5-page report · 8-minute demo video

---

## Team roles

| Role | Owner | Status | Owns |
|------|-------|--------|------|
| 1 · Data pipeline | Manahil | Done | Real dataset (Kaggle News Summary, 2,691 examples), preprocess, vocab, train/val/test |
| 2 · LSTM seq2seq | Safdar | Code done · final training run pending | `src/model.py`, `train.py`, `scripts/finish_training.py` |
| 3 · LLM baseline | Manahil | Done | `llm_baseline.py`, `results/llm_outputs.jsonl` (Gemini, zero-shot + few-shot) |
| 4 · Evaluation | Noah | Scripts ready · needs a checkpoint to run against | `evaluate.py`, `compare_results.py` |
| 5 · Report & video | Farhan | Not started | PDF report + 8-minute demo video — see [docs/report-outline.md](docs/report-outline.md) |

**Current state.** The data pipeline and the LLM baseline are finished. The LSTM
implementation, training loop and evaluation scripts are all written and verified
to run end to end, but the **final training run has not been completed**, so
`results/best_model.pt` is not yet produced and the LSTM-side metrics do not exist
yet.

**To finish the project, in order:**

1. Run the training command in [step 2 below](#2-train-the-lstm) (~2.5 min/epoch on
   CPU; a partial 5-epoch run reached validation loss ≈ 5.17).
2. Run `evaluate.py` and `compare_results.py` ([step 4](#4-evaluate-and-compare)).
3. Write the report using [docs/report-outline.md](docs/report-outline.md).

### LLM baseline results (already measured, 270 test examples)

| Prompt setting | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| Gemini zero-shot | 3.85 | 0.4536 | 0.2050 | 0.3870 |
| Gemini few-shot (k=3) | 2.64 | 0.4388 | 0.1944 | 0.3758 |

Note that **zero-shot outperforms few-shot** here. The few-shot exemplars in
`llm_baseline.py` are Western title-case headlines, while this corpus uses a
terser style, so the examples steer the model away from the reference style.
Note also that BLEU is very low while ROUGE-1 is reasonable: on ~10-token
headlines with a single reference, BLEU's brevity penalty and 4-gram precision
make it largely uninformative. Prefer ROUGE for this task and discuss the
limitation in the report.

---

## LSTM model (Role 2)

Built with standard PyTorch layers only (`nn.LSTM`, `nn.Embedding`, `nn.Linear`) — no Fairseq / OpenNMT / HuggingFace Seq2SeqTrainer.

- **Encoder:** bidirectional LSTM (+ pack/pad)  
- **Attention:** Bahdanau (additive), padding masked  
- **Decoder:** unidirectional LSTM with attention each step  
- **Train:** teacher forcing (optional decay), grad clip, early stopping, checkpointing  
- **Infer:** greedy decoding  

---

## Setup

```bash
git clone https://github.com/manahilbashir/cp468-seq2seq-project.git
cd cp468-seq2seq-project

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

`requirements.txt` pins every dependency (including `sacrebleu`, `rouge_score`
and `certifi`, which the evaluation and LLM scripts import), so no extra
installs are needed. Verified on Python 3.9 / macOS.

---

## How to run

Run these from the repo root, in order. Steps 1–4 reproduce every number in
the report.

### 1. Preprocess *(Role 1)*

```bash
python src/preprocess.py
```

Reads `data/raw/dataset.csv` and writes `train/validation/test.jsonl`,
`source_vocab.json`, `target_vocab.json` and `metadata.json` into
`data/processed/`. Vocabularies are built from the **training split only**,
so there is no validation/test leakage. Seed is fixed at 42.

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
  --seed 42
```

Writes `results/best_model.pt`, `training_curves.png`, `training_history.json`
and `training_config.json`.

Optional light hyperparameter search (trains a grid, promotes the best run by
validation loss):

```bash
python scripts/finish_training.py --quick   # 2 configs
python scripts/finish_training.py           # 4 configs (slow on CPU)
```

### 3. LLM baseline *(Role 3)*

Uses the Gemini API on the **same** `data/processed/test.jsonl`, in two prompt
settings (zero-shot and 3-example few-shot) as required by PRD §4.2. Both exact
prompts live at the top of `llm_baseline.py`.

```bash
export GEMINI_API_KEY=your_key_here

python llm_baseline.py --limit 0     # 0 = full test set
```

Writes `results/llm_outputs.jsonl` with one record per test example containing
the article, reference headline, and both generated headlines.

### 4. Evaluate and compare *(Role 4)*

```bash
# LSTM only: BLEU / ROUGE + qualitative table
python evaluate.py \
  --checkpoint results/best_model.pt \
  --data-dir data/processed \
  --results-dir results \
  --num-examples 10

# LSTM vs LLM (zero-shot and few-shot) on the identical test set
python compare_results.py
```

`compare_results.py` writes `results/comparison_metrics.json` (BLEU/ROUGE for
all three systems plus an LLM token-cost estimate) and
`results/qualitative_comparison.md` (a side-by-side table spread across the
article-length distribution, per PRD §4.3).

---

## Repository layout

```
data/raw/                     Raw CSV (Role 1)
data/processed/               Splits + vocabularies + metadata (Role 1)
src/preprocess.py             Cleaning, splitting, vocab building (Role 1)
src/model.py                  LSTM encoder–decoder + attention (Role 2)
train.py                      Single training run (Role 2)
scripts/finish_training.py    Train + light hyperparameter grid (Role 2)
llm_baseline.py               Gemini zero-shot + few-shot baseline (Role 3)
evaluate.py                   BLEU / ROUGE + examples (Role 4)
compare_results.py            LSTM vs LLM comparison + cost (Role 4)
results/                      Checkpoint, curves, metrics, predictions
requirements.txt              Pinned dependencies
docs/plans/                   Optional interactive TUI demo plan
```

---

## Reproducibility

- Random seed fixed at **42** in `src/preprocess.py`, `train.py` and
  `scripts/finish_training.py` (`src/utils.py:set_seed` seeds Python, NumPy and
  PyTorch).
- Splits are created before any model development and vocabularies are built
  from the training split only.
- All dependencies pinned in `requirements.txt`.
- Model size, training time and hardware are printed at the end of every
  training run and stored in `results/training_config.json`.

---

## License

- **Dataset:** Kaggle *News Summary* (`sunnysai12345/news-summary`), GPL-2.0.
  See `data/processed/metadata.json` for the source URL and license.
- **Code:** academic use for CP468
