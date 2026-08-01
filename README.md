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
| 1 · Data pipeline | Manahil | Incomplete | Real dataset, preprocess, vocab, train/val/test |
| 2 · LSTM seq2seq | Safdar | Model done; train/tune after Role 1 | `src/model.py`, `train.py`, `scripts/finish_training.py` |
| 3 · LLM baseline | Morad | Pending | `llm_baseline.py` |
| 4 · Evaluation | Noah | Pending | Metrics + qualitative analysis via `evaluate.py` |
| 5 · Report & video | Farhan | Pending | PDF report + demo video |

**Blocker for Roles 2–4 training/eval:** Role 1 must replace the 10-example placeholder CSV with a real headline corpus, then re-run preprocessing. Role 2 model code is ready; it cannot be meaningfully trained until that data lands.

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
pip install -r requirements.txt

# Needed for evaluate.py (Role 4)
pip install sacrebleu rouge-score
```

---

## How to run

### 1. Preprocess *(Role 1 — after real data is in place)*

```bash
python src/preprocess.py
```

Expects `data/raw/dataset.csv` with `source,target` columns (article, headline).

---

## After Role 1 cleans the dataset *(Role 2 — do this next)*

Once the placeholder CSV is replaced with a real article→headline corpus:

```bash
# 1) Rebuild splits + vocabularies
python src/preprocess.py

# 2) Check data/processed/metadata.json
#    train examples and vocab sizes should be large (not ~8)

# 3) Train + light hyperparameter tuning (picks best val loss)
python scripts/finish_training.py --preprocess

# Faster smoke grid (2 configs, 15 epochs):
python scripts/finish_training.py --quick --device cpu

# GPU:
python scripts/finish_training.py --device cuda
```

What `scripts/finish_training.py` does:

- Stops if data still looks like the toy placeholder (unless `--allow-tiny-data`)
- Trains a small grid (baseline / smaller LR / more dropout / narrower)
- Saves runs under `results/tuning/<name>/`
- Copies the best run to `results/best_model.pt`, `training_curves.png`, and `tuning_summary.json`

Single-config training (manual):

```bash
python train.py \
  --data-dir data/processed \
  --results-dir results \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --num-layers 2 \
  --dropout 0.3 \
  --batch-size 32 \
  --epochs 50 \
  --learning-rate 0.001 \
  --teacher-forcing-ratio 1.0 \
  --teacher-forcing-decay 0.02 \
  --clip-grad 1.0 \
  --patience 7 \
  --seed 42
```

### Evaluate LSTM *(Role 4)*

```bash
python evaluate.py \
  --checkpoint results/best_model.pt \
  --data-dir data/processed \
  --results-dir results \
  --num-examples 10
```

### LLM baseline *(Role 3 — not implemented yet)*

Will live in `llm_baseline.py` once Role 3 lands.

---

## Repository layout

```
data/raw/                     Raw CSV (Role 1)
data/processed/               Splits + vocabularies (Role 1)
src/model.py                  LSTM encoder–decoder + attention (Role 2)
train.py                      Single training run (Role 2)
scripts/finish_training.py    Train + tune after real data (Role 2)
evaluate.py                   BLEU / ROUGE + examples (Role 4)
llm_baseline.py               LLM comparison (Role 3 — TBD)
results/                      Best checkpoint, curves, tuning summary
requirements.txt              Pinned dependencies
```

---

## License

- **Dataset:** to be filled by Role 1  
- **Code:** academic use for CP468
