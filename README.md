# CP468 AI Course Project: LSTM vs. LLM Sequence-to-Sequence Model

**Course:** CP468 — Artificial Intelligence (Wilfrid Laurier University)  
**Semester:** Spring 2026  
**Team:** Seq2Seq Research Group  
**Project Type:** Sequence-to-Sequence Modeling — LSTM vs. LLM

---

## Project Topic: Text Simplification

We chose **Text Simplification** as our sequence-to-sequence task:

> **Input:** A complex sentence with advanced vocabulary and syntactic structures  
> **Output:** A simplified version that preserves meaning while improving readability

### Example
| Source (Complex) | Target (Simplified) |
|---|---|
| "The professor requested that everyone remain silent." | "The professor asked everyone to be quiet." |
| "He commenced his journey early in the morning." | "He started his trip early." |
| "They utilized the equipment to complete the task." | "They used the equipment to finish the task." |

Text simplification is a well-studied NLP task with publicly available datasets, making it ideal for comparing a trained-from-scratch LSTM model against modern LLM baselines.

---

## PRD Requirements Status

This section tracks compliance with the course PRD (*Course Project Requirements: Sequence-to-Sequence Modeling — LSTM vs. LLM*).

### ✅ Section 4.1 — LSTM Seq2Seq Model (Your Implementation)

| PRD Requirement | Status | Where Implemented | Notes |
|---|---|---|---|
| Framework: PyTorch or TensorFlow | ✅ Done | `src/model.py`, `train.py` | PyTorch 2.13.0 |
| **No prebuilt seq2seq pipelines** | ✅ Done | `src/model.py` | Built from scratch using `nn.LSTM`, `nn.Embedding` only. No Fairseq, OpenNMT, or HuggingFace Seq2SeqTrainer. |
| Architecture: embedding → LSTM encoder | ✅ Done | `src/model.py` `Encoder` | `nn.Embedding` → BiLSTM |
| Architecture: bidirectional encoder encouraged | ✅ Done | `src/model.py` `Encoder` | `bidirectional=True` |
| Architecture: attention mechanism | ✅ Done | `src/model.py` `BahdanauAttention` | Bahdanau (additive) attention with masking |
| Architecture: LSTM decoder | ✅ Done | `src/model.py` `Decoder` | Unidirectional LSTM, attends to encoder outputs |
| Architecture: output projection | ✅ Done | `src/model.py` `Decoder` | `nn.Linear(hidden + context, vocab_size)` |
| Reproducibility: fixed random seeds | ✅ Done | `src/utils.py` | `set_seed(42)` covers Python, NumPy, PyTorch, CUDA |
| Reproducibility: requirements.txt | ✅ Done | `requirements.txt` | Pinned versions |
| Reproducibility: README with exact commands | ✅ Done | This file | See Usage section below |
| Report model size (parameter count) | ✅ Done | `train.py` | `model.count_parameters()` printed at start |
| Report training time | ✅ Done | `train.py` | Total time logged at end of training |
| Report hardware used | ✅ Done | `train.py` | Device (CPU/GPU name) printed at start |

### ⏳ Section 4.2 — LLM Baseline

| PRD Requirement | Status | Owner | Notes |
|---|---|---|---|
| Use LLM on identical test set | ⏳ Pending | **Role 3** | Script `llm_baseline.py` needed |
| Test zero-shot setting | ⏳ Pending | **Role 3** | |
| Test few-shot (k=3–5) setting | ⏳ Pending | **Role 3** | |
| At least 2 prompt variants | ⏳ Pending | **Role 3** | |
| Include exact prompts in report | ⏳ Pending | **Role 3** | |
| Estimate/report cost (USD or GPU-hours) | ⏳ Pending | **Role 3** | |

### ⏳ Section 4.3 — Evaluation

| PRD Requirement | Status | Where / Owner | Notes |
|---|---|---|---|
| Automatic metrics on test set: LSTM | ✅ Code ready | `evaluate.py` | BLEU + ROUGE computed; **blocked by dataset size** |
| Automatic metrics on test set: LLM | ⏳ Pending | **Role 4** | Needs Role 3 output |
| Automatic metrics: ablations | ⏳ Pending | **Role 2 + 4** | Run after training on real data |
| Qualitative: 10+ side-by-side examples | ✅ Code ready | `evaluate.py` | Markdown table generated; **blocked by dataset size** |
| Categorize errors (repetition, hallucination, OOV, fluency vs adequacy) | ⏳ Pending | **Role 4** | Manual analysis needed |

### ⏳ Section 5 — Deliverables

| PRD Requirement | Status | Owner | Notes |
|---|---|---|---|
| System Report PDF (5 pages) | ⏳ Pending | **Role 5** | Needs results from Roles 2–4 |
| Git Repository (public, reproducible) | ✅ Done | All | This repo |
| README with setup & execution instructions | ✅ Done | **Role 2** | See below |
| Pinned dependencies | ✅ Done | `requirements.txt` | |
| Fixed random seeds | ✅ Done | `src/utils.py` | |
| 8-minute demo video | ⏳ Pending | **Role 5** | |
| Contribution statement appendix | ⏳ Pending | **Role 5** | |
| AI-use disclosure appendix | ⏳ Pending | **Role 5** | |

### ⏳ Suggested Discussions (for Report)

| Discussion Topic | Status | Owner |
|---|---|---|
| Quantitative gap (LSTM vs LLM) | ⏳ Pending | **Role 5** |
| Why the gap exists (course concepts) | ⏳ Pending | **Role 5** |
| Failure mode contrast | ⏳ Pending | **Role 4** |
| Fairness of comparison | ⏳ Pending | **Role 5** |
| Engineering trade-offs (cost, latency, privacy, deployment) | ⏳ Pending | **Role 5** |
| Limitations & ethics (metrics, bias, contamination, compute) | ⏳ Pending | **Role 5** |

---

## Team Roles & Status

| Role | Member | Status | Deliverable |
|---|---|---|---|
| **Role 1:** Data Pipeline & Repository Lead | *TBD* | ⚠️ **INCOMPLETE — BLOCKS ALL OTHER ROLES** | See "Role 1 Remaining Tasks" below |
| **Role 2:** Custom LSTM Seq2Seq Engineer | *You* | ✅ **COMPLETE** | `src/model.py`, `train.py` — fully implemented and tested |
| **Role 3:** LLM Baseline & Prompting Lead | *TBD* | ⏳ **PENDING** | `llm_baseline.py` |
| **Role 4:** Evaluation & Qualitative Error Lead | *TBD* | ⏳ **PENDING** | Run `evaluate.py`, analyze errors, produce comparison tables |
| **Role 5:** Report Lead & Video Producer | *TBD* | ⏳ **PENDING** | 5-page PDF + 8-min demo video |

---

## ⚠️ CRITICAL ISSUE: Dataset is a Placeholder

### What Role 1 Actually Delivered

The current dataset contains **only 10 toy examples** (8 train / 1 validation / 1 test):

```
data/raw/dataset.csv (10 lines):
"This is a complicated sentence.","This is a simple sentence."
"The weather is extremely cold today.","It is very cold today."
"Please provide a detailed explanation.","Please explain."
"He commenced his journey early in the morning.","He started his trip early."
"The examination was difficult for many students.","The test was hard for many students."
"She purchased a new computer yesterday.","She bought a new computer yesterday."
"The child was frightened by the loud noise.","The child was scared by the noise."
"They utilized the equipment to complete the task.","They used the equipment to finish the task."
"The meeting was postponed because of the storm.","The meeting was delayed because of the storm."
"The professor requested that everyone remain silent.","The professor asked everyone to be quiet."
```

**Metadata:**
- Total clean examples: **10**
- Training examples: **8**
- Validation examples: **1**
- Test examples: **1**
- Source vocabulary size: **8 tokens**
- Target vocabulary size: **9 tokens**

### Why This Blocks Everyone

| Role | Task | Blocked? | Reason |
|---|---|---|---|
| **Role 2** (LSTM Engineer) | Train model, tune hyperparameters, log curves | ✅ **YES** | 8 training examples = instant overfitting. No meaningful training curves. No hyperparameter tuning possible. |
| **Role 3** (LLM Baseline) | Run LLM on test set in zero/few-shot | ✅ **YES** | 1 test example is statistically meaningless. Cannot compute BLEU/ROUGE with n=1. |
| **Role 4** (Evaluation) | Compute BLEU/ROUGE, qualitative analysis | ✅ **YES** | Metrics require sufficient test set size. 10 examples is far below any credible threshold. |
| **Role 5** (Report) | Write 5-page report with results | ✅ **YES** | No results to report. |

### Role 1 — Remaining Tasks (Checklist)

Role 1 is **not finished**. The following tasks from the PRD and role description are still outstanding:

- [ ] **Replace placeholder dataset** with a real, publicly available, citable text simplification corpus (e.g., WikiAuto, ASSET, TurkCorpus)
- [ ] **Document dataset license** in the README and report
- [ ] **Ensure train/val/test split has sufficient size** — recommend minimum 5K–10K training examples for meaningful results
- [ ] **Re-run `python src/preprocess.py`** to regenerate vocabularies from the new dataset
- [ ] **Verify vocabulary sizes** are in the thousands (not single digits)
- [ ] **Verify no data leakage** — confirm vocabularies are built from training split only (code already does this; just verify after re-running)
- [ ] **Update README** with dataset source, citation, and license information
- [ ] **Confirm `requirements.txt` is complete** — Role 1 should verify all dependencies install cleanly in a fresh environment
- [ ] **Add dataset download script or instructions** if the dataset cannot be redistributed directly

**Recommended datasets (publicly available, citable):**

| Dataset | Size | Description | License | Ease of Use |
|---|---|---|---|---|
| **WikiLarge / WikiAuto** | ~108K pairs | Wikipedia sentence pairs (complex → simple) | CC BY-SA | ⭐ Easiest |
| **ASSET** | ~2K sentences | Multiple reference simplifications | CC BY 4.0 | ⭐ Easy |
| **TurkCorpus** | ~2K sentences | 8 reference simplifications per sentence | Research use | ⭐ Easy |
| **Newsela** | ~10K articles | News articles at 4 reading levels | Requires application | ⭐ Harder |

> **Recommendation:** Download **WikiAuto** via HuggingFace `datasets` library. It is the largest, easiest to access, and has a permissive license.

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/manahilbashir/cp468-seq2seq-project.git
cd cp468-seq2seq-project

# Install dependencies
pip install -r requirements.txt

# (Optional) Install evaluation libraries for Role 4
pip install sacrebleu rouge-score
```

---

## Usage

### 1. Preprocess the Dataset

> **Prerequisite:** Replace `data/raw/dataset.csv` with a real dataset first (see Role 1 tasks above).

```bash
python src/preprocess.py
```

This will:
- Clean and tokenize the data
- Split into train (80%) / validation (10%) / test (10%)
- Build source and target vocabularies from training data only
- Save processed files to `data/processed/`

### 2. Train the LSTM Seq2Seq Model

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

**Outputs:**
- `results/best_model.pt` — Best model checkpoint (lowest validation loss)
- `results/training_config.json` — Hyperparameters and model config
- `results/training_history.json` — Loss and perplexity per epoch
- `results/training_curves.png` — Matplotlib plot of loss + perplexity curves

### 3. Evaluate the Model

```bash
python evaluate.py \
    --checkpoint results/best_model.pt \
    --data-dir data/processed \
    --results-dir results \
    --num-examples 10
```

**Outputs:**
- `results/evaluation_metrics.json` — BLEU and ROUGE scores
- `results/qualitative_examples.md` — Side-by-side comparison table
- `results/test_predictions.json` — All predictions on the test set

### 4. Run LLM Baseline (Role 3)

> *To be implemented by Role 3. The script should:*
> - Load the same test set
> - Query an LLM API (Claude/GPT/Gemini) with at least 2 prompt variants
> - Test zero-shot and few-shot (k=3-5) settings
> - Compute BLEU/ROUGE using the same metrics as `evaluate.py`
> - Report API cost in USD

---

## Model Architecture

### LSTM Encoder-Decoder with Bahdanau Attention

```
Source Tokens
     │
     ▼
┌─────────────────┐
│   Embedding     │  (vocab_size → embedding_dim)
└────────┬────────┘
         ▼
┌─────────────────┐
│  BiLSTM Encoder │  (2 layers, bidirectional)
│                 │  → encoder_outputs: (batch, src_len, hidden×2)
│                 │  → hidden/cell: (num_layers, batch, hidden)
└────────┬────────┘
         │
         ├──→ Attention computes context vector at each decoder step
         │
         ▼
┌─────────────────┐
│ LSTM Decoder    │  (2 layers, unidirectional)
│ + Attention     │  Input: [prev_token_embed || context_vector]
│                 │  Output: vocabulary logits
└────────┬────────┘
         ▼
   Predicted Tokens
```

**Key Features:**
- **Bidirectional LSTM Encoder** captures context from both directions
- **Bahdanau (Additive) Attention** allows the decoder to focus on relevant source tokens at each step
- **Teacher Forcing** during training (with optional decay)
- **Greedy Decoding** during inference
- **Gradient Clipping** for training stability
- **Early Stopping** based on validation loss

**Model Size:** ~17M parameters (with default config: embedding_dim=256, hidden_dim=512, 2 layers)

---

## Training Pipeline

```
Raw CSV (source, target)
    │
    ▼
Clean & Tokenize ──→ Remove too short/long examples
    │
    ▼
Split ─────────────→ Train (80%) / Val (10%) / Test (10%)
    │                    (NO data leakage — vocab built from train only)
    ▼
Build Vocabulary ──→ source_vocab.json + target_vocab.json
    │                    (min_freq=2, max_size=30K)
    ▼
Encode ────────────→ Convert tokens to IDs (+ <bos>, <eos>)
    │
    ▼
Save ──────────────→ train.jsonl, validation.jsonl, test.jsonl
```

---

## Reproducibility

All scripts use a fixed random seed (default: 42) via `utils.set_seed()`:
- Python `random`
- NumPy
- PyTorch (CPU + CUDA)
- `PYTHONHASHSEED` environment variable

Run the exact same command twice → identical results.

---

## Evaluation Metrics

| Metric | Purpose | Notes |
|---|---|---|
| **BLEU** | N-gram overlap between prediction and reference | Standard for generation tasks; reports BLEU-4 |
| **ROUGE-1** | Unigram overlap | Measures content coverage |
| **ROUGE-2** | Bigram overlap | Measures fluency |
| **ROUGE-L** | Longest common subsequence | Captures sentence structure |
| **Perplexity** | Model confidence | Computed from validation loss |

---

## Hyperparameter Tuning Guide

| Parameter | Default | Tune Range | Effect |
|---|---|---|---|
| `embedding_dim` | 256 | 128–512 | Higher = richer token representations |
| `hidden_dim` | 512 | 256–1024 | Higher = more model capacity |
| `num_layers` | 2 | 1–4 | More layers = deeper representations |
| `dropout` | 0.3 | 0.1–0.5 | Higher = more regularization |
| `learning_rate` | 0.001 | 0.0001–0.01 | Lower = slower but more stable |
| `batch_size` | 32 | 16–128 | Larger = faster, more stable gradients |
| `teacher_forcing_ratio` | 1.0 | 0.5–1.0 | Lower = exposes model to own errors early |
| `teacher_forcing_decay` | 0.0 | 0.0–0.05 | Gradually reduce teacher forcing |

**Tuning Strategy:**
1. Start with defaults
2. If overfitting → increase dropout, reduce hidden_dim, add decay
3. If underfitting → increase hidden_dim, num_layers, or learning_rate
4. Always monitor validation perplexity — stop when it stops improving

---

## License & Attribution

- Dataset license: **[TO BE FILLED BY ROLE 1]**
- Code: For academic use in CP468 course project
- If using WikiAuto: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- If using Newsela: Requires institutional access agreement

---

## Contact

For questions about the repository, open an issue on GitHub or contact the team lead.
