# LSTM vs. LLM Sequence-to-Sequence Model

**CP468 — Artificial Intelligence (Wilfrid Laurier University)** | Spring 2026   
**Contributors:** Safdar, Farhan, Noah, Manahil, Morad

**Task:** **Article Headline Generation**  
> **Input:** A news article body (multiple sentences)  
> **Output:** A concise, informative headline summarizing the article

---
## Model Architecture

**LSTM Encoder-Decoder with Bahdanau Attention**

- **Encoder:** Bidirectional LSTM
- **Attention:** Bahdanau (additive) with padding masking
- **Decoder:** Unidirectional LSTM, attends to encoder outputs at each step
- **Training:** Teacher forcing with optional decay
- **Inference:** Greedy decoding

Built from scratch using only PyTorch standard layers (`nn.LSTM`, `nn.Embedding`, `nn.Linear`). No prebuilt seq2seq pipelines.

---

## Repository Structure

```
cp468-seq2seq-project/
├── data/
│   ├── raw/
│   │   └── dataset.csv              ← Raw dataset (article, headline pairs)
│   └── processed/                   ← Preprocessed train/val/test + vocabularies
│       ├── train.jsonl
│       ├── validation.jsonl
│       ├── test.jsonl
│       ├── source_vocab.json
│       ├── target_vocab.json
│       └── metadata.json
├── models/                          ← Saved model checkpoints
├── results/                         ← Training curves, metrics, predictions
├── scripts/
│   └── test_dataset.py              ← Quick data pipeline smoke test
├── src/
│   ├── __init__.py
│   ├── dataset.py                   ← PyTorch Dataset + collate (padding/masking)
│   ├── model.py                     ← Encoder-Decoder + Bahdanau Attention
│   ├── preprocess.py                ← Data cleaning, tokenization, vocab, splitting
│   ├── tokenizer.py                 ← Text cleaning + regex tokenization
│   ├── utils.py                     ← Reproducibility (set_seed)
│   └── vocabulary.py                ← Vocabulary class
├── train.py                         ← Training loop (Role 2)
├── evaluate.py                      ← Inference + BLEU/ROUGE (Role 4)
├── llm_baseline.py                  ← LLM API baseline (Role 3 — to be added)
├── requirements.txt                 ← Python dependencies
└── README.md                        ← This file
```

---

## Team Roles & Deliverables

| Role | Member | Status | Deliverable |
|---|---|---|---|
| **Role 1:** Data Pipeline | Manahil | ⚠️ **INCOMPLETE** | Real dataset, vocabularies, train/val/test splits |
| **Role 2:** LSTM Seq2Seq | Safdar | ✅ **COMPLETE** | `src/model.py`, `train.py` — built & tested |
| **Role 3:** LLM Baseline | Morad | ⏳ **PENDING** | `llm_baseline.py` |
| **Role 4:** Evaluation | Noah | ⏳ **PENDING** | Run `evaluate.py`, produce metrics + qualitative analysis |
| **Role 5:** Report & Video | Farhan | ⏳ **PENDING** | 5-page PDF report + 8-min demo video |

> **Note:** Role 1 must replace the placeholder dataset (`data/raw/dataset.csv` currently has only **10 examples**) with a real headline-generation corpus before any training, evaluation, or baseline comparison can proceed. See "Dataset Issue" below.

---

## Setup & Installation

```bash
# Clone the repository
git clone https://github.com/manahilbashir/cp468-seq2seq-project.git
cd cp468-seq2seq-project

# Install dependencies
pip install -r requirements.txt

# (Optional) Install evaluation libraries
pip install sacrebleu rouge-score
```

---

## Usage

### 1. Preprocess the Dataset

> **Prerequisite:** Replace `data/raw/dataset.csv` with a real dataset first.

```bash
python src/preprocess.py
```

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

### 3. Evaluate the Model

```bash
python evaluate.py \
    --checkpoint results/best_model.pt \
    --data-dir data/processed \
    --results-dir results \
    --num-examples 10
```

### 4. Run LLM Baseline (Role 3)

> *To be implemented by Role 3.*

---

## ⚠️ Dataset Issue

1. Replace the CSV with a real headline-generation dataset (e.g., [AG News](https://www.kaggle.com/datasets/amananandrai/ag-news-classification-dataset), [BBC News](https://www.kaggle.com/datasets/hgultekin/bbcnewsarchive), or [CNN/DailyMail](https://huggingface.co/datasets/cnn_dailymail))
2. Re-run `python src/preprocess.py`
3. Verify vocabulary sizes are in the thousands

---

## License

Dataset license: **[TO BE FILLED BY ROLE 1]**  
Code: For academic use in CP468 course project.
