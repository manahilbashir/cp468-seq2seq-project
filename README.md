# LSTM vs LLM for Headline Generation

**CP468 Artificial Intelligence · Wilfrid Laurier University · Spring 2026**

**Contributors:** Safdar · Farhan · Noah · Manahil · Morad

A course research project comparing a sequence-to-sequence model built from scratch against a modern pretrained language model on the same task, the same data, and the same metrics.

## Deliverables

| | |
|---|---|
| **Report** | [docs/report.pdf](docs/report.pdf) |
| **Demo video** | [8-minute walkthrough](https://drive.google.com/file/d/19RTTB-FwnYwS2zguLjpoPKNLi3XXlN49/view?usp=sharing) |
| **Code** | This repository, fully reproducible from the commands below |

---

## Research question: How large is the gap?**

Given a news article, generate a one-line headline. We built two systems for that task:

1. **An LSTM encoder-decoder with attention, written from scratch** and trained only on our 2,152-example training split
2. **A pretrained LLM baseline**, Gemini prompted zero-shot and few-shot on the identical test set



## What we found

**The gap is large, consistent, and mostly structural.** The LLM scores about 5.5x higher on ROUGE-1. That ratio holds almost exactly across short and long articles (5.8x on the shortest tercile, 5.7x on the longest), so the LSTM is not specifically weak on long inputs. It is weak everywhere.

**A large share of the gap is a hard ceiling, not a learning failure.** 23.56% of the reference headline tokens in our test set fall outside the model's 2,527-word output vocabulary, and 95.6% of references contain at least one such token. The LSTM cannot emit those words under any decoding strategy. The LLM, decoding over a subword vocabulary, has no equivalent limit. Part of the measured gap was decided by preprocessing before training ever started.

**Data scale was the binding constraint.** With 2,152 training examples, validation loss bottomed out at epoch 7 and rose afterwards while training loss kept falling. The model has enough capacity to memorize the training set and not enough data to generalize from it.

**Two of our three most useful findings came from bugs.** The decoder first collapsed to emitting `<unk>` at nearly every position, which turned out to be rational: `<unk>` was 13.91% of training headline tokens, roughly 4x the most frequent real word, so constant `<unk>` was genuinely the lowest-loss policy available. Separately, computing validation loss free-running while training teacher-forced made the two curves measure different things, and early stopping selected the least-trained checkpoint. Both are preserved as reproducible ablations rather than quietly fixed.

**The textbook LLM failure modes mostly did not appear.** We expected hallucination, over-elaboration, and ignored formatting. Measured across all 270 test examples, Gemini produced unsupported content in 4-5% of cases against the LSTM's 84.8%, and violated the prompt's 15-word limit 3 times zero-shot and never few-shot. We report this because we measured it, not because it was the expected answer.

**Cost is not the argument for the small model.** The entire LLM baseline cost $0.0951. Any honest case for a task-specific model has to rest on offline operation, privacy, latency, and control, not on price.

---

## Dataset

Kaggle [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary) (`sunnysai12345`), GPL-2.0. Indian and UK news from April to May 2017.

| | |
|---|---|
| Raw rows | 4,514 |
| After cleaning, deduplication and length filters | 2,691 |
| Train / validation / test | 2,152 / 269 / 270 |
| Source vocabulary | 15,881 |
| Target vocabulary | 2,527 |

Splits are created with seed 42 **before** any vocabulary is built, and vocabularies come from the training split only, so there is no validation or test leakage. 

## Architecture

Standard PyTorch layers (`nn.LSTM`, `nn.Embedding`, `nn.Linear`)
```
embedding -> bidirectional LSTM encoder -> Bahdanau attention -> LSTM decoder -> output projection
```

- **Encoder:** bidirectional LSTM with packed sequences, so padding never enters the recurrence
- **Attention:** Bahdanau additive, padding masked to `-inf` before the softmax
- **Decoder:** unidirectional LSTM, recomputes attention at every timestep
- **Training:** teacher forcing with optional decay, gradient clipping, early stopping, checkpointing
- **Inference:** greedy decoding
- **Ablation switch:** `--no-attention` replaces the per-step context with the encoder's final bidirectional state held constant, the classic recurrent bottleneck. Removes exactly 197,376 parameters and changes nothing else.

**6,469,599 trainable parameters. 31.5 minutes on an Apple M4 CPU.** 12 epochs with early stopping, best validation loss 5.0501 at epoch 7.

---

## Results

All three systems scored on the same 270 test examples with the same code. Source: `results/comparison_metrics.json`.

| System | BLEU | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|---|
| LSTM + attention | 0.44 | 0.0821 | 0.0117 | 0.0781 |
| Gemini zero-shot | 10.46 | 0.4536 | 0.2050 | 0.3870 |
| Gemini few-shot (k=3) | 10.70 | 0.4388 | 0.1944 | 0.3758 |

### Ablations

| Run | What changed | BLEU | ROUGE-1 | ROUGE-L |
|---|---|---|---|---|
| Shipped | biLSTM + Bahdanau attention | 0.44 | 0.0821 | 0.0781 |
| A · `ablation_unk_weight_1.0/` | `<unk>` left in the loss | 0.00 | 0.0067 | 0.0067 |
| B · `ablation_freerunning_val_loss/` | free-running validation loss | training-curve finding, no checkpoint saved | | |

### Does the gap depend on article length?

Equal-count terciles of the test split, ROUGE-1:

| Bucket | Articles | Source tokens | LSTM | Zero-shot | Few-shot |
|---|---|---|---|---|---|
| 1 of 3 | 90 | 19-154 | 0.0851 | 0.4899 | 0.4736 |
| 2 of 3 | 90 | 154-251 | 0.0846 | 0.4318 | 0.4176 |
| 3 of 3 | 90 | 252-347 | 0.0767 | 0.4392 | 0.4252 |

No. 5.8x on the shortest bucket, 5.7x on the longest. Both systems degrade slightly with length and neither collapses.

### Failure modes, all 270 examples

Measured by `scripts/error_analysis.py`, not asserted.

| Category | LSTM | Zero-shot | Few-shot |
|---|---|---|---|
| Repetition loop | 41.5% | 0.0% | 0.0% |
| Over the prompt's 15-word limit | 24.4% | 1.1% | 0.0% |
| Content >=50% absent from the article | 84.8% | 4.1% | 5.2% |
| Mean length (words; references 9.19) | 14.42 | 11.29 | 10.60 |

The LSTM's repetition is its dominant failure: `india ' s son ' s son ' s son ' s son`. Greedy decoding plus an underfit model settles into high-frequency loops. Per-category sample IDs are in `results/error_analysis.md` so any claim can be spot-checked.

### Notes on the metrics

**Zero-shot versus few-shot depends on the metric.** Few-shot wins narrowly on BLEU (10.70 vs 10.46) while zero-shot wins all three ROUGE scores. The few-shot examples use Western title-case phrasing while this corpus is terser, so they push the model away from the reference style. ROUGE registers that through content overlap and BLEU does not. Neither setting is better across the board.

**BLEU is weak for this task.** On roughly 10-token headlines with a single reference, the brevity penalty and 4-gram precision make it unstable, which is exactly why the two Gemini settings rank differently across metrics. ROUGE leads our analysis and we say so.

**BLEU casing.** BLEU uses `lowercase=True` because preprocessing lowercases the corpus. The LSTM can only produce lowercase while references and Gemini outputs keep capitalization, so case-sensitive BLEU would penalize a preprocessing choice rather than prediction quality. ROUGE is unaffected, since `rouge_score` lowercases internally.

**LLM cost.** $0.0951 total for 540 requests at `gemini-3.5-flash-lite` rates of $0.30/1M input and $2.50/1M output ([pricing](https://ai.google.dev/gemini-api/docs/pricing), retrieved 2026-08-08), or $0.176 per 1,000 requests. Token counts are estimated at 4 characters per token, so treat it as a ballpark.

---

## Limitations

Stated plainly, because they bound every number above.

1. **Data scale.** 2,152 training examples is far too few for a from-scratch seq2seq model. Dropping the 400-token article cap, or using the dataset's shorter `text` column, would roughly double or triple the usable set. We documented that trade-off rather than taking it, because changing it invalidates both the trained model and the measured baseline.
2. **The comparison is unfair in both directions.** Gemini has almost certainly seen 2017 Indian news during pretraining, so our test set is not clean with respect to it. Equally, the LSTM is a specialized model that runs offline at zero marginal cost. A fairer middle point would be fine-tuning a small pretrained transformer.
3. **Metric limits.** One reference per example, roughly 10-token outputs, and BLEU is unstable at that length. The unsupported-content figure is a surface-overlap proxy, not a hallucination detector: a correct synonym or a fair abstraction trips it too.
4. **Narrow provenance.** Two months of 2017, India-weighted coverage. Nothing here should be presented as a general-purpose headline generator.

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

## Reproducing the results

Run from the repository root, in order. All outputs are already committed, so this is for verification. Step 1 needs the Kaggle dataset and step 3 a Gemini API key.

**1. Preprocess**

```bash
python src/preprocess.py
```

Writes the three splits, both vocabularies and `metadata.json` into `data/processed/`. Seed 42.

**2. Train the LSTM** (~31.5 min on CPU)

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

Add `| tee results/training_log.txt` to save the per-epoch log. Pass `--unk-loss-weight 1.0` to reproduce the `<unk>` collapse, or `--no-attention` to train without attention.

**3. LLM baseline**

```bash
export GEMINI_API_KEY=your_key_here
python llm_baseline.py --limit 0     # 0 = full test set
```

Both prompts are at the top of `llm_baseline.py`.

**4. Evaluate and compare**

```bash
python evaluate.py --checkpoint results/best_model_inference.pt --data-dir data/processed --results-dir results --num-examples 10
python compare_results.py
```

**5. Error analysis**

```bash
python scripts/error_analysis.py
```

Needs no checkpoint. It reads the committed prediction files.

**Score the `<unk>` ablation**, which ships with its own checkpoint:

```bash
python evaluate.py --checkpoint results/ablation_unk_weight_1.0/best_model_inference.pt --data-dir data/processed --results-dir results/ablation_unk_weight_1.0
```

---

## Repository layout

```
data/raw/                     Raw CSV
data/processed/               Splits, vocabularies, metadata
src/preprocess.py             Cleaning, splitting, vocabulary building
src/model.py                  LSTM encoder-decoder with attention
src/tokenizer.py              Regex tokenizer and text normalization
src/vocabulary.py             Vocabulary construction and encoding
src/dataset.py                Dataset and padding collate function
train.py                      Training loop
llm_baseline.py               Gemini zero-shot and few-shot
evaluate.py                   BLEU / ROUGE and qualitative examples
compare_results.py            Three-system comparison, length buckets, cost
scripts/error_analysis.py     Measured failure modes across all 270
scripts/export_checkpoint.py  Slim inference-only checkpoint
scripts/finish_training.py    Optional hyperparameter grid
results/                      Checkpoint, curves, metrics, predictions
docs/report.pdf               Final report
requirements.txt              Pinned dependencies
```

The full training checkpoint (`results/best_model.pt`, 74 MiB with optimizer state) is excluded from version control. The inference-only export is committed, so every number above can be reproduced without retraining.

## Reproducibility

- Seed fixed at 42 in `src/preprocess.py`, `train.py` and `scripts/finish_training.py`. `src/utils.py:set_seed` seeds Python, NumPy and PyTorch.
- Splits created before any model development; vocabularies built from the training split only.
- All dependencies pinned in `requirements.txt`.
- Model size, training time and hardware are printed at the end of every training run and stored in `results/training_config.json`.
- Re-running `src/preprocess.py` regenerates all six files in `data/processed/` byte-identical.

## Contributions

| Role | Owner | Main files |
|---|---|---|
| Data pipeline | Manahil | `src/preprocess.py`, `src/tokenizer.py`, `src/vocabulary.py`, `data/` |
| LSTM seq2seq | Safdar | `src/model.py`, `src/dataset.py`, `train.py` |
| LLM baseline | Manahil | `llm_baseline.py` |
| Evaluation | Noah | `evaluate.py`, `compare_results.py`, `scripts/error_analysis.py` |
| Report and video | Farhan | `docs/report.pdf`, demo video |

## License

Dataset: Kaggle [News Summary](https://www.kaggle.com/datasets/sunnysai12345/news-summary), GPL-2.0, non-commercial academic use. Copyright in the underlying article text remains with the originating publishers. Details in [data/README.md](data/README.md).

Code: academic use for CP468.
