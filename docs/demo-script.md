# 8-minute demo video - shot list and script

Everything below is runnable from the repo root today. No step needs a Gemini
API key, a GPU, or a retrain: all artefacts are committed. Total live runtime if
you run the commands on camera is about 15 seconds.

**Setup before recording**

```bash
source .venv/bin/activate
```

Terminal at a large font, repo root, `results/` open in a file browser or second
pane. Have `results/training_curves.png` and `results/error_analysis.md` open in
tabs so you are not hunting for them mid-take.

---

## 0:00-0:50 · Task and dataset

**Show:** `data/processed/metadata.json`, then one raw row.

**Say:** Task is headline generation - full news article in, one-line headline
out. Kaggle *News Summary*, GPL-2.0, cited in `data/README.md`. 4,514 raw rows
become 2,691 usable examples after cleaning; 1,649 rows are dropped for
exceeding 400 tokens. Split 2,152 / 269 / 270, seed 42, **split before the
vocabulary is built** so there is no leakage.

```bash
cat data/processed/metadata.json
```

**Point out on screen:** `"vocabulary_built_from": "training split only"`. That
is the anti-leakage claim, visible as data rather than asserted.

---

## 0:50-2:20 · Architecture

**Show:** `src/model.py`, scrolling to each class in turn.

**Say:** Written from scratch on standard PyTorch layers - `nn.LSTM`,
`nn.Embedding`, `nn.Linear`. No Fairseq, no OpenNMT, no HuggingFace trainer.
Four pieces, in the order the PRD asks for:

| Scroll to | Say |
|---|---|
| `class Encoder` | Embedding → bidirectional LSTM, packed so padding never enters the recurrence |
| `class BahdanauAttention` | Additive attention, `v^T tanh(W_h h_i + W_s s_t)`, padding masked to `-inf` before the softmax |
| `class Decoder` | Unidirectional LSTM, recomputes attention every timestep, context concatenated into both the LSTM input and the output projection |
| `count_parameters` | 6,469,599 trainable parameters |

**Say:** 6.47M parameters, 31.5 minutes on one Apple M4 CPU. Hold that number - it is the whole point of the comparison later.

---

## 2:20-3:40 · Training, and two real bugs

**Show:** `results/training_curves.png`, then the two ablation READMEs.

**Say:** 12 epochs, early stopping, best validation loss 5.0501 at epoch 7.
Validation loss falls for 7 epochs then rises while training loss keeps
falling - textbook overfitting with a clean model-selection point.

**This is the strongest part of the project - spend the time here.** Two
defects were found and each is preserved as a reproducible ablation:

```bash
cat results/ablation_unk_weight_1.0/README.md
```

**Say:** First, the decoder collapsed onto emitting `<unk>` at nearly every
position. Cause is measurable, not guessed: `<unk>` is 13.91% of training
headline tokens, about 4x the most frequent real word, so constant `<unk>` is
the lowest-loss constant policy available. Fix was to zero-weight `<unk>` in the
loss. Scored on the test set it is ROUGE-1 0.0067 against the shipped 0.0821 - a 12x drop.

**Say:** Second, validation loss was computed free-running while training was
teacher forced. Those measure different things, so validation loss rose from
epoch 2 and early stopping selected the epoch-1 checkpoint, which emitted one
identical headline for every article. Fix was to teacher-force validation so it
ranks checkpoints meaningfully.

---

## 3:40-5:10 · Results

**Show:** run it live, it takes about 8 seconds.

```bash
python compare_results.py
```

**Say:** All three systems, same 270 test examples, same scoring code.

| System | BLEU | ROUGE-1 |
|---|---|---|
| LSTM + attention | 0.44 | 0.0821 |
| Gemini zero-shot | 10.46 | 0.4536 |
| Gemini few-shot (k=3) | 10.70 | 0.4388 |

**Say:** Roughly 5.5x on ROUGE-1. And note the ranking flips between metrics - few-shot wins BLEU, zero-shot wins all three ROUGE measures. Neither prompt is
uniformly better, which is exactly why we do not lead with BLEU on 10-token
headlines against a single reference.

**Point at the bucket table that just printed:** The gap is consistent across
article length - 5.8x on the shortest tercile, 5.7x on the longest. The LSTM is
not worse on long articles; it is uniformly poor.

---

## 5:10-6:40 · Failure modes, measured

**Show:** `results/error_analysis.md` and `results/qualitative_comparison.md`.

```bash
python scripts/error_analysis.py
```

**Say:** The assignment warns not to just assert the typical finding, so we
measured every category across all 270 examples.

**Confirmed:** the LSTM loops. 41.5% of its predictions have under 70% unique
bigrams - `india ' s son ' s son ' s son`. It over-generates too, 14.4 words
against a 9.2-word reference average.

**Refuted, and this is our most interesting result:** the expected LLM failures
mostly do not happen here. Unsupported content 4-5% against the LSTM's 85%.
Format compliance - the prompt says "under 15 words" - violated 3 times out of
270 zero-shot, and **zero** times few-shot.

**Say:** And there is the trade-off in one line: few-shot buys perfect format
compliance and pays for it in ROUGE. The title-case exemplars steer Gemini away
from this corpus's terser style.

**Show two side-by-side rows** from `qualitative_comparison.md` - row 1
(American Tourister) for the repetition loop, row 9 (Jadhav) for the extreme
case.

---

## 6:40-7:30 · Why the gap, and the honest cost story

**Say, connecting to course concepts:**

- **Capacity and data.** 6.47M parameters trained on 2,152 examples, against a
  model pretrained on a corpus we cannot enumerate. Transfer learning is the
  single biggest difference.
- **A hard ceiling we can quantify.** 23.56% of test reference headline tokens
  are outside our 2,527-word target vocabulary, and 95.6% of references contain
  at least one. The LSTM **cannot** emit them under any decoding strategy. The
  LLM's subword vocabulary has no such limit. A large part of the gap is
  structural, not learned.
- **The bottleneck.** Attention exists precisely to avoid forcing a 240-token
  article through one fixed vector. `--no-attention` is in the repo to test
  that directly.

**Say on cost - do not oversell it:** the entire LLM baseline cost **$0.0951**.
540 requests, about ten cents. Cost is not the argument for the small model. The
real arguments are offline operation, no per-request dependency, data never
leaving the machine, and full control of the output distribution.

---

## 7:30-8:00 · Limitations

**Say:** Three, honestly:

1. **Data scale is the binding constraint.** 2,152 training examples is far too
   few for a from-scratch seq2seq model. Dropping the 400-token cap or using the
   dataset's shorter `text` column would roughly double or triple the usable
   set. We documented the trade-off rather than taking it, because changing it
   invalidates the trained model and the measured baseline.
2. **The comparison is unfair in both directions.** Gemini has almost certainly
   seen 2017 Indian news; our test set is not clean with respect to its
   pretraining. Equally, the LSTM is a specialised model that runs offline for
   free. A fairer middle point is fine-tuning a small pretrained transformer.
3. **Metric limits.** Single reference, ~10-token outputs, BLEU unstable at that
   length; the unsupported-content figure is a surface-overlap proxy, not a
   hallucination detector.

---

## Pre-flight checklist

```bash
source .venv/bin/activate
python compare_results.py && python scripts/error_analysis.py
git status --short
```

- [ ] Both commands run clean on camera
- [ ] `results/training_curves.png` opens
- [ ] GitHub repo is **public** - check in a logged-out browser window
- [ ] Slide or terminal showing 6,469,599 params and 31.5 min
- [ ] Under 8:00 on a timed practice run
