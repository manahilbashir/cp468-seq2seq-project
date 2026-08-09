# Ablation B — free-running validation loss

Kept as evidence for the report. **This is not the shipped model.**

`<unk>` was already masked out of the loss here (the fix from ablation A), but
`validate_epoch` still computed validation loss with
`teacher_forcing_ratio=0.0` — the decoder conditioning on its own predictions —
while training loss was teacher forced.

## What happened

6 epochs, early-stopped, **best epoch 1**, validation loss 5.9266, 14.3 min on
CPU.

| Epoch | Train loss | Val loss (free-running) |
|---|---|---|
| 1 | 6.4858 | **5.9266** ← selected |
| 2 | 5.8749 | 6.6127 |
| 3 | 5.6425 | 6.4821 |
| 4 | 5.3488 | 6.2433 |
| 5 | 5.0614 | 6.2485 |
| 6 | 4.7725 | 6.2747 |

The `<unk>` collapse was gone — 0 of 40 test predictions were empty. But the
selected epoch-1 checkpoint emits the identical string `delhi to to to to in ' s`
for every article, because it had barely trained.

## Why

Teacher-forced training loss and free-running validation loss measure different
things. As the decoder grows confident but is not yet accurate, exposure bias
compounds across timesteps and free-running loss rises even while the model
genuinely improves. Validation loss climbed from epoch 2 onward, so early
stopping fired at epoch 6 and model selection kept the least-trained checkpoint.

The fix is to teacher-force validation so it measures the same quantity as
training. That is what `train.py` does now, and it produced the shipped model in
`results/`.

Validation losses are not comparable across the two ablations and the shipped
run: they differ in whether `<unk>` targets are included and in whether decoding
is teacher forced. Compare decoded output and BLEU/ROUGE instead.
