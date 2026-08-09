# Ablation B - free-running validation loss

Kept as evidence for the report. Not the shipped model.

`<unk>` was already masked out of the loss here, but validation loss was
computed free-running (the decoder conditioning on its own predictions) while
training loss was teacher forced.

6 epochs, best epoch 1, validation loss 5.9266, 14.3 min on CPU.

| Epoch | Train loss | Val loss (free-running) |
|---|---|---|
| 1 | 6.4858 | 5.9266 ← selected |
| 2 | 5.8749 | 6.6127 |
| 3 | 5.6425 | 6.4821 |
| 4 | 5.3488 | 6.2433 |
| 5 | 5.0614 | 6.2485 |
| 6 | 4.7725 | 6.2747 |

The `<unk>` collapse was gone, but the selected epoch-1 checkpoint emits the
same string `delhi to to to to in ' s` for every article because it had barely
trained. Free-running loss rises from exposure bias even while the model
improves, so early stopping fired at epoch 6 and kept the least-trained
checkpoint.

Fixed by teacher-forcing validation, which is what `train.py` does now.

Validation losses are not comparable across the two ablations and the shipped
run - compare decoded output and BLEU/ROUGE instead.
