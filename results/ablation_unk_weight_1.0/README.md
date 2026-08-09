# Ablation A - `<unk>` left in the training loss

Kept as evidence for the report. Not the shipped model.

10 epochs, best validation loss 5.1689 at epoch 5, 24.4 min on CPU. The decoder
learned to emit `<unk>` almost everywhere: over the first 40 test articles it
produced 310 `<unk>` and only 19 real words, leaving 34 of 40 predictions empty
once special tokens are stripped.

`<unk>` is 13.91% of training headline tokens, about 4x the most frequent real
word, so always predicting it was the cheapest option the model had.

Fixed with `--unk-loss-weight 0.0`. Reproduce this run with `1.0`:

```bash
python train.py --data-dir data/processed --results-dir results \
  --embedding-dim 128 --hidden-dim 256 --num-layers 1 --dropout 0.3 \
  --batch-size 32 --epochs 30 --learning-rate 0.001 \
  --teacher-forcing-ratio 1.0 --teacher-forcing-decay 0.02 \
  --clip-grad 1.0 --patience 5 --seed 42 --unk-loss-weight 1.0
```
