# Ablation A — `<unk>` left in the training loss

Kept as evidence for the report. **This is not the shipped model.**

Reproduce with:

```bash
python train.py --data-dir data/processed --results-dir results \
  --embedding-dim 128 --hidden-dim 256 --num-layers 1 --dropout 0.3 \
  --batch-size 32 --epochs 30 --learning-rate 0.001 \
  --teacher-forcing-ratio 1.0 --teacher-forcing-decay 0.02 \
  --clip-grad 1.0 --patience 5 --seed 42 --unk-loss-weight 1.0
```

## What happened

10 epochs, early-stopped, best validation loss 5.1689 at epoch 5, 24.4 min on
CPU. Training loss fell steadily (6.1393 → 3.7914) while validation loss never
improved after epoch 5.

The decoder converged to emitting `<unk>` at nearly every position. Over the
first 40 test articles it produced 310 `<unk>`, 40 `<eos>` and 19 real word
tokens total; 34 of 40 predictions are empty once special tokens are stripped.

## Why

`<unk>` accounts for 13.91% of all training headline tokens — roughly 4× the
most frequent real target token (`'` at 3.3%, `to` at 2.9%). Constant `<unk>` is
therefore the lowest-cross-entropy constant policy available, and with only
2,152 training examples the model settles there.

The fix is `--unk-loss-weight 0.0`, which zero-weights `<unk>` in the loss. See
`results/ablation_freerunning_val_loss/` for what happened next.
