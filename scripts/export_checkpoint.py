#!/usr/bin/env python3
"""
Export a slim, inference-only copy of a training checkpoint (Role 2 -> Role 4).

train.py saves the Adam optimizer state alongside the weights so a run can be
resumed. That state is roughly two extra copies of every parameter and accounts
for about two thirds of the file — evaluate.py never reads it
(evaluate.py:load_model only touches "model_state_dict" and "config").

Stripping it produces a file small enough to share with the rest of the team
without the full training checkpoint. The output is a drop-in replacement for
--checkpoint in evaluate.py.

Usage:
    python scripts/export_checkpoint.py
    python scripts/export_checkpoint.py --input results/best_model.pt \
        --output results/best_model_inference.pt
"""

import argparse
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]

# Keys evaluate.py needs, plus provenance so the file is self-describing.
KEEP_KEYS = ("model_state_dict", "config", "epoch", "val_loss")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Strip optimizer state from a checkpoint for handoff"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=REPO_ROOT / "results" / "best_model.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results" / "best_model_inference.pt",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"No checkpoint at {args.input}. Run train.py first."
        )

    checkpoint = torch.load(
        args.input, map_location="cpu", weights_only=False
    )

    missing = [k for k in ("model_state_dict", "config") if k not in checkpoint]
    if missing:
        raise SystemExit(
            f"{args.input} is missing required keys: {missing}. "
            "It may not be a train.py checkpoint."
        )

    slim = {k: checkpoint[k] for k in KEEP_KEYS if k in checkpoint}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, args.output)

    before = args.input.stat().st_size / 1024 ** 2
    after = args.output.stat().st_size / 1024 ** 2

    print(f"Input:  {args.input}  ({before:.1f} MiB)")
    print(f"Output: {args.output}  ({after:.1f} MiB)")
    print(f"Dropped: {sorted(set(checkpoint) - set(slim))}")
    print(f"Epoch {slim.get('epoch')}, val_loss {slim.get('val_loss'):.4f}")
    print(
        "\nEvaluate with:\n"
        f"  python evaluate.py --checkpoint {args.output.relative_to(REPO_ROOT)} "
        "--data-dir data/processed --results-dir results"
    )


if __name__ == "__main__":
    main()
