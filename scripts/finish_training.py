#!/usr/bin/env python3
"""
Finish Role 2 training + light hyperparameter tuning.

Run this AFTER Role 1 replaces the placeholder dataset and (re)builds
data/processed/. It:

  1. Sanity-checks that processed data looks real (not the 10-example toy set)
  2. Optionally re-runs preprocessing
  3. Trains a small grid of configs via train.py
  4. Picks the best validation loss run
  5. Copies that checkpoint/curves into results/ for Role 4

Example:
  python scripts/finish_training.py
  python scripts/finish_training.py --preprocess --device cuda
  python scripts/finish_training.py --quick
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
TUNING_ROOT = DEFAULT_RESULTS_DIR / "tuning"


FULL_GRID = [
    {
        "name": "baseline",
        "embedding_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "teacher_forcing_decay": 0.02,
    },
    {
        "name": "smaller_lr",
        "embedding_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.0005,
        "teacher_forcing_decay": 0.02,
    },
    {
        "name": "more_dropout",
        "embedding_dim": 256,
        "hidden_dim": 512,
        "num_layers": 2,
        "dropout": 0.5,
        "learning_rate": 0.001,
        "teacher_forcing_decay": 0.02,
    },
    {
        "name": "narrower",
        "embedding_dim": 128,
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.3,
        "learning_rate": 0.001,
        "teacher_forcing_decay": 0.02,
    },
]

QUICK_GRID = [
    FULL_GRID[0],
    FULL_GRID[1],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the LSTM across a small hyperparameter grid"
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument(
        "--preprocess",
        action="store_true",
        help="Run src/preprocess.py before training",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Smaller grid and fewer epochs",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--min-train-examples", type=int, default=100)
    parser.add_argument("--min-vocab-size", type=int, default=100)
    parser.add_argument(
        "--allow-tiny-data",
        action="store_true",
        help="Skip placeholder guards (debug only)",
    )
    return parser.parse_args()


def load_metadata(data_dir: Path) -> dict:
    path = data_dir / "metadata.json"
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run: python src/preprocess.py"
        )
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def assert_data_ready(metadata: dict, args) -> None:
    if args.allow_tiny_data:
        print("WARNING: --allow-tiny-data set; skipping placeholder checks.")
        return

    train_n = int(metadata.get("train_examples", 0))
    src_v = int(metadata.get("source_vocabulary_size", 0))
    tgt_v = int(metadata.get("target_vocabulary_size", 0))

    problems = []
    if train_n < args.min_train_examples:
        problems.append(
            f"train_examples={train_n} < {args.min_train_examples}"
        )
    if src_v < args.min_vocab_size:
        problems.append(
            f"source_vocabulary_size={src_v} < {args.min_vocab_size}"
        )
    if tgt_v < args.min_vocab_size:
        problems.append(
            f"target_vocabulary_size={tgt_v} < {args.min_vocab_size}"
        )

    if problems:
        raise SystemExit(
            "Processed data still looks like the placeholder set:\n  - "
            + "\n  - ".join(problems)
            + "\n\nRole 1 must replace data/raw/dataset.csv with a real "
            "article->headline corpus, then run:\n"
            "  python src/preprocess.py\n"
            "Re-run this script after that. "
            "(Use --allow-tiny-data only for debugging.)"
        )

    print(
        f"Data looks usable: train={train_n}, "
        f"src_vocab={src_v}, tgt_vocab={tgt_v}"
    )


def run_preprocess() -> None:
    cmd = [sys.executable, str(REPO_ROOT / "src" / "preprocess.py")]
    print("\n>>", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)


def best_val_loss(run_dir: Path) -> float:
    history_path = run_dir / "training_history.json"
    checkpoint_path = run_dir / "best_model.pt"

    if history_path.exists():
        with open(history_path, encoding="utf-8") as f:
            history = json.load(f)
        val_losses = history.get("val_loss") or []
        if val_losses:
            return float(min(val_losses))

    if checkpoint_path.exists():
        import torch

        ckpt = torch.load(checkpoint_path, map_location="cpu")
        if "val_loss" in ckpt:
            return float(ckpt["val_loss"])

    return float("inf")


def train_one(config: dict, args, epochs: int) -> dict:
    run_dir = TUNING_ROOT / config["name"]
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        str(REPO_ROOT / "train.py"),
        "--data-dir",
        str(args.data_dir),
        "--results-dir",
        str(run_dir),
        "--embedding-dim",
        str(config["embedding_dim"]),
        "--hidden-dim",
        str(config["hidden_dim"]),
        "--num-layers",
        str(config["num_layers"]),
        "--dropout",
        str(config["dropout"]),
        "--batch-size",
        str(args.batch_size),
        "--epochs",
        str(epochs),
        "--learning-rate",
        str(config["learning_rate"]),
        "--teacher-forcing-ratio",
        "1.0",
        "--teacher-forcing-decay",
        str(config["teacher_forcing_decay"]),
        "--clip-grad",
        "1.0",
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
    ]
    if args.device:
        cmd.extend(["--device", args.device])

    print("\n" + "=" * 60)
    print(f"Tuning run: {config['name']}")
    print("=" * 60)
    print(">>", " ".join(cmd))
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)

    val_loss = best_val_loss(run_dir)
    result = {
        **config,
        "results_dir": str(run_dir),
        "best_val_loss": val_loss,
        "checkpoint": str(run_dir / "best_model.pt"),
    }
    print(f"Run {config['name']} best val_loss={val_loss:.4f}")
    return result


def promote_best(best: dict, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = Path(best["results_dir"])

    for name in (
        "best_model.pt",
        "training_curves.png",
        "training_history.json",
        "training_config.json",
    ):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, results_dir / name)
            print(f"Copied {name} -> {results_dir / name}")


def main():
    args = parse_args()
    grid = QUICK_GRID if args.quick else FULL_GRID
    epochs = args.epochs
    if epochs is None:
        epochs = 15 if args.quick else 50

    print("Role 2 finish script")
    print(f"Repo: {REPO_ROOT}")
    print(f"Grid: {[c['name'] for c in grid]} | epochs={epochs}")

    if args.preprocess:
        run_preprocess()

    metadata = load_metadata(args.data_dir)
    assert_data_ready(metadata, args)

    TUNING_ROOT.mkdir(parents=True, exist_ok=True)
    results = [train_one(config, args, epochs) for config in grid]

    results_sorted = sorted(results, key=lambda r: r["best_val_loss"])
    best = results_sorted[0]

    summary = {
        "selected": best["name"],
        "best_val_loss": best["best_val_loss"],
        "runs": results_sorted,
        "epochs": epochs,
        "quick": args.quick,
        "data_metadata": metadata,
    }
    args.results_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.results_dir / "tuning_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print("TUNING SUMMARY")
    print("=" * 60)
    for row in results_sorted:
        mark = " <-- BEST" if row["name"] == best["name"] else ""
        print(
            f"  {row['name']:16s}  val_loss={row['best_val_loss']:.4f}{mark}"
        )
    print(f"\nSummary written to {summary_path}")

    promote_best(best, args.results_dir)
    print(
        "\nDone. Role 4 can evaluate with:\n"
        "  python evaluate.py "
        "--checkpoint results/best_model.pt "
        "--data-dir data/processed "
        "--results-dir results"
    )


if __name__ == "__main__":
    main()
