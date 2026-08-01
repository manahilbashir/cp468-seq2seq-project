"""
Training script for LSTM Seq2Seq with Attention.

Implements training from scratch with:
  - Teacher forcing (with optional ratio decay)
  - Gradient clipping
  - Validation-based checkpointing
  - Early stopping
  - Matplotlib training curve logging
  - Hyperparameter JSON logging

No high-level trainer wrappers used.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataset import Seq2SeqDataset, create_collate_function
from model import Encoder, Decoder, Seq2Seq
from utils import set_seed
from vocabulary import Vocabulary
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import Seq2SeqDataset, create_collate_function
from model import Encoder, Decoder, Seq2Seq
from utils import set_seed
from vocabulary import Vocabulary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train LSTM Seq2Seq with Attention"
    )

    # Data paths
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory containing processed data files",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help="Directory to save results (plots, checkpoints, logs)",
    )

    # Model architecture
    parser.add_argument(
        "--embedding-dim",
        type=int,
        default=256,
        help="Token embedding dimension",
    )
    parser.add_argument(
        "--hidden-dim",
        type=int,
        default=512,
        help="LSTM hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=2,
        help="Number of LSTM layers in encoder and decoder",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.3,
        help="Dropout probability",
    )

    # Training hyperparameters
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate",
    )
    parser.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        default=1.0,
        help="Initial teacher forcing ratio",
    )
    parser.add_argument(
        "--teacher-forcing-decay",
        type=float,
        default=0.0,
        help="Decay teacher forcing ratio by this amount each epoch (set 0 to disable)",
    )
    parser.add_argument(
        "--clip-grad",
        type=float,
        default=1.0,
        help="Gradient clipping max norm",
    )

    # Training utilities
    parser.add_argument(
        "--patience",
        type=int,
        default=7,
        help="Early stopping patience (epochs without improvement)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    return parser.parse_args()


def save_training_curves(history, save_path):
    """
    Plot and save training/validation loss and perplexity curves.

    Args:
        history: dict with keys 'train_loss', 'val_loss', 'train_ppl', 'val_ppl'
        save_path: path to save the figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss plot
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-Entropy Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Perplexity plot
    axes[1].plot(epochs, history["train_ppl"], "b-", label="Train PPL")
    axes[1].plot(epochs, history["val_ppl"], "r-", label="Val PPL")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Training & Validation Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved to {save_path}")


def calculate_perplexity(loss):
    """Convert cross-entropy loss to perplexity."""
    return torch.exp(torch.tensor(loss)).item()


def train_epoch(model, dataloader, criterion, optimizer, device, teacher_forcing_ratio, clip_grad):
    """
    Run one training epoch.

    Returns:
        average_loss: Mean loss across all batches.
    """
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        source_ids = batch["source_ids"].to(device)
        target_ids = batch["target_ids"].to(device)
        source_lengths = batch["source_lengths"].to(device)

        # target_ids includes <bos> and <eos>
        # Model predicts tokens after <bos>, so input is target_ids[:, :-1]
        # But our model expects full target_ids and handles slicing internally
        # Actually in our model.forward: outputs shape is (batch, tgt_seq_len - 1, vocab)
        # target for loss should be target_ids[:, 1:] (skip <bos>)

        optimizer.zero_grad()

        outputs = model(
            source_ids,
            source_lengths,
            target_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # outputs: (batch_size, tgt_seq_len - 1, vocab_size)
        # targets: (batch_size, tgt_seq_len - 1)
        targets = target_ids[:, 1:]

        # Reshape for CrossEntropyLoss
        # (batch_size * seq_len, vocab_size)
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.reshape(-1)

        # Ignore padding in loss computation
        loss = criterion(outputs, targets)

        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches


def validate_epoch(model, dataloader, criterion, device):
    """
    Run one validation epoch (no teacher forcing, no gradient updates).

    Returns:
        average_loss: Mean loss across all batches.
    """
    model.eval()
    epoch_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in dataloader:
            source_ids = batch["source_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            source_lengths = batch["source_lengths"].to(device)

            # No teacher forcing during validation
            outputs = model(
                source_ids,
                source_lengths,
                target_ids,
                teacher_forcing_ratio=0.0,
            )

            targets = target_ids[:, 1:]
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.reshape(-1)

            loss = criterion(outputs, targets)

            epoch_loss += loss.item()
            num_batches += 1

    return epoch_loss / num_batches


def main():
    args = parse_arguments()

    # Reproducibility
    set_seed(args.seed)

    # Device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Paths
    data_dir = Path(args.data_dir)
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = results_dir / "best_model.pt"
    config_path = results_dir / "training_config.json"
    curves_path = results_dir / "training_curves.png"
    history_path = results_dir / "training_history.json"

    # Load vocabularies
    print("Loading vocabularies...")
    source_vocab = Vocabulary.load(data_dir / "source_vocab.json")
    target_vocab = Vocabulary.load(data_dir / "target_vocab.json")
    print(f"Source vocab size: {len(source_vocab)}")
    print(f"Target vocab size: {len(target_vocab)}")

    # Load datasets
    print("Loading datasets...")
    train_dataset = Seq2SeqDataset(data_dir / "train.jsonl")
    val_dataset = Seq2SeqDataset(data_dir / "validation.jsonl")
    print(f"Train examples: {len(train_dataset)}")
    print(f"Validation examples: {len(val_dataset)}")

    # Collate function
    collate_fn = create_collate_function(
        source_pad_id=source_vocab.pad_id,
        target_pad_id=target_vocab.pad_id,
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    # Build model
    print("Building model...")
    encoder = Encoder(
        vocab_size=len(source_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        padding_idx=source_vocab.pad_id,
    )

    decoder = Decoder(
        vocab_size=len(target_vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        padding_idx=target_vocab.pad_id,
    )

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        source_pad_id=source_vocab.pad_id,
        target_bos_id=target_vocab.bos_id,
        target_eos_id=target_vocab.eos_id,
        device=device,
    ).to(device)

    num_params = model.count_parameters()
    print(f"Model parameters: {num_params:,}")

    # Loss function: ignore padding index
    criterion = nn.CrossEntropyLoss(ignore_index=target_vocab.pad_id)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Save training config for reproducibility
    config = vars(args)
    config["source_vocab_size"] = len(source_vocab)
    config["target_vocab_size"] = len(target_vocab)
    config["model_parameters"] = num_params
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    print(f"Training config saved to {config_path}")

    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "train_ppl": [],
        "val_ppl": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    teacher_forcing_ratio = args.teacher_forcing_ratio

    print("\n" + "=" * 60)
    print("Starting training...")
    print("=" * 60 + "\n")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            teacher_forcing_ratio,
            args.clip_grad,
        )

        val_loss = validate_epoch(
            model, val_loader, criterion, device
        )

        train_ppl = calculate_perplexity(train_loss)
        val_ppl = calculate_perplexity(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_ppl"].append(train_ppl)
        history["val_ppl"].append(val_ppl)

        epoch_time = time.time() - epoch_start

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} (PPL: {train_ppl:.2f}) | "
            f"Val Loss: {val_loss:.4f} (PPL: {val_ppl:.2f}) | "
            f"TF Ratio: {teacher_forcing_ratio:.2f} | "
            f"Time: {epoch_time:.1f}s"
        )

        # Checkpoint best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": config,
                },
                checkpoint_path,
            )
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= args.patience:
            print(
                f"\nEarly stopping triggered after {epoch} epochs "
                f"(no improvement for {args.patience} epochs)."
            )
            break

        # Decay teacher forcing ratio
        if args.teacher_forcing_decay > 0:
            teacher_forcing_ratio = max(
                0.0, teacher_forcing_ratio - args.teacher_forcing_decay
            )

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print(f"Training completed in {total_time:.1f}s ({total_time / 60:.1f}m)")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print("=" * 60)

    # Save training history
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Training history saved to {history_path}")

    # Plot training curves
    save_training_curves(history, curves_path)

    # Print summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model parameters:     {num_params:,}")
    print(f"Device:               {device}")
    print(f"Total epochs trained: {epoch}")
    print(f"Total training time:  {total_time:.1f}s")
    print(f"Best val loss:        {best_val_loss:.4f}")
    print(f"Best val perplexity:  {calculate_perplexity(best_val_loss):.2f}")
    print(f"Checkpoint saved to:  {checkpoint_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
