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
    parser.add_argument(
        "--unk-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Loss weight for the <unk> target token. <unk> is ~14%% of "
            "training headline tokens -- roughly 4x the most frequent real "
            "token -- so at weight 1.0 the decoder collapses onto emitting "
            "<unk> at every position. 0.0 removes it from the loss; use 1.0 "
            "to reproduce that collapse."
        ),
    )

    parser.add_argument(
        "--no-attention",
        action="store_true",
        help=(
            "Ablation: remove Bahdanau attention. The decoder then sees the "
            "encoder's final bidirectional state as a constant context at "
            "every timestep, so the whole article passes through one "
            "fixed-width vector. Everything else is held identical."
        ),
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

        optimizer.zero_grad()

        outputs = model(
            source_ids,
            source_lengths,
            target_ids,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        # outputs: (batch, tgt_len - 1, vocab); targets skip <bos>
        targets = target_ids[:, 1:]
        outputs = outputs.view(-1, outputs.size(-1))
        targets = targets.reshape(-1)

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
    Run one validation epoch (teacher forced, no gradient updates).

    Validation is teacher forced so that it measures the same quantity as the
    training loss and is therefore usable for model selection and early
    stopping.

    An earlier version ran this free-running (teacher_forcing_ratio=0.0). That
    measures something else: how well the model continues from its own
    predictions. Because exposure bias compounds as the decoder grows confident
    but not yet accurate, that loss rose monotonically from epoch 2 onward even
    while training loss fell -- early stopping fired at epoch 6 and selected the
    epoch-1 checkpoint, which emits one constant headline for every article.
    Preserved for comparison in results/ablation_freerunning_val_loss/.

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

            outputs = model(
                source_ids,
                source_lengths,
                target_ids,
                teacher_forcing_ratio=1.0,
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
        use_attention=not args.no_attention,
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
    print(
        f"Attention: {'disabled (ablation)' if args.no_attention else 'Bahdanau'}"
    )

    # Loss function: ignore padding, and down-weight <unk> as a target.
    #
    # ignore_index only takes one token, so <unk> is handled with a class
    # weight vector instead. With reduction="mean" the loss is divided by the
    # summed weights of the targets, so a zero-weighted class drops out of both
    # the numerator and the denominator -- equivalent to ignoring it.
    #
    # This matters: <unk> is 13.91% of training headline tokens, about 4x the
    # most frequent real token, so at weight 1.0 predicting <unk> at every
    # position is the lowest-loss constant policy and the decoder converges
    # there -- a run at weight 1.0 emitted 310 <unk> against 19 real word
    # tokens over 40 test articles.
    loss_weight = torch.ones(len(target_vocab), device=device)
    loss_weight[target_vocab.unk_id] = args.unk_loss_weight

    criterion = nn.CrossEntropyLoss(
        weight=loss_weight,
        ignore_index=target_vocab.pad_id,
    )
    print(
        f"Loss: CrossEntropy(ignore_index=<pad>, "
        f"<unk> weight={args.unk_loss_weight})"
    )

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
