import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))

from dataset import Seq2SeqDataset, create_collate_function
from vocabulary import Vocabulary


def main():
    source_vocab = Vocabulary.load(
        "data/processed/source_vocab.json"
    )

    target_vocab = Vocabulary.load(
        "data/processed/target_vocab.json"
    )

    train_dataset = Seq2SeqDataset(
        "data/processed/train.jsonl"
    )

    collate_function = create_collate_function(
        source_pad_id=source_vocab.pad_id,
        target_pad_id=target_vocab.pad_id,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_function,
    )

    batch = next(iter(train_loader))

    print("Number of training examples:", len(train_dataset))
    print("Source batch shape:", batch["source_ids"].shape)
    print("Target batch shape:", batch["target_ids"].shape)
    print("Source mask shape:", batch["source_mask"].shape)
    print("Target mask shape:", batch["target_mask"].shape)
    print("Source lengths:", batch["source_lengths"])
    print("Target lengths:", batch["target_lengths"])
    print("First source text:", batch["source_text"][0])
    print("First target text:", batch["target_text"][0])


if __name__ == "__main__":
    main()