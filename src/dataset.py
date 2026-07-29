import json

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


class Seq2SeqDataset(Dataset):
    """
    Loads the processed JSONL dataset for PyTorch.
    """

    def __init__(self, jsonl_path):
        self.examples = []

        with open(jsonl_path, "r", encoding="utf-8") as file:
            for line in file:
                example = json.loads(line)
                self.examples.append(example)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        example = self.examples[index]

        return {
            "source_ids": torch.tensor(
                example["source_ids"],
                dtype=torch.long,
            ),
            "target_ids": torch.tensor(
                example["target_ids"],
                dtype=torch.long,
            ),
            "source_text": example["source"],
            "target_text": example["target"],
        }


def create_collate_function(
    source_pad_id,
    target_pad_id,
):
    """
    Create a function that pads sequences inside each batch.

    Mask values:
    True = real token
    False = padding token
    """

    def collate_function(batch):
        source_sequences = [
            example["source_ids"]
            for example in batch
        ]

        target_sequences = [
            example["target_ids"]
            for example in batch
        ]

        source_ids = pad_sequence(
            source_sequences,
            batch_first=True,
            padding_value=source_pad_id,
        )

        target_ids = pad_sequence(
            target_sequences,
            batch_first=True,
            padding_value=target_pad_id,
        )

        source_mask = source_ids != source_pad_id
        target_mask = target_ids != target_pad_id

        source_lengths = source_mask.sum(dim=1)
        target_lengths = target_mask.sum(dim=1)

        return {
            "source_ids": source_ids,
            "target_ids": target_ids,
            "source_mask": source_mask,
            "target_mask": target_mask,
            "source_lengths": source_lengths,
            "target_lengths": target_lengths,
            "source_text": [
                example["source_text"]
                for example in batch
            ],
            "target_text": [
                example["target_text"]
                for example in batch
            ],
        }

    return collate_function