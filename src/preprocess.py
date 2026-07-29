import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tokenizer import tokenize
from utils import set_seed
from vocabulary import build_vocabulary


# Reproducibility
SEED = 42

# File locations
RAW_DATA_PATH = "data/raw/dataset.csv"
OUTPUT_DIRECTORY = Path("data/processed")

# Your dataset must use these two column names
SOURCE_COLUMN = "source"
TARGET_COLUMN = "target"

# Dataset split
TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10

# Vocabulary settings
MIN_WORD_FREQUENCY = 2
MAX_VOCABULARY_SIZE = 30000

# Length filtering
MAX_SOURCE_LENGTH = 120
MAX_TARGET_LENGTH = 60


def load_and_clean_data():
    """
    Load the raw CSV dataset and clean invalid examples.
    """

    data = pd.read_csv(RAW_DATA_PATH)

    required_columns = {
        SOURCE_COLUMN,
        TARGET_COLUMN,
    }

    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing these columns: {missing_columns}. "
            f"The CSV must contain '{SOURCE_COLUMN}' and "
            f"'{TARGET_COLUMN}' columns."
        )

    # Keep only the two columns needed for seq2seq
    data = data[
        [
            SOURCE_COLUMN,
            TARGET_COLUMN,
        ]
    ].copy()

    # Remove rows with missing values
    data = data.dropna()

    # Convert values to strings and remove extra spaces
    data[SOURCE_COLUMN] = (
        data[SOURCE_COLUMN]
        .astype(str)
        .str.strip()
    )

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype(str)
        .str.strip()
    )

    # Remove empty rows
    data = data[
        (data[SOURCE_COLUMN] != "")
        & (data[TARGET_COLUMN] != "")
    ]

    # Remove duplicate source-target examples
    data = data.drop_duplicates(
        subset=[
            SOURCE_COLUMN,
            TARGET_COLUMN,
        ]
    )

    # Tokenize source and target text
    data["source_tokens"] = (
        data[SOURCE_COLUMN]
        .apply(tokenize)
    )

    data["target_tokens"] = (
        data[TARGET_COLUMN]
        .apply(tokenize)
    )

    # Remove examples that are too short or too long
    data = data[
        data["source_tokens"]
        .apply(len)
        .between(
            1,
            MAX_SOURCE_LENGTH,
        )
    ]

    data = data[
        data["target_tokens"]
        .apply(len)
        .between(
            1,
            MAX_TARGET_LENGTH,
        )
    ]

    return data.reset_index(drop=True)


def split_data(data):
    """
    Split examples into train, validation, and test sets.

    This happens before vocabulary construction to prevent
    validation and test data leakage.
    """

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(total_ratio - 1.0) > 0.000001:
        raise ValueError(
            "Train, validation, and test ratios must add to 1.0."
        )

    train_data, temporary_data = train_test_split(
        data,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=SEED,
        shuffle=True,
    )

    relative_test_ratio = (
        TEST_RATIO
        / (VALIDATION_RATIO + TEST_RATIO)
    )

    validation_data, test_data = train_test_split(
        temporary_data,
        test_size=relative_test_ratio,
        random_state=SEED,
        shuffle=True,
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def save_processed_split(
    data,
    output_path,
    source_vocabulary,
    target_vocabulary,
):
    """
    Save one processed dataset split as a JSONL file.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for _, row in data.iterrows():
            example = {
                "source": row[SOURCE_COLUMN],
                "target": row[TARGET_COLUMN],
                "source_tokens": row["source_tokens"],
                "target_tokens": row["target_tokens"],
                "source_ids": source_vocabulary.encode(
                    row["source_tokens"]
                ),
                "target_ids": target_vocabulary.encode(
                    row["target_tokens"]
                ),
            }

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


def main():
    """
    Run the complete preprocessing pipeline.
    """

    set_seed(SEED)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Loading and cleaning dataset...")

    data = load_and_clean_data()

    print(
        f"Clean examples found: {len(data)}"
    )

    print(
        "Creating train, validation, and test splits..."
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_data(data)

    print(
        "Building source and target vocabularies "
        "from training data only..."
    )

    source_vocabulary = build_vocabulary(
        train_data["source_tokens"].tolist(),
        min_frequency=MIN_WORD_FREQUENCY,
        max_size=MAX_VOCABULARY_SIZE,
    )

    target_vocabulary = build_vocabulary(
        train_data["target_tokens"].tolist(),
        min_frequency=MIN_WORD_FREQUENCY,
        max_size=MAX_VOCABULARY_SIZE,
    )

    source_vocabulary.save(
        OUTPUT_DIRECTORY
        / "source_vocab.json"
    )

    target_vocabulary.save(
        OUTPUT_DIRECTORY
        / "target_vocab.json"
    )

    print("Saving processed dataset splits...")

    save_processed_split(
        train_data,
        OUTPUT_DIRECTORY / "train.jsonl",
        source_vocabulary,
        target_vocabulary,
    )

    save_processed_split(
        validation_data,
        OUTPUT_DIRECTORY / "validation.jsonl",
        source_vocabulary,
        target_vocabulary,
    )

    save_processed_split(
        test_data,
        OUTPUT_DIRECTORY / "test.jsonl",
        source_vocabulary,
        target_vocabulary,
    )

    metadata = {
        "seed": SEED,
        "total_clean_examples": len(data),
        "train_examples": len(train_data),
        "validation_examples": len(validation_data),
        "test_examples": len(test_data),
        "source_vocabulary_size": len(
            source_vocabulary
        ),
        "target_vocabulary_size": len(
            target_vocabulary
        ),
        "vocabulary_built_from": (
            "training split only"
        ),
        "train_ratio": TRAIN_RATIO,
        "validation_ratio": VALIDATION_RATIO,
        "test_ratio": TEST_RATIO,
        "maximum_source_length": (
            MAX_SOURCE_LENGTH
        ),
        "maximum_target_length": (
            MAX_TARGET_LENGTH
        ),
    }

    with open(
        OUTPUT_DIRECTORY / "metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
        )

    print("Preprocessing completed successfully.")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()