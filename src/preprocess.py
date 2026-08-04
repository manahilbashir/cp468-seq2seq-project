import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from tokenizer import tokenize
from utils import set_seed
from vocabulary import build_vocabulary


# ============================================================
# Reproducibility
# ============================================================

SEED = 42


# ============================================================
# File paths
# ============================================================

RAW_DATA_PATH = "data/raw/dataset.csv"
OUTPUT_DIRECTORY = Path("data/processed")


# ============================================================
# Standardized column names
# ============================================================

SOURCE_COLUMN = "source"
TARGET_COLUMN = "target"


# ============================================================
# Dataset split
# ============================================================

TRAIN_RATIO = 0.80
VALIDATION_RATIO = 0.10
TEST_RATIO = 0.10


# ============================================================
# Vocabulary settings
# ============================================================

MIN_WORD_FREQUENCY = 2
MAX_VOCABULARY_SIZE = 30000


# ============================================================
# Article / headline length settings
# ============================================================

# Article input
MIN_SOURCE_LENGTH = 20
MAX_SOURCE_LENGTH = 400

# Headline output
MIN_TARGET_LENGTH = 2
MAX_TARGET_LENGTH = 30


# ============================================================
# Read CSV safely
# ============================================================

def read_dataset():
    """
    Read the raw CSV.

    Some versions of the Kaggle News Summary dataset are not
    encoded as UTF-8, so this tries multiple encodings.
    """

    print(f"Reading dataset from: {RAW_DATA_PATH}")

    encodings = [
        "utf-8",
        "latin-1",
        "cp1252",
    ]

    last_error = None

    for encoding in encodings:
        try:
            print(f"Trying encoding: {encoding}")

            data = pd.read_csv(
                RAW_DATA_PATH,
                encoding=encoding,
                low_memory=False,
            )

            print(
                f"Successfully loaded dataset "
                f"using {encoding} encoding."
            )

            return data

        except UnicodeDecodeError as error:
            last_error = error

            print(
                f"{encoding} failed. "
                "Trying another encoding..."
            )

    raise last_error


# ============================================================
# Detect dataset format
# ============================================================

def standardize_columns(data):
    """
    Convert different news dataset formats into:

        source = article
        target = headline
    """

    print("\nColumns found in dataset:")

    for column in data.columns:
        print(f" - {column}")

    columns = set(data.columns)

    # --------------------------------------------------------
    # Already standardized
    # --------------------------------------------------------

    if {
        "source",
        "target",
    }.issubset(columns):

        print(
            "\nDetected source/target format."
        )

        standardized = data[
            [
                "source",
                "target",
            ]
        ].copy()

        return standardized


    # --------------------------------------------------------
    # Kaggle News Summary
    #
    # Full article:
    # ctext
    #
    # Headline:
    # headlines
    # --------------------------------------------------------

    if {
        "ctext",
        "headlines",
    }.issubset(columns):

        print(
            "\nDetected Kaggle News Summary "
            "(ctext -> headlines)."
        )

        standardized = pd.DataFrame()

        standardized["source"] = (
            data["ctext"]
        )

        standardized["target"] = (
            data["headlines"]
        )

        return standardized


    # --------------------------------------------------------
    # Kaggle news_summary_more.csv
    #
    # Short article text:
    # text
    #
    # Headline:
    # headlines
    # --------------------------------------------------------

    if {
        "text",
        "headlines",
    }.issubset(columns):

        print(
            "\nDetected Kaggle News Summary "
            "(text -> headlines)."
        )

        standardized = pd.DataFrame()

        standardized["source"] = (
            data["text"]
        )

        standardized["target"] = (
            data["headlines"]
        )

        return standardized


    # --------------------------------------------------------
    # CNN / DailyMail
    # --------------------------------------------------------

    if {
        "article",
        "highlights",
    }.issubset(columns):

        print(
            "\nDetected CNN/DailyMail "
            "(article -> highlights)."
        )

        standardized = pd.DataFrame()

        standardized["source"] = (
            data["article"]
        )

        standardized["target"] = (
            data["highlights"]
        )

        return standardized


    # --------------------------------------------------------
    # Article / headline generic format
    # --------------------------------------------------------

    if {
        "article",
        "headline",
    }.issubset(columns):

        print(
            "\nDetected article/headline format."
        )

        standardized = pd.DataFrame()

        standardized["source"] = (
            data["article"]
        )

        standardized["target"] = (
            data["headline"]
        )

        return standardized


    raise ValueError(
        "\nUnsupported dataset format.\n\n"
        f"Columns found:\n{list(data.columns)}\n\n"
        "Expected one of these formats:\n"
        "source,target\n"
        "ctext,headlines\n"
        "text,headlines\n"
        "article,headline\n"
        "article,highlights"
    )


# ============================================================
# Clean dataset
# ============================================================

def load_and_clean_data():
    """
    Load the dataset, clean text, remove invalid rows,
    tokenize and filter by sequence length.
    """

    raw_data = read_dataset()

    print(
        f"\nRaw dataset rows: "
        f"{len(raw_data):,}"
    )

    data = standardize_columns(
        raw_data
    )

    # --------------------------------------------------------
    # Remove missing article/headline values
    # --------------------------------------------------------

    before = len(data)

    data = data.dropna(
        subset=[
            SOURCE_COLUMN,
            TARGET_COLUMN,
        ]
    )

    print(
        f"Removed missing rows: "
        f"{before - len(data):,}"
    )


    # --------------------------------------------------------
    # Convert values to strings
    # --------------------------------------------------------

    data[SOURCE_COLUMN] = (
        data[SOURCE_COLUMN]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )

    data[TARGET_COLUMN] = (
        data[TARGET_COLUMN]
        .astype(str)
        .str.replace(
            r"\s+",
            " ",
            regex=True,
        )
        .str.strip()
    )


    # --------------------------------------------------------
    # Remove empty rows
    # --------------------------------------------------------

    before = len(data)

    data = data[
        (
            data[SOURCE_COLUMN] != ""
        )
        &
        (
            data[TARGET_COLUMN] != ""
        )
    ]

    print(
        f"Removed empty rows: "
        f"{before - len(data):,}"
    )


    # --------------------------------------------------------
    # Remove duplicate articles
    #
    # This helps prevent identical articles from appearing
    # in train, validation and test.
    # --------------------------------------------------------

    before = len(data)

    data = data.drop_duplicates(
        subset=[
            SOURCE_COLUMN
        ]
    )

    print(
        f"Removed duplicate articles: "
        f"{before - len(data):,}"
    )


    # --------------------------------------------------------
    # Tokenize
    # --------------------------------------------------------

    print("\nTokenizing articles...")

    data["source_tokens"] = (
        data[SOURCE_COLUMN]
        .apply(tokenize)
    )

    print("Tokenizing headlines...")

    data["target_tokens"] = (
        data[TARGET_COLUMN]
        .apply(tokenize)
    )


    # --------------------------------------------------------
    # Calculate lengths
    # --------------------------------------------------------

    data["source_length"] = (
        data["source_tokens"]
        .apply(len)
    )

    data["target_length"] = (
        data["target_tokens"]
        .apply(len)
    )


    # --------------------------------------------------------
    # Filter articles
    # --------------------------------------------------------

    before = len(data)

    data = data[
        data["source_length"]
        .between(
            MIN_SOURCE_LENGTH,
            MAX_SOURCE_LENGTH,
        )
    ]

    print(
        f"Removed articles outside "
        f"{MIN_SOURCE_LENGTH}-"
        f"{MAX_SOURCE_LENGTH} tokens: "
        f"{before - len(data):,}"
    )


    # --------------------------------------------------------
    # Filter headlines
    # --------------------------------------------------------

    before = len(data)

    data = data[
        data["target_length"]
        .between(
            MIN_TARGET_LENGTH,
            MAX_TARGET_LENGTH,
        )
    ]

    print(
        f"Removed headlines outside "
        f"{MIN_TARGET_LENGTH}-"
        f"{MAX_TARGET_LENGTH} tokens: "
        f"{before - len(data):,}"
    )


    data = data.reset_index(
        drop=True
    )

    if len(data) == 0:
        raise ValueError(
            "No usable examples remain after cleaning."
        )

    return data


# ============================================================
# Split dataset
# ============================================================

def split_data(data):
    """
    Split the dataset BEFORE building the vocabulary.

    This prevents validation/test vocabulary leakage.
    """

    total_ratio = (
        TRAIN_RATIO
        + VALIDATION_RATIO
        + TEST_RATIO
    )

    if abs(
        total_ratio - 1.0
    ) > 0.000001:

        raise ValueError(
            "Train, validation and test ratios "
            "must add to 1.0."
        )


    if len(data) < 100:
        print(
            "\nWARNING: fewer than 100 usable examples."
        )


    # --------------------------------------------------------
    # First split:
    #
    # 80% train
    # 20% temporary
    # --------------------------------------------------------

    train_data, temporary_data = (
        train_test_split(
            data,
            test_size=(
                VALIDATION_RATIO
                + TEST_RATIO
            ),
            random_state=SEED,
            shuffle=True,
        )
    )


    # --------------------------------------------------------
    # Split temporary set equally:
    #
    # 10% validation
    # 10% test
    # --------------------------------------------------------

    relative_test_ratio = (
        TEST_RATIO
        /
        (
            VALIDATION_RATIO
            + TEST_RATIO
        )
    )

    validation_data, test_data = (
        train_test_split(
            temporary_data,
            test_size=relative_test_ratio,
            random_state=SEED,
            shuffle=True,
        )
    )


    return (
        train_data.reset_index(
            drop=True
        ),
        validation_data.reset_index(
            drop=True
        ),
        test_data.reset_index(
            drop=True
        ),
    )


# ============================================================
# Save processed split
# ============================================================

def save_processed_split(
    data,
    output_path,
    source_vocabulary,
    target_vocabulary,
):
    """
    Convert text into token IDs and save as JSONL.
    """

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as file:

        for _, row in data.iterrows():

            source_tokens = (
                row["source_tokens"]
            )

            target_tokens = (
                row["target_tokens"]
            )

            example = {

                "source":
                    row[SOURCE_COLUMN],

                "target":
                    row[TARGET_COLUMN],

                "source_tokens":
                    source_tokens,

                "target_tokens":
                    target_tokens,

                "source_ids":
                    source_vocabulary.encode(
                        source_tokens
                    ),

                "target_ids":
                    target_vocabulary.encode(
                        target_tokens
                    ),
            }

            file.write(
                json.dumps(
                    example,
                    ensure_ascii=False,
                )
                + "\n"
            )


# ============================================================
# Main preprocessing pipeline
# ============================================================

def main():

    print(
        "=" * 60
    )

    print(
        "ARTICLE HEADLINE GENERATION PREPROCESSING"
    )

    print(
        "=" * 60
    )


    # --------------------------------------------------------
    # Reproducibility
    # --------------------------------------------------------

    set_seed(SEED)


    # --------------------------------------------------------
    # Create output directory
    # --------------------------------------------------------

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Load / clean
    # --------------------------------------------------------

    print(
        "\nLoading and cleaning dataset..."
    )

    data = load_and_clean_data()


    print(
        "\n"
        + "=" * 60
    )

    print(
        "CLEAN DATASET SUMMARY"
    )

    print(
        "=" * 60
    )


    print(
        f"Clean examples: "
        f"{len(data):,}"
    )

    print(
        f"Average article length: "
        f"{data['source_length'].mean():.2f} tokens"
    )

    print(
        f"Average headline length: "
        f"{data['target_length'].mean():.2f} tokens"
    )

    print(
        f"Longest retained article: "
        f"{data['source_length'].max()} tokens"
    )

    print(
        f"Longest retained headline: "
        f"{data['target_length'].max()} tokens"
    )


    # --------------------------------------------------------
    # Split
    # --------------------------------------------------------

    print(
        "\nCreating train / validation / test splits..."
    )

    (
        train_data,
        validation_data,
        test_data,
    ) = split_data(
        data
    )


    print(
        f"\nTrain examples: "
        f"{len(train_data):,}"
    )

    print(
        f"Validation examples: "
        f"{len(validation_data):,}"
    )

    print(
        f"Test examples: "
        f"{len(test_data):,}"
    )


    # --------------------------------------------------------
    # Vocabulary
    #
    # IMPORTANT:
    # TRAINING DATA ONLY
    # --------------------------------------------------------

    print(
        "\nBuilding source vocabulary "
        "from TRAINING DATA ONLY..."
    )

    source_vocabulary = (
        build_vocabulary(
            train_data[
                "source_tokens"
            ].tolist(),
            min_frequency=(
                MIN_WORD_FREQUENCY
            ),
            max_size=(
                MAX_VOCABULARY_SIZE
            ),
        )
    )


    print(
        "Building target vocabulary "
        "from TRAINING DATA ONLY..."
    )

    target_vocabulary = (
        build_vocabulary(
            train_data[
                "target_tokens"
            ].tolist(),
            min_frequency=(
                MIN_WORD_FREQUENCY
            ),
            max_size=(
                MAX_VOCABULARY_SIZE
            ),
        )
    )


    print(
        f"\nSource vocabulary size: "
        f"{len(source_vocabulary):,}"
    )

    print(
        f"Target vocabulary size: "
        f"{len(target_vocabulary):,}"
    )


    # --------------------------------------------------------
    # Save vocabulary
    # --------------------------------------------------------

    source_vocabulary.save(
        OUTPUT_DIRECTORY
        / "source_vocab.json"
    )

    target_vocabulary.save(
        OUTPUT_DIRECTORY
        / "target_vocab.json"
    )


    # --------------------------------------------------------
    # Save train
    # --------------------------------------------------------

    print(
        "\nSaving training data..."
    )

    save_processed_split(
        train_data,
        OUTPUT_DIRECTORY
        / "train.jsonl",
        source_vocabulary,
        target_vocabulary,
    )


    # --------------------------------------------------------
    # Save validation
    # --------------------------------------------------------

    print(
        "Saving validation data..."
    )

    save_processed_split(
        validation_data,
        OUTPUT_DIRECTORY
        / "validation.jsonl",
        source_vocabulary,
        target_vocabulary,
    )


    # --------------------------------------------------------
    # Save test
    # --------------------------------------------------------

    print(
        "Saving test data..."
    )

    save_processed_split(
        test_data,
        OUTPUT_DIRECTORY
        / "test.jsonl",
        source_vocabulary,
        target_vocabulary,
    )


    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "task":
            "article headline generation",

        "seed":
            SEED,

        "dataset":
            "Kaggle News Summary",

        "dataset_source":
            "https://www.kaggle.com/datasets/"
            "sunnysai12345/news-summary",

        "dataset_license":
            "GPL-2.0",

        "raw_dataset_path":
            RAW_DATA_PATH,

        "total_clean_examples":
            len(data),

        "train_examples":
            len(train_data),

        "validation_examples":
            len(validation_data),

        "test_examples":
            len(test_data),

        "train_ratio":
            TRAIN_RATIO,

        "validation_ratio":
            VALIDATION_RATIO,

        "test_ratio":
            TEST_RATIO,

        "source_vocabulary_size":
            len(source_vocabulary),

        "target_vocabulary_size":
            len(target_vocabulary),

        "vocabulary_built_from":
            "training split only",

        "minimum_source_length":
            MIN_SOURCE_LENGTH,

        "maximum_source_length":
            MAX_SOURCE_LENGTH,

        "minimum_target_length":
            MIN_TARGET_LENGTH,

        "maximum_target_length":
            MAX_TARGET_LENGTH,

        "average_source_length":
            float(
                data[
                    "source_length"
                ].mean()
            ),

        "average_target_length":
            float(
                data[
                    "target_length"
                ].mean()
            ),
    }


    metadata_path = (
        OUTPUT_DIRECTORY
        / "metadata.json"
    )


    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            metadata,
            file,
            indent=2,
        )


    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "PREPROCESSING COMPLETED SUCCESSFULLY"
    )

    print(
        "=" * 60
    )


    print(
        json.dumps(
            metadata,
            indent=2,
        )
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()