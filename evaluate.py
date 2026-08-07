"""
Evaluation script for trained LSTM Seq2Seq model.

Runs inference on the test set, computes automatic metrics (BLEU, ROUGE),
and generates qualitative comparison tables.

Note: Install evaluation libraries first:
    pip install sacrebleu rouge-score
"""

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from dataset import Seq2SeqDataset
from model import Encoder, Decoder, Seq2Seq
from vocabulary import Vocabulary


def load_model(checkpoint_path, source_vocab, target_vocab, device):
    """
    Load a trained model from a checkpoint.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        source_vocab: Source Vocabulary object.
        target_vocab: Target Vocabulary object.
        device: torch.device.

    Returns:
        model: Loaded Seq2Seq model.
    """
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device
    )

    config = checkpoint["config"]

    encoder = Encoder(
        vocab_size=len(source_vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        padding_idx=source_vocab.pad_id,
    )

    decoder = Decoder(
        vocab_size=len(target_vocab),
        embedding_dim=config["embedding_dim"],
        hidden_dim=config["hidden_dim"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
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

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def ids_to_text(token_ids, vocab):
    """
    Convert token IDs back to text.

    Structural special tokens are removed, but <unk> is kept
    so OOV failures remain visible during qualitative analysis.
    """
    tokens = vocab.decode(token_ids)

    tokens = [
        token
        for token in tokens
        if token not in [
            "<pad>",
            "<bos>",
            "<eos>",
        ]
    ]

    return " ".join(tokens)


def evaluate_model(
        model,
        dataset,
        source_vocab,
        target_vocab,
        device,
        max_length=60,
):
    """
    Run inference on a dataset and collect predictions.

    Returns:
        results: List of dictionaries containing:
            source
            reference
            prediction
    """
    results = []

    for example in dataset:
        source_ids = example["source_ids"].to(device)

        source_length = (
                source_ids != source_vocab.pad_id
        ).sum().item()

        reference_text = example["target_text"]
        source_text = example["source_text"]

        with torch.no_grad():
            predictions, _ = model.decode(
                source_ids,
                source_length,
                max_length=max_length,
            )

        pred_ids = predictions[0]

        pred_text = ids_to_text(
            pred_ids,
            target_vocab
        )

        results.append(
            {
                "source": source_text,
                "reference": reference_text,
                "prediction": pred_text,
            }
        )

    return results


def compute_metrics(results):
    """
    Compute BLEU and ROUGE scores.

    Requires:
        pip install sacrebleu rouge-score
    """
    try:
        from sacrebleu import corpus_bleu
        from rouge_score import rouge_scorer

    except ImportError:
        print(
            "ERROR: sacrebleu and rouge-score "
            "are required."
        )
        print(
            "Install with: "
            "pip install sacrebleu rouge-score"
        )
        return None

    references = [
        result["reference"]
        for result in results
    ]

    predictions = [
        result["prediction"]
        for result in results
    ]

    # SacreBLEU expects:
    # hypotheses = [prediction1, prediction2, ...]
    #
    # references =
    # [
    #     [reference1, reference2, ...]
    # ]
    bleu = corpus_bleu(
        predictions,
        [references],
    )

    scorer = rouge_scorer.RougeScorer(
        [
            "rouge1",
            "rouge2",
            "rougeL",
        ],
        use_stemmer=True,
    )

    rouge_scores = {
        "rouge1": [],
        "rouge2": [],
        "rougeL": [],
    }

    for reference, prediction in zip(
            references,
            predictions,
    ):
        scores = scorer.score(
            reference,
            prediction,
        )

        for key in rouge_scores:
            rouge_scores[key].append(
                scores[key].fmeasure
            )

    avg_rouge = {
        key: (
            sum(values) / len(values)
            if values
            else 0.0
        )
        for key, values
        in rouge_scores.items()
    }

    return {
        "bleu": bleu.score,
        "rouge1": avg_rouge["rouge1"],
        "rouge2": avg_rouge["rouge2"],
        "rougeL": avg_rouge["rougeL"],
    }


def save_qualitative_examples(
        results,
        save_path,
        num_examples=10,
):
    """
    Save a side-by-side comparison table of
    source, reference, and prediction.
    """
    lines = [
        "# Qualitative Analysis: "
        "LSTM Seq2Seq Predictions\n"
    ]

    lines.append(
        "| # | Source | Reference | Prediction |"
    )

    lines.append(
        "|---|--------|-----------|------------|"
    )

    for i, result in enumerate(
            results[:num_examples]
    ):
        source = result["source"].replace(
            "|",
            "\\|"
        )

        reference = result["reference"].replace(
            "|",
            "\\|"
        )

        prediction = result["prediction"].replace(
            "|",
            "\\|"
        )

        lines.append(
            f"| {i + 1} | "
            f"{source} | "
            f"{reference} | "
            f"{prediction} |"
        )

    with open(
            save_path,
            "w",
            encoding="utf-8",
    ) as file:
        file.write(
            "\n".join(lines)
        )

    print(
        f"Qualitative examples saved to "
        f"{save_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate trained LSTM "
            "Seq2Seq model"
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="results/best_model.pt",
        help="Path to model checkpoint",
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help=(
            "Directory containing "
            "processed test data"
        ),
    )

    parser.add_argument(
        "--results-dir",
        type=str,
        default="results",
        help=(
            "Directory to save "
            "evaluation outputs"
        ),
    )

    parser.add_argument(
        "--device",
        type=str,
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
        help="Device to run on",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=60,
        help="Maximum decoding length",
    )

    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help=(
            "Number of qualitative "
            "examples to save"
        ),
    )

    args = parser.parse_args()

    device = torch.device(
        args.device
    )

    data_dir = Path(
        args.data_dir
    )

    results_dir = Path(
        args.results_dir
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Using device: {device}"
    )

    # Load vocabularies
    print(
        "Loading vocabularies..."
    )

    source_vocab = Vocabulary.load(
        data_dir / "source_vocab.json"
    )

    target_vocab = Vocabulary.load(
        data_dir / "target_vocab.json"
    )

    # Load model
    print(
        "Loading model from checkpoint..."
    )

    model = load_model(
        args.checkpoint,
        source_vocab,
        target_vocab,
        device,
    )

    print(
        "Model loaded. Parameters: "
        f"{model.count_parameters():,}"
    )

    # Load test dataset
    print(
        "Loading test dataset..."
    )

    test_dataset = Seq2SeqDataset(
        data_dir / "test.jsonl"
    )

    print(
        f"Test examples: "
        f"{len(test_dataset)}"
    )

    # Run inference
    print(
        "\nRunning inference on test set..."
    )

    results = evaluate_model(
        model=model,
        dataset=test_dataset,
        source_vocab=source_vocab,
        target_vocab=target_vocab,
        device=device,
        max_length=args.max_length,
    )

    # Compute metrics
    print(
        "\nComputing metrics..."
    )

    metrics = compute_metrics(
        results
    )

    if metrics:
        print(
            "\n" + "=" * 50
        )

        print(
            "EVALUATION RESULTS"
        )

        print(
            "=" * 50
        )

        print(
            f"BLEU:    "
            f"{metrics['bleu']:.2f}"
        )

        print(
            f"ROUGE-1: "
            f"{metrics['rouge1']:.4f}"
        )

        print(
            f"ROUGE-2: "
            f"{metrics['rouge2']:.4f}"
        )

        print(
            f"ROUGE-L: "
            f"{metrics['rougeL']:.4f}"
        )

        print(
            "=" * 50
        )

        metrics_path = (
                results_dir
                / "evaluation_metrics.json"
        )

        with open(
                metrics_path,
                "w",
                encoding="utf-8",
        ) as file:
            json.dump(
                metrics,
                file,
                indent=2,
            )

        print(
            f"Metrics saved to "
            f"{metrics_path}"
        )

    # Save qualitative examples
    examples_path = (
            results_dir
            / "qualitative_examples.md"
    )

    save_qualitative_examples(
        results,
        examples_path,
        args.num_examples,
    )

    # Save all predictions
    predictions_path = (
            results_dir
            / "test_predictions.json"
    )

    with open(
            predictions_path,
            "w",
            encoding="utf-8",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
            ensure_ascii=False,
        )

    print(
        f"All predictions saved to "
        f"{predictions_path}"
    )


if __name__ == "__main__":
    main()