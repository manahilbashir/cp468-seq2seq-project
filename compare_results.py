"""
Compare LSTM seq2seq predictions against the LLM baseline on the identical
test set (PRD section 4.3: report BLEU/ROUGE for the LSTM and for the LLM in
each prompt setting, plus a qualitative side-by-side table).

Run after both:
    python evaluate.py --checkpoint results/best_model.pt ...
    python llm_baseline.py --limit 0
"""

import argparse
import json
from pathlib import Path

from evaluate import compute_metrics
from llm_baseline import FEW_SHOT_TEMPLATE, MODEL_NAME, ZERO_SHOT_TEMPLATE

# Rough English heuristic (~4 characters/token) since no official Gemini
# tokenizer is available offline. Good enough for an order-of-magnitude
# cost estimate -- replace with the exact $/1M-token rate for MODEL_NAME
# from https://ai.google.dev/pricing before quoting a final USD figure
# in the report.
CHARS_PER_TOKEN_ESTIMATE = 4


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))
    return records


def has_repetition(text):
    tokens = text.split()
    bigrams = list(zip(tokens, tokens[1:]))
    return len(bigrams) >= 3 and len(set(bigrams)) < len(bigrams) * 0.7


def build_rows(lstm_results, llm_results):
    rows = []
    for lstm, llm in zip(lstm_results, llm_results):
        rows.append(
            {
                "source": lstm["source"],
                "reference": lstm["reference"],
                "lstm_prediction": lstm["prediction"],
                "llm_zero_shot": llm.get("zero_shot") or "",
                "llm_few_shot": llm.get("few_shot") or "",
                "source_length": len(lstm["source"].split()),
            }
        )
    return rows


def select_diverse_examples(rows, num_examples):
    """
    Spread selection across the source-length distribution (shortest to
    longest) instead of cherry-picking, per the PRD's "not just your best
    cases" requirement.
    """
    sorted_rows = sorted(rows, key=lambda r: r["source_length"])
    n = len(sorted_rows)
    if n <= num_examples:
        return sorted_rows

    step = (n - 1) / (num_examples - 1)
    indices = sorted({round(i * step) for i in range(num_examples)})
    return [sorted_rows[i] for i in indices]


def estimate_llm_cost(llm_results):
    """
    Estimate token volume for the PRD's required LLM cost report (4.2).
    Two requests per example (zero-shot + few-shot), one prompt char count
    per template plus the article, one completion char count per response.
    """
    num_requests = 0
    prompt_chars = 0
    completion_chars = 0

    for record in llm_results:
        article = record["source"]

        for template, key in (
            (ZERO_SHOT_TEMPLATE, "zero_shot"),
            (FEW_SHOT_TEMPLATE, "few_shot"),
        ):
            if record.get(key) is None:
                continue
            num_requests += 1
            prompt_chars += len(template.format(article=article))
            completion_chars += len(record[key])

    prompt_tokens = round(prompt_chars / CHARS_PER_TOKEN_ESTIMATE)
    completion_tokens = round(completion_chars / CHARS_PER_TOKEN_ESTIMATE)

    return {
        "model": MODEL_NAME,
        "num_requests": num_requests,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_completion_tokens": completion_tokens,
        "estimation_method": (
            f"~{CHARS_PER_TOKEN_ESTIMATE} chars/token heuristic, not the "
            "real Gemini tokenizer -- treat as order-of-magnitude"
        ),
        "note": (
            f"Multiply the token counts above by the current {MODEL_NAME} "
            "$/1M-token input and output rates from https://ai.google.dev/pricing "
            "to get a USD figure for the report (PRD 4.2)."
        ),
    }


def annotate(row):
    notes = []
    if not row["lstm_prediction"].strip():
        notes.append("LSTM empty output")
    elif has_repetition(row["lstm_prediction"]):
        notes.append("LSTM repetition")

    ref_len = max(1, len(row["reference"].split()))
    if row["llm_zero_shot"] and len(row["llm_zero_shot"].split()) > 3 * ref_len:
        notes.append("LLM zero-shot over-elaborates")
    if not row["llm_zero_shot"].strip():
        notes.append("LLM zero-shot empty/error")

    return ", ".join(notes) if notes else "-"


def save_qualitative_table(rows, save_path, num_examples):
    lines = [
        "# Qualitative Comparison: LSTM vs LLM (zero-shot / few-shot)\n",
        "Examples are spread across the source-length distribution "
        "(shortest to longest article) rather than cherry-picked. "
        "`Notes` are automatic heuristic flags only — verify manually "
        "and assign the PRD error categories "
        "(under-translation, repetition, hallucination, OOV, fluency vs. "
        "adequacy) by hand for the report.\n",
        "| # | Source (truncated) | Reference | LSTM | LLM Zero-shot | LLM Few-shot | Notes |",
        "|---|---------------------|-----------|------|----------------|---------------|-------|",
    ]

    for i, row in enumerate(select_diverse_examples(rows, num_examples), start=1):
        source_preview = row["source"][:150].replace("\n", " ").replace("|", "/")
        if len(row["source"]) > 150:
            source_preview += "..."
        lines.append(
            f"| {i} | {source_preview} | {row['reference']} | "
            f"{row['lstm_prediction']} | {row['llm_zero_shot']} | "
            f"{row['llm_few_shot']} | {annotate(row)} |"
        )

    with open(save_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))

    print(f"Qualitative comparison saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare LSTM vs LLM baseline on the identical test set"
    )
    parser.add_argument("--lstm-predictions", default="results/test_predictions.json")
    parser.add_argument("--llm-outputs", default="results/llm_outputs.jsonl")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--num-examples", type=int, default=12)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(args.lstm_predictions, "r", encoding="utf-8") as file:
        lstm_results = json.load(file)

    llm_results = load_jsonl(args.llm_outputs)

    if len(lstm_results) != len(llm_results):
        print(
            f"WARNING: LSTM has {len(lstm_results)} examples, LLM has "
            f"{len(llm_results)}. Comparing on the overlap only."
        )

    n = min(len(lstm_results), len(llm_results))
    lstm_results = lstm_results[:n]
    llm_results = llm_results[:n]

    mismatches = sum(
        1 for a, b in zip(lstm_results, llm_results) if a["reference"] != b["reference"]
    )
    if mismatches:
        print(
            f"WARNING: {mismatches}/{n} examples have mismatched references "
            "between LSTM and LLM outputs -- they may not be aligned to the "
            "same test rows."
        )

    print("\nScoring LSTM predictions...")
    lstm_metrics = compute_metrics(lstm_results)

    print("Scoring LLM zero-shot predictions...")
    zero_shot_results = [
        {"reference": r["reference"], "prediction": r.get("zero_shot") or ""}
        for r in llm_results
    ]
    zero_shot_metrics = compute_metrics(zero_shot_results)

    print("Scoring LLM few-shot predictions...")
    few_shot_results = [
        {"reference": r["reference"], "prediction": r.get("few_shot") or ""}
        for r in llm_results
    ]
    few_shot_metrics = compute_metrics(few_shot_results)

    comparison = {
        "num_examples": n,
        "lstm_attention": lstm_metrics,
        "llm_zero_shot": zero_shot_metrics,
        "llm_few_shot": few_shot_metrics,
        "llm_cost_estimate": estimate_llm_cost(llm_results),
    }

    metrics_path = results_dir / "comparison_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as file:
        json.dump(comparison, file, indent=2)

    print(f"\n{'=' * 60}")
    print("LSTM vs LLM COMPARISON")
    print(f"{'=' * 60}")
    for name, metrics in comparison.items():
        if isinstance(metrics, dict) and "bleu" in metrics:
            print(
                f"{name:16s} BLEU={metrics['bleu']:.2f}  "
                f"ROUGE-1={metrics['rouge1']:.4f}  "
                f"ROUGE-2={metrics['rouge2']:.4f}  "
                f"ROUGE-L={metrics['rougeL']:.4f}"
            )
    print(f"{'=' * 60}")
    print(f"Metrics saved to {metrics_path}")

    rows = build_rows(lstm_results, llm_results)
    save_qualitative_table(rows, results_dir / "qualitative_comparison.md", args.num_examples)


if __name__ == "__main__":
    main()
