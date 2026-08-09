"""
Quantify failure modes for all three systems across the whole test set.

PRD 4.3 asks for categorized errors, and suggested discussion 3 explicitly says
to verify the typical LSTM-vs-LLM failure story against the data rather than
assert it. This script measures each category over all 270 test examples instead
of generalizing from the 12 rows in qualitative_comparison.md.

Every number here is a *proxy* computed from surface text -- stated as such in
the output -- and each category writes out sample IDs so a human can confirm the
label before it goes in the report.

Run after evaluate.py and llm_baseline.py:
    python scripts/error_analysis.py
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from tokenizer import tokenize  # noqa: E402
from vocabulary import Vocabulary  # noqa: E402

# Words too common to count as evidence of copying or of hallucination.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "as", "at", "by", "from", "is", "are", "was", "were", "be", "been",
    "has", "have", "had", "will", "would", "can", "could", "may", "might",
    "not", "no", "his", "her", "its", "their", "this", "that", "these",
    "those", "it", "he", "she", "they", "we", "you", "i", "s", "t",
}

# The prompt in llm_baseline.py instructs "Keep the headline under 15 words".
PROMPT_WORD_LIMIT = 15

# A prediction counts as repetitive when its bigrams are mostly duplicates.
REPETITION_UNIQUE_BIGRAM_RATIO = 0.7

MAX_SAMPLE_IDS = 5


def content_words(text):
    return [
        token
        for token in tokenize(text)
        if token.isalnum() and token not in STOPWORDS and len(token) > 1
    ]


def repetition_score(text):
    """Fraction of bigrams that are duplicates. 0.0 means all distinct."""
    tokens = text.split()
    bigrams = list(zip(tokens, tokens[1:]))
    if len(bigrams) < 3:
        return 0.0
    return 1.0 - (len(set(bigrams)) / len(bigrams))


def is_repetitive(text):
    tokens = text.split()
    bigrams = list(zip(tokens, tokens[1:]))
    if len(bigrams) < 3:
        return False
    return len(set(bigrams)) < len(bigrams) * REPETITION_UNIQUE_BIGRAM_RATIO


def unsupported_content_rate(prediction, source):
    """
    Share of the prediction's content words that never appear in the article.

    This is an extrinsic-hallucination *proxy*, not a hallucination detector: a
    correct synonym or a legitimate abstraction also scores here. Use it to rank
    candidates for manual inspection, and quote only hand-checked cases.
    """
    predicted = content_words(prediction)
    if not predicted:
        return 0.0, []
    source_vocab = set(content_words(source))
    unsupported = [w for w in predicted if w not in source_vocab]
    return len(unsupported) / len(predicted), unsupported


def summarize(name, records, target_vocab_tokens):
    """records: list of (example_id, source, reference, prediction)."""
    n = len(records)
    stats = {
        "system": name,
        "num_examples": n,
        "empty": 0,
        "repetitive": 0,
        "over_length_vs_reference": 0,
        "over_prompt_word_limit": 0,
        "high_unsupported_content": 0,
    }
    samples = {key: [] for key in
               ("empty", "repetitive", "over_length_vs_reference",
                "over_prompt_word_limit", "high_unsupported_content")}

    pred_lengths, ref_lengths, rep_scores, unsupported_rates = [], [], [], []
    oov_emissions = 0
    total_pred_tokens = 0

    for example_id, source, reference, prediction in records:
        pred_tokens = prediction.split()
        ref_tokens = reference.split()
        pred_lengths.append(len(pred_tokens))
        ref_lengths.append(len(ref_tokens))
        rep_scores.append(repetition_score(prediction))

        rate, _ = unsupported_content_rate(prediction, source)
        unsupported_rates.append(rate)

        for token in tokenize(prediction):
            total_pred_tokens += 1
            if token not in target_vocab_tokens:
                oov_emissions += 1

        def flag(key, condition):
            if condition:
                stats[key] += 1
                if len(samples[key]) < MAX_SAMPLE_IDS:
                    samples[key].append(example_id)

        flag("empty", not prediction.strip())
        flag("repetitive", is_repetitive(prediction))
        flag("over_length_vs_reference",
             len(pred_tokens) > 2 * max(1, len(ref_tokens)))
        flag("over_prompt_word_limit", len(pred_tokens) > PROMPT_WORD_LIMIT)
        flag("high_unsupported_content", rate >= 0.5)

    stats["mean_prediction_tokens"] = round(sum(pred_lengths) / n, 2)
    stats["mean_reference_tokens"] = round(sum(ref_lengths) / n, 2)
    stats["mean_repetition_score"] = round(sum(rep_scores) / n, 4)
    stats["mean_unsupported_content_rate"] = round(
        sum(unsupported_rates) / n, 4
    )
    stats["distinct_predictions"] = len({r[3] for r in records})
    stats["tokens_outside_lstm_target_vocab"] = (
        round(oov_emissions / total_pred_tokens, 4) if total_pred_tokens else 0.0
    )
    stats["sample_ids"] = {k: v for k, v in samples.items() if v}
    return stats


def percent(count, total):
    return f"{count} / {total} ({100 * count / total:.1f}%)"


def main():
    results_dir = REPO_ROOT / "results"
    data_dir = REPO_ROOT / "data" / "processed"

    lstm = json.loads((results_dir / "test_predictions.json").read_text("utf-8"))
    llm = [
        json.loads(line)
        for line in (results_dir / "llm_outputs.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]

    n = min(len(lstm), len(llm))
    lstm, llm = lstm[:n], llm[:n]

    target_vocab = Vocabulary.load(data_dir / "target_vocab.json")
    target_vocab_tokens = set(target_vocab.token_to_id)

    systems = {
        "LSTM + attention": [
            (i + 1, r["source"], r["reference"], r["prediction"])
            for i, r in enumerate(lstm)
        ],
        "Gemini zero-shot": [
            (i + 1, r["source"], r["reference"], r.get("zero_shot") or "")
            for i, r in enumerate(llm)
        ],
        "Gemini few-shot": [
            (i + 1, r["source"], r["reference"], r.get("few_shot") or "")
            for i, r in enumerate(llm)
        ],
    }

    all_stats = [
        summarize(name, records, target_vocab_tokens)
        for name, records in systems.items()
    ]

    # Reference-side OOV: an upper bound on what the LSTM could ever emit.
    ref_tokens = [t for r in lstm for t in tokenize(r["reference"])]
    ref_oov = sum(1 for t in ref_tokens if t not in target_vocab_tokens)
    refs_with_oov = sum(
        1 for r in lstm
        if any(t not in target_vocab_tokens for t in tokenize(r["reference"]))
    )

    # What the LSTM emits instead of the right word.
    lstm_counter = Counter(
        t for r in lstm for t in tokenize(r["prediction"])
    )

    lines = [
        "# Measured error analysis - all 270 test examples\n",
        "Generated by `scripts/error_analysis.py`. Every figure below is a "
        "surface-text **proxy**, not a human judgement. Sample IDs are given "
        "so each category can be spot-checked before it is quoted in the "
        "report; IDs are 1-based row numbers into "
        "`results/test_predictions.json` and `results/llm_outputs.jsonl`.\n",
        "## Category rates\n",
        "| Metric | LSTM + attention | Gemini zero-shot | Gemini few-shot |",
        "|---|---|---|---|",
    ]

    def row(label, key, as_percent=True):
        cells = []
        for s in all_stats:
            cells.append(
                percent(s[key], s["num_examples"]) if as_percent else str(s[key])
            )
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    row("Empty output", "empty")
    row("Repetition loop (<70% unique bigrams)", "repetitive")
    row("Over 2x reference length", "over_length_vs_reference")
    row(f"Over the prompt's {PROMPT_WORD_LIMIT}-word limit",
        "over_prompt_word_limit")
    row("Content mostly absent from article (>=50%)", "high_unsupported_content")
    row("Distinct predictions", "distinct_predictions", as_percent=False)
    row("Mean prediction length (words)", "mean_prediction_tokens", False)
    row("Mean repetition score (0 = none)", "mean_repetition_score", False)
    row("Mean unsupported-content rate", "mean_unsupported_content_rate", False)

    lines += [
        f"\nMean reference length: {all_stats[0]['mean_reference_tokens']} words.\n",
        "> **Units.** Lengths in this table are **whitespace-separated words**, "
        "because that is the unit the prompt's \"under 15 words\" rule is "
        "written in. Elsewhere the project reports **regex tokens** "
        "(`src/tokenizer.py` splits punctuation off, so `India's` is 3 tokens "
        "but 1 word). Same references measure 9.19 words and 10.85 tokens - "
        "both correct, different units. Do not mix them in one table.\n",
        "## Out-of-vocabulary ceiling (LSTM only)\n",
        f"- {ref_oov} / {len(ref_tokens)} "
        f"({100 * ref_oov / len(ref_tokens):.2f}%) of test reference tokens are "
        "outside the 2,527-token target vocabulary.",
        f"- {refs_with_oov} / {len(lstm)} "
        f"({100 * refs_with_oov / len(lstm):.1f}%) of references contain at "
        "least one such token.",
        "- The LSTM cannot emit these under any decoding strategy, so this is a "
        "hard ceiling on its attainable ROUGE. It does not apply to the LLM, "
        "which decodes over its own subword vocabulary.\n",
        "## What the LSTM emits instead - 20 most frequent predicted tokens\n",
        "| Token | Count |",
        "|---|---|",
    ]
    for token, count in lstm_counter.most_common(20):
        lines.append(f"| `{token}` | {count} |")

    lines += [
        "\n## Sample IDs per category (for manual verification)\n",
        "| System | Category | Example IDs |",
        "|---|---|---|",
    ]
    for s in all_stats:
        for category, ids in s["sample_ids"].items():
            lines.append(
                f"| {s['system']} | {category} | "
                + ", ".join(str(i) for i in ids)
                + " |"
            )

    report_path = results_dir / "error_analysis.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    json_path = results_dir / "error_analysis.json"
    json_path.write_text(
        json.dumps(
            {
                "systems": all_stats,
                "reference_oov_token_rate": round(ref_oov / len(ref_tokens), 4),
                "references_with_oov": refs_with_oov,
                "num_examples": n,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for s in all_stats:
        print(
            f"{s['system']:<20} empty={s['empty']:<4} "
            f"repetitive={s['repetitive']:<4} "
            f"over_limit={s['over_prompt_word_limit']:<4} "
            f"unsupported>=50%={s['high_unsupported_content']:<4} "
            f"mean_len={s['mean_prediction_tokens']}"
        )
    print(f"\nWritten to {report_path} and {json_path}")


if __name__ == "__main__":
    main()
