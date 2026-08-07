import argparse
import json
import os
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

import certifi


MODEL_NAME = "gemini-3.5-flash-lite"

API_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{MODEL_NAME}:generateContent"
)

ZERO_SHOT_TEMPLATE = """You are a professional news editor.

Generate one concise and factual headline for the article below.

Rules:
- Output only the headline.
- Do not add quotation marks.
- Do not invent information.
- Keep the headline under 15 words.

Article:
{article}

Headline:
"""

FEW_SHOT_TEMPLATE = """You are a professional news editor.

Generate one concise and factual headline for the final article.

Rules:
- Output only the headline.
- Do not add quotation marks.
- Do not invent information.
- Keep the headline under 15 words.

Example 1:
Article:
Apple announced a new series of laptops featuring faster processors and longer battery life.

Headline:
Apple Unveils Faster Laptops With Longer Battery Life

Example 2:
Article:
Heavy rainfall caused flooding across several neighbourhoods and forced officials to close roads.

Headline:
Heavy Rain Triggers Flooding and Road Closures

Example 3:
Article:
The national football team defeated its rival 2-1 after scoring in the final minutes.

Headline:
Late Goal Secures 2-1 Victory for National Team

Now generate a headline for this article.

Article:
{article}

Headline:
"""


def load_examples(path: str, limit: int | None) -> list[dict]:
    examples = []

    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue

            record = json.loads(line)

            examples.append(
                {
                    "source": record["source"],
                    "target": record["target"],
                }
            )

            if limit is not None and len(examples) >= limit:
                break

    return examples


def clean_headline(text: str) -> str:
    headline = text.strip()

    if headline.lower().startswith("headline:"):
        headline = headline[len("headline:"):].strip()

    headline = headline.strip('"')
    headline = " ".join(headline.split())

    return headline


def call_gemini(api_key: str, prompt: str) -> str:
    request_body = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "maxOutputTokens": 40
        }
    }

    request = urllib.request.Request(
        API_URL,
        data=json.dumps(request_body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
        },
        method="POST",
    )

    ssl_context = ssl.create_default_context(
        cafile=certifi.where()
    )

    try:
        with urllib.request.urlopen(
            request,
            timeout=60,
            context=ssl_context
        ) as response:
            result = json.loads(
                response.read().decode("utf-8")
            )

    except urllib.error.HTTPError as error:
        details = error.read().decode("utf-8")

        raise RuntimeError(
            f"Gemini API error {error.code}: {details}"
        ) from error

    except urllib.error.URLError as error:
        raise RuntimeError(
            f"Network error: {error}"
        ) from error

    candidates = result.get("candidates", [])

    if not candidates:
        raise RuntimeError(
            f"Gemini returned no candidates: {result}"
        )

    parts = (
        candidates[0]
        .get("content", {})
        .get("parts", [])
    )

    if not parts or "text" not in parts[0]:
        raise RuntimeError(
            f"Gemini returned no text: {result}"
        )

    return clean_headline(parts[0]["text"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate zero-shot and few-shot headlines with Gemini."
    )

    parser.add_argument(
        "--input",
        default="data/processed/test.jsonl",
        help="Path to the processed test set."
    )

    parser.add_argument(
        "--output",
        default="results/llm_outputs.jsonl",
        help="Path to save Gemini outputs."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Number of test examples. Use 0 for the full test set."
    )

    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Delay between API requests."
    )

    args = parser.parse_args()

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set in Terminal."
        )

    limit = None if args.limit == 0 else args.limit

    examples = load_examples(
        path=args.input,
        limit=limit
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"Loaded {len(examples)} examples.")
    print(f"Using model: {MODEL_NAME}")

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as output_file:

        for index, example in enumerate(
            examples,
            start=1
        ):
            print(
                f"Processing {index}/{len(examples)}"
            )

            result = {
                "id": index,
                "source": example["source"],
                "reference": example["target"],
                "model": MODEL_NAME,
                "zero_shot": None,
                "few_shot": None,
                "error": None,
            }

            try:
                zero_prompt = ZERO_SHOT_TEMPLATE.format(
                    article=example["source"]
                )

                result["zero_shot"] = call_gemini(
                    api_key=api_key,
                    prompt=zero_prompt
                )

                time.sleep(args.delay)

                few_prompt = FEW_SHOT_TEMPLATE.format(
                    article=example["source"]
                )

                result["few_shot"] = call_gemini(
                    api_key=api_key,
                    prompt=few_prompt
                )

            except Exception as error:
                result["error"] = str(error)

            output_file.write(
                json.dumps(
                    result,
                    ensure_ascii=False
                ) + "\n"
            )

            output_file.flush()
            time.sleep(args.delay)

    print(
        f"Finished. Results saved to {args.output}"
    )


if __name__ == "__main__":
    main()