# Dataset

| Field | Value |
|---|---|
| Name | News Summary |
| Source | Kaggle, uploader `sunnysai12345` |
| URL | https://www.kaggle.com/datasets/sunnysai12345/news-summary |
| License | GPL-2.0 |
| File | `data/raw/dataset.csv` (4,514 rows, 11.9 MB) |
| Retrieved | 2026-08-08 |

Machine-readable version in `data/processed/metadata.json`.

## Columns

Six columns, two used: `ctext` (full article) is our source and `headlines` is
our target. `author`, `date`, `read_more` and `text` are unused.

`src/preprocess.py` matches `ctext` before `text`
([src/preprocess.py:167](../src/preprocess.py#L167)), so the task is
**full article → headline**. `text` is a ready-made summary, so `text → headline`
would be easier with more examples. We did not switch because it would rebuild
the splits and invalidate the LLM baseline.

## Preprocessing

`python src/preprocess.py` writes the six files in `data/processed/`:

1. Read UTF-8, falling back to latin-1 then cp1252
2. Map `ctext` → `source`, `headlines` → `target`
3. Drop rows with an empty article or headline
4. Collapse whitespace, strip, NFKC-normalise, lowercase
5. Drop duplicate articles
6. Tokenise on `\w+|[^\w\s]`
7. Keep articles of 20-400 tokens and headlines of 2-30 tokens
8. Split 80/10/10 with `random_state=42`, before building any vocabulary
9. Build vocabularies from the training split only, min frequency 2, max 30,000
10. Encode to token IDs with `<bos>`/`<eos>` and write JSONL

4,514 rows → 4,341 after cleaning → **2,691** after length filters, split into
2,152 train / 269 validation / 270 test. The biggest loss is 1,649 articles over
400 tokens.

Re-running regenerates all six files byte-identical (SHA-1 match).

## Notes for the report

- GPL-2.0 is a software license and this is a dataset. It is what the publisher
  states, but the mismatch is worth a sentence in limitations.
- The articles are third-party copyrighted news, mostly from
  `indiatoday.intoday.in` (3,024), `hindustantimes.com` (1,218) and
  `theguardian.com` (272). We redistribute the Kaggle file as-is.
- Coverage is two months of 2017 and mostly Indian news, so a model trained on
  it is not a general-purpose headline generator.

```
sunnysai12345. "News Summary." Kaggle, 2018.
https://www.kaggle.com/datasets/sunnysai12345/news-summary
Licensed GPL-2.0. Accessed 2026-08-08.
```
