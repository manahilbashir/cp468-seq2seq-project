# Dataset: provenance, license and attribution

Documents where `data/raw/dataset.csv` came from, the terms it is redistributed
under, and how it was transformed into `data/processed/`. Machine-readable
equivalents of the top table are in `data/processed/metadata.json`.

## Source

| Field | Value |
|---|---|
| Name | News Summary |
| Publisher | Kaggle, uploader `sunnysai12345` |
| URL | https://www.kaggle.com/datasets/sunnysai12345/news-summary |
| Stated license | **GPL-2.0** |
| File in this repo | `data/raw/dataset.csv` (4,514 rows, 11.9 MB) |
| Retrieved | 2026-08-08 |

## What the raw file contains

Six columns. Only two are used.

| Column | Description | Used? |
|---|---|---|
| `author` | Byline of the summary writer | no |
| `date` | Publication date, 01 Apr 2017-31 May 2017 | no |
| `headlines` | Short headline - **our target** | **yes** |
| `read_more` | URL of the originating news article | no |
| `text` | ~70-token summary of the article | no - see note below |
| `ctext` | Full article body - **our source** | **yes** |

Underlying articles trace back via `read_more` to Indian and UK news outlets,
predominantly `indiatoday.intoday.in` (3,024), `hindustantimes.com` (1,218) and
`theguardian.com` (272). The compilation is redistributed here under the
GPL-2.0 terms stated by the Kaggle publisher; copyright in the underlying
article text remains with those publishers. Use here is limited to
non-commercial academic coursework for CP468.

`src/preprocess.py` matches the `ctext` column before `text`
([src/preprocess.py:167](../src/preprocess.py#L167)), so the task as trained is
**full article → headline**. The `text` column is a ready-made short summary; a
`text → headline` variant would be a different, easier task on more examples.
That trade-off is documented but not taken, because changing it would rebuild
the splits and invalidate the already-measured LLM baseline.

## Licensing notes for the report

- GPL-2.0 is a **copyleft** license. Redistributing the dataset inside this
  public repository is what the license permits; it also means downstream reuse
  of the redistributed data inherits GPL-2.0 obligations.
- GPL-2.0 was written for software, not data. It is what the publisher states,
  so it is what we cite - but the mismatch between a software license and a
  dataset is worth a sentence in the limitations section.
- The underlying articles are third-party copyrighted news content. The dataset
  redistributes short summaries and headlines rather than full publisher-owned
  pages, and we add no further redistribution beyond the Kaggle file itself.
- Personal data: the corpus contains named individuals as ordinary news
  subjects, plus summary-writer bylines in `author`. No special-category
  personal data, and `author` is unused by the pipeline.
- Provenance is narrow - two months of 2017, India-weighted coverage. Models
  trained on it should not be presented as general-purpose headline generators.

## Raw → processed

`python src/preprocess.py` reads `data/raw/dataset.csv` and writes the six files
in `data/processed/`. Steps, in order:

1. Read with UTF-8, falling back to latin-1 then cp1252
2. Map `ctext` → `source`, `headlines` → `target`
3. Drop rows with a missing or empty article or headline
4. Collapse whitespace, strip, NFKC-normalise, lowercase
5. Drop duplicate articles
6. Tokenise on `\w+|[^\w\s]`
7. Keep articles of 20-400 tokens and headlines of 2-30 tokens
8. Split 80/10/10 with `random_state=42`, **before** building any vocabulary
9. Build source and target vocabularies from the **training split only**,
   minimum frequency 2, maximum size 30,000
10. Encode each split to token IDs with `<bos>`/`<eos>` and write JSONL

4,514 raw rows → 4,341 after cleaning and deduplication → **2,691** after the
length filters, split into 2,152 train / 269 validation / 270 test. The largest
single loss is 1,649 rows whose articles exceed 400 tokens.

**Reproducibility is verified, not just claimed.** Re-running
`python src/preprocess.py` regenerates all six files in `data/processed/`
byte-identical (SHA-1 match on `metadata.json`, both vocabularies and all three
`.jsonl` splits).

## Citation

```
sunnysai12345. "News Summary." Kaggle, 2018.
https://www.kaggle.com/datasets/sunnysai12345/news-summary
Licensed GPL-2.0. Accessed 2026-08-08.
```
