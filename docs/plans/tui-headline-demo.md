# Plan: Interactive TUI Headline Demo

**Status:** Deferred — execute after Roles 1–4 (and ideally Role 3) are done  
**Owner:** Safdar (optional demo polish, not a graded Role 2 requirement)  
**Goal:** Paste a news article in the terminal and get a live LSTM headline (optionally side-by-side with the LLM baseline)

---

## Short answer: can you do that today?

**No.** With the current Role split, nothing accepts a free-form article and returns a headline interactively.

| Role | What it produces | Interactive paste → headline? |
|------|------------------|--------------------------------|
| 1 · Data | CSV → `train/val/test.jsonl` + vocabs | No — batch files only |
| 2 · LSTM | `model.py`, `train.py`, checkpoint | No — train only; `decode()` exists but no UI |
| 3 · LLM | API/script on the **test set** | No — scripted prompts, not a chat UI |
| 4 · Eval | `evaluate.py` over **pre-tokenized test.jsonl** | No — reads dataset rows, not pasted text |
| 5 · Report/video | PDF + recording | Can screen-record eval tables |

So for the demo video today you’d show metrics + saved example tables.  
**Live “type article → get headline” only appears if someone builds this TUI (or a CLI/web app) at the end.**

---

## Prerequisites before you start this plan

Do **not** build the TUI until:

1. **Role 1** shipped a real dataset + re-ran `src/preprocess.py` (vocabs in the thousands).
2. **Role 2** trained a real checkpoint: `results/best_model.pt` (after Role 1 data exists).
3. **Role 4** eval path works (proves load + decode + metrics).
4. **Optional but recommended — Role 3:** `llm_baseline.py` (or a small callable) so the TUI can show LSTM vs LLM on the same pasted article.

Until then, a TUI would only “work” on the 10 toy examples and look broken in the video.

---

## Product scope (what to build)

### In scope

- Terminal UI (TUI) that:
  - Loads checkpoint + source/target vocabs once at startup
  - Lets the user paste / edit an article
  - On **Generate**, runs the **same** preprocessing as training (`tokenize` → `Vocabulary.encode` → truncate) then `Seq2Seq.decode`
  - Shows the predicted headline
  - Shows basic status (device, model loaded, errors, OOV / length warnings)
- Optional second panel: LLM headline for the same article (calls Role 3 helper if present)
- Optional: load a sample from `test.jsonl` for one-click demos

### Out of scope

- Web hosting / Gradio / Streamlit (different plan)
- Retraining or changing the LSTM architecture
- Replacing Role 4 batch evaluation
- Editing Role 1 dataset files

---

## Architecture

```
┌─────────────────────────────────────────────┐
│  demo_tui.py  (Textual app)                 │
│  - Article input  - Headline output         │
│  - Generate / Clear / Quit / Load sample    │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│  src/inference.py  (NEW shared helper)      │
│  text → tokenize → encode → tensor          │
│  → model.decode → ids_to_text               │
│  reuse load_model from evaluate.py          │
└──────────────────┬──────────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
   results/best_model.pt   data/processed/*_vocab.json
```

**Why a shared `src/inference.py`?**  
Today `evaluate.py` only knows how to run on already-encoded `test.jsonl` rows. Interactive demo needs **raw string → IDs**. That helper should live once and be reused by the TUI (and a future one-liner CLI if wanted). Prefer extracting `load_model` / `ids_to_text` from `evaluate.py` into this helper so Role 4 and the demo don’t diverge.

### Critical correctness rule

Inference must match training preprocessing:

1. `tokenizer.tokenize(article)` (same cleaning / lowercase)
2. `source_vocab.encode(tokens)` (includes `<bos>` / `<eos>` like preprocess)
3. Truncate to `metadata.json` → `maximum_source_length` (or checkpoint config if stored)
4. `model.decode(source_ids, length, max_length=…)`
5. Strip special tokens when displaying (same as `evaluate.ids_to_text`)

If this drifts from preprocess, demos will look random even with a good model.

---

## Implementation units (execute in order)

### 1. Shared inference helper — `src/inference.py`

**Work**

- Move or wrap `load_model` + `ids_to_text` (from `evaluate.py`)
- Add `prepare_source(text, source_vocab, max_source_length) -> (ids_tensor, length)`
- Add `generate_headline(model, source_vocab, target_vocab, text, device, max_decode_length) -> str`
- Load `maximum_source_length` from `data/processed/metadata.json` when present

**Tests / checks**

- Unit-style smoke: fixed short string → non-empty headline string (with toy or real checkpoint)
- Same article twice → same headline (greedy decode is deterministic)
- Empty input → clear error, no crash
- Very long input → truncated, no shape errors

**Refactor note:** Update `evaluate.py` to import from `src/inference.py` so there is one load path.

### 2. Dependencies

- Add `textual` (pin version) to `requirements.txt`
- Keep torch / existing stack unchanged

### 3. TUI app — `demo_tui.py`

**Suggested layout (Textual)**

| Region | Content |
|--------|---------|
| Header | App title, device, checkpoint path, “model ready” |
| Left / top | Multiline article `TextArea` |
| Right / bottom | Headline result + optional LLM result |
| Footer | Keybindings: Generate · Clear · Load sample · Quit |

**Bindings (suggested)**

- `ctrl+g` — Generate LSTM headline  
- `ctrl+l` — Generate LLM headline (if Role 3 wired; else disable + hint)  
- `ctrl+o` — Load random/first test example into the article box  
- `ctrl+c` / `q` — Quit  

**CLI flags**

```bash
python demo_tui.py \
  --checkpoint results/best_model.pt \
  --data-dir data/processed \
  --device cpu   # or cuda
```

**UX details**

- Disable Generate while a run is in progress; show “Generating…”
- On failure (missing checkpoint, OOM, empty text), show message in the UI — don’t traceback the whole TUI
- Warn if article tokenizes to mostly `<unk>` (weak headline expected)

### 4. Optional LLM panel (after Role 3)

- Thin adapter: `generate_llm_headline(article: str) -> str` wrapping whatever Role 3 ships
- If `llm_baseline.py` has no callable API, add a small function there rather than duplicating prompts in the TUI
- TUI calls it only when user hits the LLM action / when API key env vars exist

### 5. Docs

- Short “Interactive demo” subsection in `README.md`: install Textual, run command, prerequisites (trained checkpoint)
- Note for Role 5: record the TUI for the live portion of the 8‑min video

---

## Effort estimate

| Piece | Time |
|-------|------|
| `src/inference.py` + wire `evaluate.py` | ~1–2 hours |
| Basic Textual TUI (LSTM only) | ~2–4 hours |
| Polish + sample loader + warnings | ~1–2 hours |
| LLM side-by-side | ~1–2 hours (depends on Role 3 API shape) |
| **Total** | **~½–1 day** once prerequisites exist |

---

## Risks

| Risk | Mitigation |
|------|------------|
| Building before real data/checkpoint | Wait; toy demo looks broken on camera |
| Preprocess mismatch (tokenize/encode) | Share one helper with training path |
| Textual version churn | Pin in `requirements.txt` |
| LLM API cost/latency in live demo | Cache 2–3 canned articles; optional offline LLM skip |
| Greedy decode quality on long news | Truncate + use articles similar to training domain |

---

## Definition of done

- [ ] `python demo_tui.py` opens UI with model loaded  
- [ ] Paste article → Generate → headline appears without leaving the TUI  
- [ ] Load sample from test set works  
- [ ] README documents how to run it  
- [ ] (Optional) Same article shows LSTM + LLM headlines side by side  
- [ ] Role 5 can screen-record a 30–60s live generation clip  

---

## Explicit non-goals for this plan

- Do not change Role 1 dataset or preprocess defaults as part of the TUI  
- Do not block Role 2/3/4 completion on this work  
- Do not treat the TUI as a required course deliverable — it’s demo sugar  

---

## When you’re ready to execute

1. Confirm `results/best_model.pt` exists and `evaluate.py` looks sane on the real test set.  
2. Switch to Agent mode (or implement yourself) starting with **unit 1** (`src/inference.py`), then the Textual app.  
3. Only then wire Role 3 into the second panel.
