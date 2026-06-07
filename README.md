# 🗜️ TokenShrink

![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat-square&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.58.0-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![tiktoken](https://img.shields.io/badge/tiktoken-0.13.0-412991?style=flat-square&logo=openai&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-multi--stage-2496ED?style=flat-square&logo=docker&logoColor=white)
![Tests](https://img.shields.io/badge/tests-67%20passed-22c55e?style=flat-square&logo=pytest&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-f59e0b?style=flat-square)
![Languages](https://img.shields.io/badge/languages-5-8b5cf6?style=flat-square)

**Multilingual Prompt Compressor** — a Streamlit app that removes greetings, redundant phrases, duplicate sentences, and unnecessary connectives from prompts before you send them to an AI model, saving tokens and reducing API costs.

---

## Features

- **5 languages supported:** English, Portuguese, Spanish, French, and German
- **4 compression filters**, each independently toggleable:
  - Remove greetings and sign-offs
  - Normalize spaces and empty lines
  - Remove redundant connectives
  - Remove redundant phrases and duplicate sentences
- **Token counter** using `tiktoken` with 4 tokenizer options (`cl100k_base`, `p50k_base`, `gpt2`, `r50k_base`)
- **Cost estimator** for GPT-4o, GPT-4o mini, Claude 3.5 Sonnet, Claude 3 Haiku, and Claude 3.7 Sonnet
- **Side-by-side diff** showing original vs. optimized prompt
- **One-click copy** via a code block rendered below the result
- **Compression progress bar** with percentage saved
- **Session history** — last 5 compressions kept in memory

---

## Project Structure

```
tokenshrink/
├── src/
│   └── streamlit_app.py    # Application source
├── tests/
│   └── test_tokenshrink.py # Pytest suite (67 tests)
├── Dockerfile              # Multi-stage production image
├── requirements.txt        # Runtime dependencies (pinned)
├── requirements-dev.txt    # Dev/test dependencies (pinned)
└── README.md
```

---

## Getting Started

### Running locally

```bash
# 1. Install runtime dependencies
pip install -r requirements.txt

# 2. Launch the app
streamlit run src/streamlit_app.py
```

The app will be available at `http://localhost:8501`.

### Running with Docker

```bash
# Build the image
docker build -t tokenshrink .

# Run the container
docker run -p 8501:8501 tokenshrink
```

Then open `http://localhost:8501` in your browser.

---

## Running Tests

```bash
# Install dev dependencies
pip install -r requirements-dev.txt

# Run the full test suite
pytest tests/ -v

# Run with coverage report
pytest tests/ -v --cov=src --cov-report=term-missing
```

All 67 tests should pass with no network access required — the tokenizer is mocked in unit tests.

---

## Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | 1.58.0 | UI framework |
| `tiktoken` | 0.13.0 | Token counting |
| `pytest` | 8.4.2 | Test runner *(dev only)* |
| `pytest-cov` | 6.1.0 | Coverage reports *(dev only)* |

---

## How It Works

Each compression pass runs a series of regex substitutions against language-specific rule sets defined in `LANG_RULES`. The pipeline is:

1. **Greeting removal** — strips salutations and sign-offs at the start/end of the prompt
2. **Connective removal** — replaces verbose connectives (*"in order to"*, *"due to the fact that"*) with a single space
3. **Phrase removal** — rewrites or removes filler constructions (*"I want you to"*, *"in a clear way"*)
4. **Deduplication** — drops sentences that appear more than once (case-insensitive), including across newlines
5. **Whitespace normalization** — collapses multiple spaces and blank lines, optionally preserving paragraph breaks

The result is then capitalised and stripped of any leading punctuation left behind by the removals.

---

## Supported Languages

| Language | Greetings | Connectives | Redundant Phrases |
|---|---|---|---|
| English | ✅ | ✅ | ✅ |
| Portuguese | ✅ | ✅ | ✅ |
| Spanish | ✅ | ✅ | ✅ |
| French | ✅ | ✅ | ✅ |
| German | ✅ | ✅ | ✅ |

---

## Docker Details

The image uses a **multi-stage build** to keep the final artifact lean:

- **Stage `builder`** — installs `build-essential` and compiles Python wheels
- **Stage `runtime`** — copies only the pre-built wheels and app source; no compiler toolchain included

The container runs as a **non-root user** (`appuser`) and includes a health check against Streamlit's built-in health endpoint:

```
GET http://localhost:8501/_stcore/health
```

---

## License

MIT
