# question-dedup

Duplicate question detection, context selection, and answer generation using embeddings and Gemini LLM.

## Project Structure

```text
question-dedup/
├── .env.example              # Environment variables template (GEMINI_API_KEY)
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── demo.py                   # Demonstration script
├── data/
│   └── corpus.json           # Question corpus
├── src/
│   └── qdedup/
│       ├── __init__.py
│       ├── models.py         # Data models
│       ├── normalize.py      # Text normalization
│       ├── store.py          # Storage and indexing
│       ├── embedder.py       # Sentence embeddings
│       ├── detector.py       # Duplicate detection
│       ├── context_selector.py # Context selection and reranking
│       ├── generator.py      # Response generation
│       └── llm.py            # LLM interface (Gemini & MockLLM)
├── tests/                    # Unit and integration tests
└── eval/                     # Evaluation benchmarks
```

## Setup & Installation

1. Create a Python 3.11 virtual environment and activate it:
   ```bash
   uv venv .venv --python 3.11
   source .venv/bin/activate
   ```
2. Install dependencies:
   ```bash
   uv pip install -r requirements.txt
   ```
3. (Optional) Set up your Gemini API key in `.env`:
   ```bash
   cp .env.example .env
   # Edit .env and set GEMINI_API_KEY
   ```

## Running the Demo

The project includes a deterministic `MockLLM` so the entire pipeline can run without any API key:

```bash
python demo.py
```

## Running Tests

```bash
pytest
```
