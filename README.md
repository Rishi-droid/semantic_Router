# Semantic Router

Routes a user query to the right handler by meaning, not keywords.
Built with LangChain, sentence-transformers and Streamlit.

## How it works
1. Each route is defined by example sentences (`data/routes.json`).
2. All examples are embedded once with a local model (all-MiniLM-L6-v2) or OpenAI embeddings.
3. A query is embedded and compared to every example by cosine similarity.
4. A route's score is its closest example (`max`) or the average of its examples (`centroid`).
5. The top route wins. If its score is below the threshold, the query goes to `fallback`.
6. The chosen route's handler (a LangChain chain, optionally LLM-backed) produces the answer.

## Setup (Windows)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```
The first run downloads the embedding model (about 90MB).

## API keys (optional)
Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add keys,
or paste them in the app sidebar. Without a key, handlers return placeholder text
and routing still works.

## Project structure
- `router.py`: embeddings, scoring, threshold logic
- `handlers.py`: one LangChain handler per route
- `evaluation.py`: accuracy, per-route metrics, confusion matrix, threshold sweep
- `app.py`: Streamlit UI (Chat tab and Evaluate tab)
- `data/routes.json`: route names and example sentences
- `data/eval_data.json`: labeled test queries (use label `fallback` for out-of-scope)

## Evaluation
The Evaluate tab reports accuracy, latency, a confusion matrix, misrouted queries and
an accuracy-vs-threshold curve, and benchmarks embedders against scoring modes.
Tune the threshold on one split of the data and report accuracy on a held-out split.

## Results
Fill in after running the benchmark on your machine.