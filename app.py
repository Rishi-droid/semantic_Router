import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from langchain_core.runnables import RunnableLambda

from evaluation import best_threshold, evaluate, sweep
from handlers import build_handlers, make_llm
from router import FALLBACK, SemanticRouter, make_embedder

BASE = Path(__file__).resolve().parent
st.set_page_config(page_title="Semantic Router", page_icon="🧭", layout="wide")


def secret(name: str) -> str:
    try:
        return str(st.secrets[name])
    except Exception:
        return os.environ.get(name, "")


def validate_routes(obj) -> str | None:
    if not isinstance(obj, dict) or not obj:
        return "Routes must be a JSON object: route name -> list of example strings."
    for name, utts in obj.items():
        if not isinstance(utts, list) or not utts or not all(isinstance(u, str) for u in utts):
            return f"Route '{name}' needs a non-empty list of strings."
        if name == FALLBACK:
            return f"'{FALLBACK}' is reserved; pick another route name."
    return None


@st.cache_resource(show_spinner="Loading embedding model (first run downloads ~90MB)...")
def get_embedder(kind: str, api_key: str):
    return make_embedder(kind, api_key or None)


@st.cache_resource(show_spinner="Embedding route examples...")
def get_router(routes_json: str, kind: str, api_key: str) -> SemanticRouter:
    return SemanticRouter(get_embedder(kind, api_key), json.loads(routes_json))


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
openai_key = secret("OPENAI_API_KEY")
anthropic_key = secret("ANTHROPIC_API_KEY")

with st.sidebar:
    st.header("Routing")
    embed_kind = st.selectbox(
        "Embeddings", ["local", "openai"], help="local = MiniLM on your machine; openai = API"
    )
    mode = st.radio("Scoring", SemanticRouter.MODES, horizontal=True,
                    help="max: closest example. centroid: mean of a route's examples.")
    threshold = st.slider("Confidence threshold", 0.0, 0.9, 0.45, 0.01)

    st.header("Handlers (LLM)")
    provider = st.selectbox("Provider", ["none", "openai", "anthropic"])
    default_model = {"openai": "gpt-4o-mini", "anthropic": "claude-haiku-5-5"}.get(provider, "")
    model = st.text_input("Model", value=default_model, key=f"model_{provider}",
                          disabled=provider == "none")
    if embed_kind == "openai" or provider == "openai":
        openai_key = st.text_input("OpenAI API key", type="password", value=openai_key)
    if provider == "anthropic":
        anthropic_key = st.text_input("Anthropic API key", type="password", value=anthropic_key)

    st.header("Routes")
    routes_text = st.text_area(
        "JSON: route name -> example utterances",
        value=(BASE / "data" / "routes.json").read_text(encoding="utf-8"),
        height=320,
    )

try:
    routes = json.loads(routes_text)
except json.JSONDecodeError as e:
    st.error(f"Invalid routes JSON: {e}")
    st.stop()
if (msg := validate_routes(routes)):
    st.error(msg)
    st.stop()

names = list(routes)
routes_json = json.dumps(routes)

if embed_kind == "openai" and not openai_key:
    st.warning("Enter an OpenAI API key in the sidebar to use OpenAI embeddings.")
    st.stop()

try:
    router = get_router(routes_json, embed_kind, openai_key if embed_kind == "openai" else "")
except Exception as e:
    st.error(f"Could not build the router: {e}")
    st.stop()

llm = None
if provider != "none":
    key = openai_key if provider == "openai" else anthropic_key
    if key:
        try:
            llm = make_llm(provider, key, model)
        except Exception as e:
            st.sidebar.error(f"Could not create the LLM: {e}")
    else:
        st.sidebar.warning("No API key set, so handlers return placeholder text.")
handlers = build_handlers(names, llm)


def build_chain():
    def decide(inp: dict) -> dict:
        return {**inp, "result": router.route(inp["query"], threshold, mode)}

    def dispatch(inp: dict) -> dict:
        res = inp["result"]
        answer = handlers[res.route].invoke({"query": inp["query"]})
        return {**inp, "answer": answer}

    return RunnableLambda(decide) | RunnableLambda(dispatch)


chain = build_chain()

st.title("🧭 Semantic Router")
chat_tab, eval_tab = st.tabs(["💬 Chat", "📊 Evaluate"])

# ---------------------------------------------------------------------------
# Chat tab
# ---------------------------------------------------------------------------
with chat_tab:
    if "history" not in st.session_state:
        st.session_state.history = []

    for item in st.session_state.history:
        with st.chat_message(item["role"]):
            st.write(item["content"])
            if item["role"] == "assistant":
                st.caption(f"Route: `{item['route']}` · routing took {item['ms']:.0f} ms")
                with st.expander("Similarity scores"):
                    st.bar_chart(pd.Series(item["scores"]))

    if query := st.chat_input("Ask something..."):
        try:
            with st.spinner("Routing..."):
                out = chain.invoke({"query": query})
        except Exception as e:
            st.error(f"Something went wrong: {e}")
        else:
            res = out["result"]
            st.session_state.history.append({"role": "user", "content": query})
            st.session_state.history.append(
                {"role": "assistant", "content": out["answer"], "route": res.route,
                 "scores": res.scores, "ms": res.latency_ms}
            )
            st.rerun()

# ---------------------------------------------------------------------------
# Evaluation tab
# ---------------------------------------------------------------------------
with eval_tab:
    st.subheader("Routing accuracy on a labeled set")
    uploaded = st.file_uploader("Labeled queries: JSON list of {query, route} (optional)", type="json")
    try:
        data = json.load(uploaded) if uploaded else json.loads(
            (BASE / "data" / "eval_data.json").read_text(encoding="utf-8")
        )
    except json.JSONDecodeError as e:
        st.error(f"Invalid dataset JSON: {e}")
        st.stop()

    if not (isinstance(data, list) and data and all(
        isinstance(d, dict) and isinstance(d.get("query"), str) and isinstance(d.get("route"), str)
        for d in data
    )):
        st.error("Dataset must be a non-empty JSON list of {query, route} objects.")
        st.stop()

    unknown = sorted({d["route"] for d in data} - set(names) - {FALLBACK})
    if unknown:
        st.error(f"Dataset labels not present in the current routes: {unknown}")
        st.stop()

    queries = [d["query"] for d in data]
    y_true = [d["route"] for d in data]
    st.caption(
        f"{len(data)} labeled queries. Use the label `{FALLBACK}` for out-of-scope queries "
        "that should match no route."
    )

    sig = (embed_kind, routes_json, json.dumps(data, sort_keys=True))
    if st.button("Run evaluation", type="primary"):
        with st.spinner("Embedding queries..."):
            Q, lat = router.embed_timed(queries)
        st.session_state.eval_state = {"sig": sig, "Q": Q, "lat": lat}

    state = st.session_state.get("eval_state")
    if state and state["sig"] == sig:
        S = router.scores(state["Q"], mode)
        result = evaluate(S, queries, y_true, names, threshold)
        sw = sweep(S, y_true, names)
        best_t, best_acc = best_threshold(sw)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric(f"Accuracy @ {threshold:.2f}", f"{result['accuracy']:.1%}")
        c2.metric("Best accuracy", f"{best_acc:.1%}", f"threshold ~ {best_t:.2f}", delta_color="off")
        c3.metric("Mean latency", f"{state['lat'].mean():.0f} ms")
        c4.metric("p95 latency", f"{np.percentile(state['lat'], 95):.0f} ms")

        st.markdown("**Accuracy vs. threshold**")
        st.line_chart(sw.set_index("threshold"))
        st.markdown("**Per-route metrics**")
        st.dataframe(result["report"], hide_index=True, width="stretch")
        st.markdown("**Confusion matrix** (rows = actual, columns = predicted)")
        st.dataframe(result["confusion"], width="stretch")
        st.markdown(f"**Misrouted queries** ({len(result['errors'])})")
        st.dataframe(result["errors"], hide_index=True, width="stretch")
    elif state:
        st.info("Routes, embedder or dataset changed since the last run. Click Run evaluation again.")

    # -- Benchmark -----------------------------------------------------
    st.divider()
    st.subheader("Benchmark: embedders x scoring modes")
    bench_openai_key = openai_key or secret("OPENAI_API_KEY")
    if not bench_openai_key:
        st.caption("Only the local embedder will be benchmarked. Add an OpenAI key to compare both.")
    if st.button("Run benchmark"):
        rows = []
        kinds = ["local"] + (["openai"] if bench_openai_key else [])
        with st.spinner("Benchmarking (API embeddings make one call per query)..."):
            for kind in kinds:
                try:
                    r = get_router(routes_json, kind, bench_openai_key if kind == "openai" else "")
                    Qb, lat = r.embed_timed(queries)
                except Exception as e:
                    st.warning(f"{kind} embeddings failed: {e}")
                    continue
                for m in SemanticRouter.MODES:
                    t, a = best_threshold(sweep(r.scores(Qb, m), y_true, names))
                    rows.append({
                        "Embedder": "local MiniLM-L6" if kind == "local" else "OpenAI text-embedding-3-small",
                        "Scoring": m,
                        "Best accuracy (%)": round(a * 100, 1),
                        "Best threshold": round(t, 2),
                        "Mean latency (ms)": round(float(lat.mean()), 1),
                    })
        st.session_state.bench = pd.DataFrame(rows)

    if "bench" in st.session_state and not st.session_state.bench.empty:
        st.dataframe(st.session_state.bench, hide_index=True, width="stretch")
        st.caption(
            "Thresholds are tuned on this same dataset, so these accuracies are optimistic. "
            "For a number you can quote, tune on one split and report accuracy on a held-out split."
        )