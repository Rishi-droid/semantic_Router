"""Route handlers: one LangChain runnable per route."""
from __future__ import annotations

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from router import FALLBACK

SYSTEM_PROMPTS = {
    "small_talk": "You are a friendly assistant. Reply warmly in one or two short sentences.",
    "timetable": (
        "You are a scheduling assistant. You do not have calendar access in this demo, "
        "so say what data you would need and help the user phrase or plan the request."
    ),
    "coding_help": (
        "You are a patient coding tutor. Give a short explanation and a minimal, "
        "correct code example when useful."
    ),
    "course_qa": (
        "You are a tutor for university courses in machine learning, NLP, data science "
        "and cybersecurity. Explain concepts clearly with a short example."
    ),
}

FALLBACK_MESSAGE = (
    "No route matched confidently. Try rephrasing, or lower the confidence threshold."
)


def make_llm(provider: str, api_key: str, model: str):
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=api_key, temperature=0.3)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=api_key, temperature=0.3, max_tokens=1024)
    return None


def _stub(route: str) -> RunnableLambda:
    msg = f"[{route}] handler reached. Add an API key in the sidebar to get a real answer."
    return RunnableLambda(lambda _: msg)


def build_handlers(route_names: list[str], llm) -> dict:
    handlers = {FALLBACK: RunnableLambda(lambda _: FALLBACK_MESSAGE)}
    for name in route_names:
        if llm is None:
            handlers[name] = _stub(name)
            continue
        system = SYSTEM_PROMPTS.get(
            name, f"You are a helpful assistant specialised in: {name.replace('_', ' ')}."
        )
        prompt = ChatPromptTemplate.from_messages([("system", system), ("human", "{query}")])
        handlers[name] = prompt | llm | StrOutputParser()
    return handlers