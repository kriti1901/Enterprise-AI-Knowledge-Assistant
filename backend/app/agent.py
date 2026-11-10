"""LangGraph agent that answers questions by calling tools over the user's authorized documents.

Graph:  decide --search--> search --> decide
                 --list_documents--> list_documents --> decide
                 --answer--> answer --> END

Tools only ever see the caller's departments, so the agent cannot reach documents the user
is not authorized for, regardless of what the model asks for.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db import Document
from app.llm import ChatModel
from app.retrieval import Hit, search

MAX_STEPS = 3
HISTORY_TURNS = 4

DECIDE_SYSTEM = (
    "You are a planning step for an enterprise document assistant. Decide the next action. "
    "Respond ONLY with JSON, one of:\n"
    '{"action": "search", "query": "<search query>"} to search the authorized documents,\n'
    '{"action": "list_documents"} to list the documents the user can access,\n'
    '{"action": "answer"} when the retrieved results are enough to answer.'
)

ANSWER_SYSTEM = (
    "You answer questions using ONLY the numbered sources provided. Cite every factual claim "
    "with its source number in square brackets, for example [2]. If the sources do not contain "
    "the answer, say that you could not find it in the documents you have access to. "
    "Be concise."
)

REFUSAL = "I could not find this in the documents you have access to."


class AgentState(TypedDict):
    question: str
    history: list[dict[str, str]]
    query: str
    action: str
    hits: list[Hit]
    listing: list[str]
    steps: int
    tool_calls: list[str]
    answer: str


@dataclass
class AgentResult:
    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    sources: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


def _parse_action(raw: str, question: str) -> tuple[str, str]:
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    try:
        data = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        data = {}
    action = data.get("action")
    if action not in {"search", "list_documents", "answer"}:
        action = "answer"
    query = str(data.get("query") or question).strip() or question
    return action, query


def _merge_hits(existing: list[Hit], new: list[Hit]) -> list[Hit]:
    seen = {hit.key for hit in existing}
    return existing + [hit for hit in new if hit.key not in seen]


def _format_history(history: list[dict[str, str]]) -> str:
    recent = history[-HISTORY_TURNS * 2 :]
    return "\n".join(f"{m['role']}: {m['content']}" for m in recent) or "(none)"


def build_graph(session: Session, departments: list[str], llm: ChatModel, mode: str):
    def decide(state: AgentState) -> dict[str, Any]:
        if state["steps"] >= MAX_STEPS:
            return {"action": "answer", "query": state["question"]}
        prompt = (
            f"Question: {state['question']}\n"
            f"Tool results so far: {len(state['hits'])}\n"
            "Available actions: search, list_documents, answer"
        )
        raw = llm.complete(
            [{"role": "system", "content": DECIDE_SYSTEM}, {"role": "user", "content": prompt}]
        )
        action, query = _parse_action(raw, state["question"])
        return {"action": action, "query": query}

    def search_node(state: AgentState) -> dict[str, Any]:
        new_hits = search(state["query"], departments, session, mode=mode)
        return {
            "hits": _merge_hits(state["hits"], new_hits),
            "steps": state["steps"] + 1,
            "tool_calls": state["tool_calls"] + [f"search({state['query']!r})"],
        }

    def list_node(state: AgentState) -> dict[str, Any]:
        titles = session.scalars(
            select(Document.title)
            .where(Document.department.in_(departments), Document.status == "indexed")
            .order_by(Document.title)
        ).all()
        return {
            "listing": list(titles),
            "steps": state["steps"] + 1,
            "tool_calls": state["tool_calls"] + ["list_documents()"],
        }

    def answer_node(state: AgentState) -> dict[str, Any]:
        if not state["hits"] and not state["listing"]:
            return {"answer": REFUSAL}
        sources = "\n".join(
            f"[{i}] {hit.text.replace(chr(10), ' ')} (Source: {hit.title}, page {hit.page})"
            for i, hit in enumerate(state["hits"], start=1)
        )
        listing = "\n".join(f"- {t}" for t in state["listing"]) or "(none)"
        prompt = (
            f"Conversation so far:\n{_format_history(state['history'])}\n\n"
            f"Documents you can access:\n{listing}\n\n"
            f"Sources:\n{sources or '(none)'}\n\n"
            f"Question: {state['question']}"
        )
        answer = llm.complete(
            [{"role": "system", "content": ANSWER_SYSTEM}, {"role": "user", "content": prompt}]
        ).strip()
        return {"answer": answer or REFUSAL}

    graph = StateGraph(AgentState)
    graph.add_node("decide", decide)
    graph.add_node("search", search_node)
    graph.add_node("list_documents", list_node)
    graph.add_node("answer", answer_node)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide",
        lambda state: state["action"],
        {"search": "search", "list_documents": "list_documents", "answer": "answer"},
    )
    graph.add_edge("search", "decide")
    graph.add_edge("list_documents", "decide")
    graph.add_edge("answer", END)
    return graph.compile()


def run_agent(
    question: str,
    departments: list[str],
    session: Session,
    llm: ChatModel,
    history: list[dict[str, str]] | None = None,
    mode: str = "hybrid",
) -> AgentResult:
    graph = build_graph(session, departments, llm, mode)
    final = graph.invoke(
        {
            "question": question,
            "history": history or [],
            "query": question,
            "action": "",
            "hits": [],
            "listing": [],
            "steps": 0,
            "tool_calls": [],
            "answer": "",
        }
    )
    hits: list[Hit] = final["hits"]
    answer = final["answer"]
    cited_numbers = {
        int(n) for n in re.findall(r"\[(\d+)\]", answer) if 1 <= int(n) <= len(hits)
    }
    sources = [_hit_to_source(i, hit) for i, hit in enumerate(hits, start=1)]
    citations = [src for src in sources if src["n"] in cited_numbers]
    return AgentResult(
        answer=answer,
        citations=citations,
        sources=sources,
        tool_calls=final["tool_calls"],
    )


def _hit_to_source(n: int, hit: Hit) -> dict[str, Any]:
    return {
        "n": n,
        "document_id": hit.document_id,
        "title": hit.title,
        "department": hit.department,
        "page": hit.page,
        "score": round(hit.score, 4),
        "snippet": hit.text[:300],
    }
