"""Streamlit front end for the Enterprise AI Knowledge Assistant."""

from __future__ import annotations

import os
from datetime import datetime

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 180

st.set_page_config(page_title="Knowledge Assistant", page_icon="📚", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1100px;}
      section[data-testid="stSidebar"] {background: #f7f7f9;}
      .source-card {border-left: 3px solid #4f46e5; padding: .4rem .8rem; margin: .4rem 0;
                    background: #fff; border-radius: 6px; font-size: .9rem;}
      .muted {color: #6b7280; font-size: .85rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------- API helpers ----------

def api(method: str, path: str, **kwargs):
    headers = kwargs.pop("headers", None) or {}
    token = st.session_state.get("token")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = requests.request(method, f"{API_URL}{path}", headers=headers, timeout=TIMEOUT, **kwargs)
    if response.status_code == 401 and token:
        st.session_state.clear()
        st.rerun()
    return response


def fmt_date(value: str) -> str:
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).strftime("%d %b %Y, %H:%M")
    except (ValueError, AttributeError):
        return value or ""


# ---------- login ----------

def login_view() -> None:
    _, centre, _ = st.columns([1, 1.2, 1])
    with centre:
        st.title("📚 Knowledge Assistant")
        st.caption("Ask questions about your department's documents. Answers cite their sources.")
        with st.form("login"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", type="primary", use_container_width=True)
        if submitted:
            response = requests.post(
                f"{API_URL}/auth/login",
                data={"username": username, "password": password},
                timeout=TIMEOUT,
            )
            if response.status_code == 200:
                st.session_state.update(response.json())
                st.session_state.active_conversation = None
                st.rerun()
            else:
                st.error("Incorrect username or password.")


# ---------- sidebar ----------

def sidebar() -> str:
    with st.sidebar:
        st.markdown(f"**{st.session_state.username}**")
        departments = st.session_state.get("departments") or []
        st.caption(f"Role: {st.session_state.role} · Departments: {', '.join(departments) or 'none'}")

        pages = ["Chat"]
        if st.session_state.role == "admin":
            pages += ["Documents", "Evaluation"]
        page = st.radio("Go to", pages, label_visibility="collapsed")

        if page == "Chat":
            st.divider()
            if st.button("＋ New chat", use_container_width=True):
                st.session_state.active_conversation = None
                st.rerun()
            st.markdown("**History**")
            conversations = api("GET", "/conversations").json()
            if not conversations:
                st.caption("Your conversations will appear here.")
            for conv in conversations:
                label = conv["title"] if len(conv["title"]) < 40 else conv["title"][:37] + "…"
                if st.button(label, key=f"conv-{conv['id']}", use_container_width=True, help=fmt_date(conv["updated_at"])):
                    st.session_state.active_conversation = conv["id"]
                    st.rerun()
            st.divider()
            st.session_state.mode = st.selectbox(
                "Search mode",
                ["hybrid", "semantic"],
                index=0 if st.session_state.get("mode", "hybrid") == "hybrid" else 1,
                help="Hybrid combines meaning-based and keyword search.",
            )

        st.divider()
        if st.button("Sign out", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    return page


# ---------- chat ----------

def render_sources(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for c in citations:
            st.markdown(
                f"<div class='source-card'><b>[{c['n']}] {c['title']}</b> · "
                f"{c['department']} · page {c['page']}<br><span class='muted'>{c['snippet']}</span></div>",
                unsafe_allow_html=True,
            )


def render_conversation(conversation_id: int | None) -> None:
    messages = []
    if conversation_id is not None:
        detail = api("GET", f"/conversations/{conversation_id}").json()
        messages = detail.get("messages", [])
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_sources(message.get("citations") or [])

    question = st.chat_input("Ask about your documents…")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching authorized documents…"):
                response = api(
                    "POST",
                    "/chat",
                    json={
                        "question": question,
                        "conversation_id": conversation_id,
                        "mode": st.session_state.get("mode", "hybrid"),
                    },
                )
            if response.status_code != 200:
                st.error(response.json().get("detail", "Something went wrong."))
                return
            body = response.json()
            st.markdown(body["answer"])
            render_sources(body["citations"])
            if body.get("cached"):
                st.caption("Answered from cache.")
        st.session_state.active_conversation = body["conversation_id"]
        st.rerun()


def chat_page() -> None:
    st.header("Ask a question")
    render_conversation(st.session_state.get("active_conversation"))


# ---------- admin pages ----------

def documents_page() -> None:
    st.header("Documents")
    with st.expander("Upload a document", expanded=True):
        with st.form("upload", clear_on_submit=True):
            file = st.file_uploader("PDF, DOCX or TXT", type=["pdf", "docx", "txt"])
            col1, col2 = st.columns(2)
            department = col1.text_input("Department", placeholder="e.g. HR")
            title = col2.text_input("Title (optional)")
            submitted = st.form_submit_button("Upload and index", type="primary")
        if submitted:
            if file is None or not department.strip():
                st.warning("Choose a file and enter a department.")
            else:
                response = api(
                    "POST",
                    "/documents",
                    files={"file": (file.name, file.getvalue())},
                    data={"department": department.strip(), "title": title.strip()},
                )
                if response.status_code == 201:
                    st.success(f"Indexed '{response.json()['title']}'.")
                else:
                    st.error(response.json().get("detail", "Upload failed."))

    docs = api("GET", "/documents").json()
    if not docs:
        st.info("No documents yet.")
        return
    for doc in docs:
        col1, col2, col3, col4 = st.columns([4, 2, 2, 1])
        col1.markdown(f"**{doc['title']}**  \n<span class='muted'>{doc['filename']}</span>", unsafe_allow_html=True)
        col2.markdown(f"{doc['department']}")
        status_label = {"indexed": "✅ indexed", "failed": "❌ failed"}.get(doc["status"], doc["status"])
        col3.markdown(f"{status_label} · {doc['chunk_count']} chunks")
        if col4.button("Delete", key=f"del-{doc['id']}"):
            api("DELETE", f"/documents/{doc['id']}")
            st.rerun()
        if doc.get("error"):
            st.caption(f"Error: {doc['error']}")

    with st.expander("Create a user"):
        with st.form("create-user", clear_on_submit=True):
            username = st.text_input("Username")
            password = st.text_input("Temporary password", type="password")
            role = st.selectbox("Role", ["user", "admin"])
            departments = st.text_input("Departments (comma-separated)", placeholder="HR, Finance")
            if st.form_submit_button("Create user"):
                response = api(
                    "POST",
                    "/users",
                    json={
                        "username": username,
                        "password": password,
                        "role": role,
                        "departments": [d.strip() for d in departments.split(",") if d.strip()],
                    },
                )
                if response.status_code == 201:
                    st.success(f"Created {username}.")
                else:
                    st.error(response.json().get("detail", "Could not create user."))


def evaluation_page() -> None:
    st.header("Evaluation")
    st.caption("Runs the golden question set against the current index and reports retrieval and answer quality.")
    departments = st.text_input("Evaluate as departments (comma-separated)", value=", ".join(st.session_state.departments or []))
    if st.button("Run evaluation", type="primary"):
        params = [("departments", d.strip()) for d in departments.split(",") if d.strip()]
        with st.spinner("Running evaluation…"):
            response = api("POST", "/eval/run", params=params)
        if response.status_code != 200:
            st.error(response.json().get("detail", "Evaluation failed."))
            return
        metrics = response.json()
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Hit@{metrics['k']}", f"{metrics['hit_at_k']:.0%}")
        col2.metric("MRR", f"{metrics['mrr']:.2f}")
        col3.metric("Answer accuracy", f"{metrics['answer_accuracy']:.0%}" if metrics["answer_accuracy"] is not None else "—")
        col4.metric("Avg latency", f"{metrics['avg_latency_ms']:.0f} ms")
        st.dataframe(metrics["details"], use_container_width=True)


# ---------- main ----------

def main() -> None:
    for key, default in {"token": None, "active_conversation": None, "mode": "hybrid"}.items():
        st.session_state.setdefault(key, default)
    if not st.session_state.token:
        login_view()
        return
    page = sidebar()
    if page == "Chat":
        chat_page()
    elif page == "Documents":
        documents_page()
    elif page == "Evaluation":
        evaluation_page()


main()
