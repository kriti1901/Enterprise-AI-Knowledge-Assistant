"""FastAPI application: auth, documents, chat with citations, conversation history, evaluation."""

from __future__ import annotations

import datetime as dt
import time
from contextlib import asynccontextmanager
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, UploadFile, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.agent import run_agent
from app.cache import get_cached_answer, set_cached_answer
from app.config import get_settings
from app.db import Conversation, Document, EvalRun, Message, User, get_session, init_db, new_session
from app.ingestion import UnsupportedFileError
from app.llm import get_llm
from app.observability import configure_logging, emit_metric, log_event
from app.security import create_access_token, decode_access_token, hash_password, verify_password
from app.services import delete_document, ingest_document

bearer = HTTPBearer(auto_error=False)
DbSession = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    init_db()
    settings = get_settings()
    with new_session() as session:
        if settings.admin_username and settings.admin_password:
            exists = session.scalar(select(User).where(User.username == settings.admin_username))
            if not exists:
                session.add(
                    User(
                        username=settings.admin_username,
                        password_hash=hash_password(settings.admin_password),
                        role="admin",
                        departments=[],
                    )
                )
                session.commit()
    yield


app = FastAPI(title="Enterprise AI Knowledge Assistant", version="1.0.0", lifespan=lifespan)


# ---------- schemas ----------

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str
    departments: list[str]


class UserOut(BaseModel):
    id: int
    username: str
    role: str
    departments: list[str]


class UserCreate(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=8, max_length=200)
    role: Literal["admin", "user"] = "user"
    departments: list[str] = Field(default_factory=list)


class DocumentOut(BaseModel):
    id: int
    title: str
    filename: str
    department: str
    status: str
    chunk_count: int
    error: str | None
    created_at: dt.datetime


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: int | None = None
    mode: Literal["hybrid", "semantic"] = "hybrid"


class CitationOut(BaseModel):
    n: int
    document_id: int
    title: str
    department: str
    page: int
    score: float
    snippet: str


class ChatResponse(BaseModel):
    conversation_id: int
    answer: str
    citations: list[CitationOut]
    tool_calls: list[str]
    latency_ms: int
    cached: bool = False


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    citations: list | None
    created_at: dt.datetime


class ConversationOut(BaseModel):
    id: int
    title: str
    updated_at: dt.datetime


class ConversationDetail(ConversationOut):
    messages: list[MessageOut]


# ---------- auth ----------

def get_current_user(
    creds: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    session: DbSession,
) -> User:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Not authenticated")
    username = decode_access_token(creds.credentials)
    user = session.scalar(select(User).where(User.username == username)) if username else None
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token")
    return user  # departments are read from the database on every request


def require_admin(user: Annotated[User, Depends(get_current_user)]) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin role required")
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]
AdminUser = Annotated[User, Depends(require_admin)]


@app.post("/auth/login", response_model=Token)
def login(form: Annotated[OAuth2PasswordRequestForm, Depends()], session: DbSession):
    user = session.scalar(select(User).where(User.username == form.username))
    if user is None or not verify_password(form.password, user.password_hash):
        log_event("login_failed", username=form.username)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect username or password")
    log_event("login", username=user.username)
    return Token(
        access_token=create_access_token(user.username),
        username=user.username,
        role=user.role,
        departments=list(user.departments or []),
    )


@app.get("/auth/me", response_model=UserOut)
def me(user: CurrentUser):
    return UserOut(id=user.id, username=user.username, role=user.role, departments=user.departments or [])


@app.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, _: AdminUser, session: DbSession):
    if session.scalar(select(User).where(User.username == body.username)):
        raise HTTPException(status.HTTP_409_CONFLICT, "Username already exists")
    user = User(
        username=body.username,
        password_hash=hash_password(body.password),
        role=body.role,
        departments=sorted({d.strip() for d in body.departments if d.strip()}),
    )
    session.add(user)
    session.commit()
    session.refresh(user)
    return UserOut(id=user.id, username=user.username, role=user.role, departments=user.departments)


# ---------- documents ----------

@app.get("/documents", response_model=list[DocumentOut])
def list_documents(user: CurrentUser, session: DbSession):
    query = select(Document).order_by(Document.created_at.desc())
    if user.role != "admin":
        query = query.where(Document.department.in_(user.departments or []))
    return [DocumentOut.model_validate(d, from_attributes=True) for d in session.scalars(query)]


@app.post("/documents", response_model=DocumentOut, status_code=201)
async def upload_document(
    user: AdminUser,
    session: DbSession,
    file: Annotated[UploadFile, File()],
    department: Annotated[str, Form(min_length=1, max_length=100)],
    title: Annotated[str, Form(max_length=300)] = "",
):
    content = await file.read()
    try:
        document = ingest_document(
            session,
            filename=file.filename or "upload",
            content=content,
            title=title,
            department=department,
            uploaded_by=user.username,
        )
    except UnsupportedFileError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    return DocumentOut.model_validate(document, from_attributes=True)


@app.delete("/documents/{document_id}", status_code=204)
def remove_document(document_id: int, _: AdminUser, session: DbSession):
    document = session.get(Document, document_id)
    if document is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
    delete_document(session, document)
    return None


# ---------- chat ----------

@app.get("/conversations", response_model=list[ConversationOut])
def list_conversations(user: CurrentUser, session: DbSession):
    rows = session.scalars(
        select(Conversation)
        .where(Conversation.user_id == user.id)
        .order_by(Conversation.updated_at.desc())
    ).all()
    return [ConversationOut(id=c.id, title=c.title, updated_at=c.updated_at) for c in rows]


def _owned_conversation(session: Session, user: User, conversation_id: int) -> Conversation:
    conversation = session.get(Conversation, conversation_id)
    if conversation is None or conversation.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Conversation not found")
    return conversation


@app.get("/conversations/{conversation_id}", response_model=ConversationDetail)
def get_conversation(conversation_id: int, user: CurrentUser, session: DbSession):
    c = _owned_conversation(session, user, conversation_id)
    return ConversationDetail(
        id=c.id,
        title=c.title,
        updated_at=c.updated_at,
        messages=[
            MessageOut(id=m.id, role=m.role, content=m.content, citations=m.citations, created_at=m.created_at)
            for m in c.messages
        ],
    )


@app.delete("/conversations/{conversation_id}", status_code=204)
def delete_conversation(conversation_id: int, user: CurrentUser, session: DbSession):
    session.delete(_owned_conversation(session, user, conversation_id))
    session.commit()
    return None


@app.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest, user: CurrentUser, session: DbSession):
    started = time.perf_counter()
    departments = list(user.departments or [])

    if body.conversation_id is None:
        conversation = Conversation(user_id=user.id, title=body.question[:60])
        session.add(conversation)
        session.flush()
    else:
        conversation = _owned_conversation(session, user, body.conversation_id)

    history = [{"role": m.role, "content": m.content} for m in conversation.messages]
    cached_payload = None if history else get_cached_answer(departments, body.question, body.mode)

    if cached_payload is not None:
        result = cached_payload
        cached = True
    else:
        agent_result = run_agent(body.question, departments, session, get_llm(), history, body.mode)
        result = {
            "answer": agent_result.answer,
            "citations": agent_result.citations,
            "tool_calls": agent_result.tool_calls,
        }
        cached = False
        if not history:
            set_cached_answer(departments, body.question, body.mode, result)

    session.add(Message(conversation_id=conversation.id, role="user", content=body.question))
    session.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content=result["answer"],
            citations=result["citations"],
        )
    )
    conversation.updated_at = dt.datetime.now(dt.UTC)
    session.commit()

    latency_ms = int((time.perf_counter() - started) * 1000)
    emit_metric("QueryLatencyMs", latency_ms, unit="Milliseconds")
    emit_metric("RetrievalHitCount", len(result["citations"]))
    log_event(
        "chat",
        user=user.username,
        conversation_id=conversation.id,
        citations=len(result["citations"]),
        tool_calls=result["tool_calls"],
        latency_ms=latency_ms,
        cached=cached,
    )
    return ChatResponse(
        conversation_id=conversation.id,
        answer=result["answer"],
        citations=[CitationOut(**c) for c in result["citations"]],
        tool_calls=result["tool_calls"],
        latency_ms=latency_ms,
        cached=cached,
    )


# ---------- evaluation ----------

@app.post("/eval/run")
def run_evaluation(
    user: AdminUser,
    session: DbSession,
    departments: Annotated[list[str] | None, Query()] = None,
):
    from app.evaluation import run_eval

    scope = departments or list(user.departments or [])
    metrics = run_eval(session, departments=scope)
    session.add(EvalRun(metrics=metrics))
    session.commit()
    return metrics


@app.get("/eval/runs")
def list_eval_runs(_: AdminUser, session: DbSession):
    runs = session.scalars(select(EvalRun).order_by(EvalRun.id.desc()).limit(10)).all()
    return [{"id": r.id, "created_at": r.created_at, "metrics": r.metrics} for r in runs]


@app.get("/health")
def health():
    return {"status": "ok"}
