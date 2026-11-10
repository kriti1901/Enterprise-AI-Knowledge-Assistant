"""Test configuration: offline embeddings, fake LLM, temporary storage. No API keys required."""

import os
import tempfile
from pathlib import Path

_tmp = Path(tempfile.mkdtemp(prefix="kb-test-"))
REPO_ROOT = Path(__file__).resolve().parents[2]

# CI sets DATABASE_URL to a PostgreSQL service; locally it defaults to a temporary SQLite file.
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp / 'app.db'}")

os.environ.update(
    {
        "CHROMA_PATH": str(_tmp / "chroma"),
        "UPLOAD_DIR": str(_tmp / "uploads"),
        "EMBEDDING_BACKEND": "hash",
        "LLM_PROVIDER": "fake",
        "JWT_SECRET": "test-secret-that-is-at-least-32-characters-long",
        "ADMIN_USERNAME": "admin",
        "ADMIN_PASSWORD": "admin-password-123",
        "EVAL_DIR": str(REPO_ROOT / "eval"),
        "REDIS_URL": "",
        "CLOUDWATCH_ENABLED": "false",
    }
)

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402


@pytest.fixture(scope="session")
def client():
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def admin_headers(client):
    token = client.post("/auth/login", data={"username": "admin", "password": "admin-password-123"}).json()
    return {"Authorization": f"Bearer {token['access_token']}"}


def login(client, username: str, password: str) -> dict:
    token = client.post("/auth/login", data={"username": username, "password": password}).json()
    return {"Authorization": f"Bearer {token['access_token']}"}
