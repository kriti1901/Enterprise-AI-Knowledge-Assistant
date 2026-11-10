"""End-to-end API tests: auth, department access control, citations, history, evaluation."""

from conftest import login

LEAVE_POLICY = (
    b"Full-time employees accrue 20 days of paid annual leave per calendar year. "
    b"Employees receive 10 paid sick days per year. Leave requests need 2 weeks notice."
)
FINANCE_POLICY = (
    b"Expense claims must be submitted within 30 days of purchase. "
    b"Hotels are capped at 180 USD per night."
)


def _upload(client, headers, name, content, department, title):
    return client.post(
        "/documents",
        headers=headers,
        data={"department": department, "title": title},
        files={"file": (name, content, "text/plain")},
    )


def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_protected_routes_require_token(client):
    assert client.get("/documents").status_code == 401
    assert client.post("/chat", json={"question": "hi"}).status_code == 401


def test_bad_password_is_rejected(client):
    response = client.post("/auth/login", data={"username": "admin", "password": "wrong"})
    assert response.status_code == 401


def test_admin_creates_users_and_regular_user_cannot_upload(client, admin_headers):
    created = client.post(
        "/users",
        headers=admin_headers,
        json={"username": "hr_user", "password": "hr-password-123", "departments": ["HR"]},
    )
    assert created.status_code == 201
    client.post(
        "/users",
        headers=admin_headers,
        json={"username": "fin_user", "password": "fin-password-123", "departments": ["Finance"]},
    )
    hr_headers = login(client, "hr_user", "hr-password-123")
    response = _upload(client, hr_headers, "a.txt", LEAVE_POLICY, "HR", "Leave Policy")
    assert response.status_code == 403


def test_upload_index_and_department_scoped_chat(client, admin_headers):
    assert _upload(client, admin_headers, "leave.txt", LEAVE_POLICY, "HR", "Leave Policy").status_code == 201
    assert _upload(client, admin_headers, "expense.txt", FINANCE_POLICY, "Finance", "Expense Policy").status_code == 201

    hr = login(client, "hr_user", "hr-password-123")
    fin = login(client, "fin_user", "fin-password-123")

    hr_answer = client.post("/chat", headers=hr, json={"question": "How many days of paid annual leave are there?"})
    assert hr_answer.status_code == 200
    body = hr_answer.json()
    assert body["citations"], "answer should carry citations"
    assert all(c["department"] == "HR" for c in body["citations"])
    assert "[1]" in body["answer"]

    # Finance user asking about HR content must never see HR sources.
    fin_answer = client.post("/chat", headers=fin, json={"question": "How many days of paid annual leave are there?"})
    fin_body = fin_answer.json()
    assert all(c["department"] == "Finance" for c in fin_body["citations"])
    assert all(c["title"] != "Leave Policy" for c in fin_body["citations"])


def test_user_without_department_gets_refusal_and_no_sources(client):
    outsider_headers = login(client, "admin", "admin-password-123")  # admin has no departments
    response = client.post("/chat", headers=outsider_headers, json={"question": "annual leave days"})
    body = response.json()
    assert body["citations"] == []
    assert "could not find" in body["answer"].lower()


def test_documents_list_is_filtered_by_department(client, admin_headers):
    hr = login(client, "hr_user", "hr-password-123")
    titles = {d["title"] for d in client.get("/documents", headers=hr).json()}
    assert titles == {"Leave Policy"}
    admin_titles = {d["title"] for d in client.get("/documents", headers=admin_headers).json()}
    assert {"Leave Policy", "Expense Policy"} <= admin_titles


def test_conversation_history_is_saved_and_private(client):
    hr = login(client, "hr_user", "hr-password-123")
    first = client.post("/chat", headers=hr, json={"question": "sick days per year"}).json()
    conversation_id = first["conversation_id"]
    client.post("/chat", headers=hr, json={"question": "and leave notice?", "conversation_id": conversation_id})

    listing = client.get("/conversations", headers=hr).json()
    assert any(c["id"] == conversation_id for c in listing)

    detail = client.get(f"/conversations/{conversation_id}", headers=hr).json()
    roles = [m["role"] for m in detail["messages"]]
    assert roles == ["user", "assistant", "user", "assistant"]
    assert detail["messages"][1]["citations"]

    fin = login(client, "fin_user", "fin-password-123")
    assert client.get(f"/conversations/{conversation_id}", headers=fin).status_code == 404


def test_conversation_can_be_deleted(client):
    hr = login(client, "hr_user", "hr-password-123")
    conversation_id = client.post("/chat", headers=hr, json={"question": "leave"}).json()["conversation_id"]
    assert client.delete(f"/conversations/{conversation_id}", headers=hr).status_code == 204
    assert client.get(f"/conversations/{conversation_id}", headers=hr).status_code == 404


def test_document_delete_removes_it_from_search(client, admin_headers):
    upload = _upload(client, admin_headers, "temp.txt", b"Temporary cafeteria menu policy.", "HR", "Temp Menu")
    doc_id = upload.json()["id"]
    assert client.delete(f"/documents/{doc_id}", headers=admin_headers).status_code == 204
    hr = login(client, "hr_user", "hr-password-123")
    body = client.post("/chat", headers=hr, json={"question": "cafeteria menu"}).json()
    assert all(c["title"] != "Temp Menu" for c in body["citations"])


def test_evaluation_endpoint_reports_metrics(client, admin_headers):
    from pathlib import Path

    from app.config import get_settings
    from app.db import new_session
    from app.evaluation import seed_sample_corpus

    session = new_session()
    seed_sample_corpus(session, Path(get_settings().eval_dir) / "sample_docs")
    session.close()

    response = client.post(
        "/eval/run?departments=HR&departments=Finance&departments=Engineering", headers=admin_headers
    )
    assert response.status_code == 200
    metrics = response.json()
    assert metrics["num_questions"] == 7
    assert 0.0 <= metrics["mrr"] <= 1.0
    assert metrics["hit_at_k"] >= 0.5
    assert metrics["answer_accuracy"] is not None

    runs = client.get("/eval/runs", headers=admin_headers).json()
    assert runs and runs[0]["metrics"]["num_questions"] == 7
