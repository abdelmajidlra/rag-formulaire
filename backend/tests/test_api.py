from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_healthcheck():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_query(monkeypatch):
    sample = {
        "answer": "Test",
        "confidence": 0.9,
        "forms": ["IMM0001"],
        "sources": ["/tmp/doc.pdf"],
        "meta": {"route": "SINGLE"},
    }
    monkeypatch.setattr("core.pipeline.answer_question", lambda *_args, **_kwargs: sample)

    resp = client.post("/query", json={"question": "Salut"})
    assert resp.status_code == 200
    assert resp.json()["answer"] == "Test"


def test_query_validation_error():
    resp = client.post("/query", json={"wrong": "field"})
    assert resp.status_code == 422
