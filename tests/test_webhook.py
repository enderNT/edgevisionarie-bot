from fastapi.testclient import TestClient

from app.main import create_app


def test_healthcheck():
    client = TestClient(create_app())
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_chatwoot_webhook_accepts_immediately():
    client = TestClient(create_app())
    payload = {
        "content": "Necesito una cita",
        "conversation": {"id": 321},
        "contact": {"id": 654, "name": "Maria"},
    }

    response = client.post("/webhooks/chatwoot", json=payload)

    assert response.status_code == 202
    assert response.json() == {"status": "accepted", "conversation_id": "321"}
