from fastapi.testclient import TestClient

from earshot.api import create_app
from earshot.config import Settings


def client(token: str = "") -> TestClient:
    settings = Settings(_env_file=None, llm_api_key="", api_token=token, embedder="hashing", reranker="lexical")
    return TestClient(create_app(settings))


def test_health_reports_offline_mode():
    assert client().get("/health").json()["brain"] == "offline"


def test_turn_returns_a_grounded_reply():
    body = client().post("/v1/turns", json={"text": "Are you allergic to any medicines?"}).json()
    assert "penicillin" in body["suggestions"][0]["text"]
    assert body["hud"]


def test_empty_and_oversized_text_are_rejected():
    api = client()
    assert api.post("/v1/turns", json={"text": ""}).status_code == 422
    assert api.post("/v1/turns", json={"text": "x" * 501}).status_code == 422


def test_note_upload_needs_the_token_when_set():
    api = client("secret")
    note = [{"title": "T", "text": "My bank is Harbor."}]
    assert api.post("/v1/users/a/notes", json=note).status_code == 401
    headers = {"Authorization": "Bearer secret"}
    assert api.post("/v1/users/a/notes", json=note, headers=headers).status_code == 200


def test_private_user_needs_the_token_when_set():
    api = client("secret")
    response = api.post("/v1/turns", json={"user_id": "a", "text": "Which bank do you use?"})
    assert response.status_code == 401


def test_demo_notes_are_read_only():
    response = client().post("/v1/users/demo/notes", json=[{"title": "T", "text": "x"}])
    assert response.status_code == 403


def test_stream_reports_nodes_then_the_result():
    with client().websocket_connect("/v1/stream") as socket:
        socket.send_json({"text": "How are you today?"})
        events = []
        while not events or events[-1]["type"] != "result":
            events.append(socket.receive_json())
    assert [e["node"] for e in events[:-1]] == ["triage", "simplify", "finish"]
    assert events[-1]["result"]["intent"] == "general_question"


def test_stream_survives_bad_input():
    with client().websocket_connect("/v1/stream") as socket:
        socket.send_json({"text": ""})
        assert socket.receive_json()["type"] == "error"


def test_metrics_count_turns():
    api = client()
    api.post("/v1/turns", json={"text": "Hello there."})
    summary = api.get("/v1/metrics").json()
    assert summary["turns"] == 1
    assert "p95" in summary["latency_ms"]


def test_demo_page_is_served():
    assert "Earshot" in client().get("/").text
