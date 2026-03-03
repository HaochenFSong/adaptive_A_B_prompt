import os

from fastapi.testclient import TestClient

os.environ["PROMPT_COACH_BACKEND"] = "mock"

from app.main import app

client = TestClient(app)


def test_clarify_then_execute_a() -> None:
    clarify = client.post("/v1/clarify", json={"message": "Where should I go for sushi tonight?"})
    assert clarify.status_code == 200
    data = clarify.json()
    assert data["prompt_a_clarified"]

    execute = client.post(
        "/v1/execute",
        json={"original_message": "Where should I go for sushi tonight?", "selected": ["A"]},
    )
    assert execute.status_code == 200
    runs = execute.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["variant"] == "A"


def test_clarify_then_execute_b() -> None:
    clarify = client.post("/v1/clarify", json={"message": "How can I learn Python quickly?"})
    assert clarify.status_code == 200

    execute = client.post(
        "/v1/execute",
        json={"original_message": "How can I learn Python quickly?", "selected": ["B"]},
    )
    assert execute.status_code == 200
    runs = execute.json()["runs"]
    assert len(runs) == 1
    assert runs[0]["variant"] == "B"


def test_execute_both_variants_returns_two_runs() -> None:
    execute = client.post(
        "/v1/execute",
        json={"original_message": "I need to choose a CRM tool", "selected": ["A", "B"]},
    )
    assert execute.status_code == 200
    runs = execute.json()["runs"]
    assert len(runs) == 2
    assert {run["variant"] for run in runs} == {"A", "B"}


def test_execute_uses_prompt_overrides_without_clarify_roundtrip() -> None:
    execute = client.post(
        "/v1/execute",
        json={
            "original_message": "ignored because prompts are provided",
            "selected": ["A", "B"],
            "prompt_a_clarified": "Prompt override A",
            "prompt_b_vision": "Prompt override B",
        },
    )
    assert execute.status_code == 200
    runs = execute.json()["runs"]
    assert runs[0]["prompt"] == "Prompt override A"
    assert runs[1]["prompt"] == "Prompt override B"


def test_high_ambiguity_input_returns_followups() -> None:
    clarify = client.post("/v1/clarify", json={"message": "Give me the best thing"})
    assert clarify.status_code == 200
    data = clarify.json()
    assert len(data["follow_up_questions"]) >= 1
    assert len(data["follow_up_questions"]) <= 2


def test_clarify_rejects_empty_input() -> None:
    clarify = client.post("/v1/clarify", json={"message": "   "})
    assert clarify.status_code in {400, 422}


def test_direct_ask_endpoint() -> None:
    direct = client.post("/v1/direct", json={"message": "Just answer directly"})
    assert direct.status_code == 200
    payload = direct.json()
    assert payload["prompt"] == "Just answer directly"
    assert payload["answer"]
