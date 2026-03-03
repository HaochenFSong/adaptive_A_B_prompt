import os

os.environ["PROMPT_COACH_BACKEND"] = "mock"

from app.services.clarifier import ClarifierService, jaccard_similarity


def test_empty_input_rejected() -> None:
    service = ClarifierService()
    try:
        service.generate("   ")
        assert False, "Expected ValueError for empty message"
    except ValueError:
        assert True


def test_clarifier_returns_required_fields() -> None:
    service = ClarifierService()
    result = service.generate("Where should I go for sushi tonight?")

    assert result.latent_goal
    assert result.prompt_a_clarified
    assert result.prompt_b_vision
    assert isinstance(result.model_interpretations, list)
    assert isinstance(result.assumptions, list)
    assert isinstance(result.ambiguities, list)


def test_distinctness_guard_triggers_regeneration() -> None:
    service = ClarifierService()

    prompt_a = "Answer this request with explicit constraints and assumptions."
    prompt_b = "Answer this request with explicit constraints and assumptions."

    regenerated = service._enforce_distinctness(  # noqa: SLF001
        prompt_a,
        prompt_b,
        lambda: "Reframe this into an outcome-first strategy with trade-offs.",
    )

    assert regenerated != prompt_b
    assert jaccard_similarity(prompt_a, regenerated) < service.DISTINCTNESS_THRESHOLD


def test_vision_b_exists_for_every_response() -> None:
    service = ClarifierService()
    result = service.generate("Help me write a product launch post")
    assert result.prompt_b_vision.strip() != ""


def test_follow_up_question_limit() -> None:
    service = ClarifierService()
    result = service.generate("Give me the best option")
    assert len(result.follow_up_questions) <= 2
