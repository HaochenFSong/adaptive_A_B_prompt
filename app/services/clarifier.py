from __future__ import annotations

import json
import os
import re
from typing import Any

from app.models import ClarifyResponse

WORD_RE = re.compile(r"[a-z0-9]+")
AMBIGUOUS_TERMS = {
    "good",
    "best",
    "better",
    "nice",
    "soon",
    "fast",
    "quick",
    "cheap",
    "affordable",
    "improve",
    "help",
}
TIME_HINTS = {"today", "tonight", "tomorrow", "week", "month", "deadline"}
LOCATION_HINTS = {"near", "nearby", "around", "in", "at", "location", "city"}
SELECTION_QUESTION = "Do you want Answer A, Answer B, or both?"

CLARIFIER_SYSTEM_PROMPT = """You are a Prompt Clarifier / Prompt Coach.

Before answering any user question, do the following:

1. Generate Prompt A:
- Clarify what the user likely means.
- Make explicit any assumptions the model must infer.
- Identify ambiguities.
- State the likely underlying goal.
- Rewrite the question in a sharper, more precise form.

2. Generate Prompt B:
- Reframe the problem at a higher level.
- Surface deeper objectives, identity implications, or long-term optimization.
- Challenge the framing if appropriate.
- Offer a version that may better achieve the user's deeper goal.

Keep both A and B concise and clear.
Hard length limits:
- prompt_a_clarified: max 45 words
- prompt_b_vision: max 45 words
- why_a and why_b: one short sentence each

Then ask:
\"Do you want Answer A, Answer B, or both?\"

Do not answer the question until the user chooses.

If the user explicitly says \"skip clarification\" or \"just answer,\" then answer normally.

Return only valid JSON with exactly these keys:
- latent_goal: string
- model_interpretations: array of strings
- assumptions: array of strings (use [\"none\"] only if truly none)
- ambiguities: array of strings (use [\"none\"] only if truly none)
- prompt_a_clarified: string
- prompt_b_vision: string
- why_a: string
- why_b: string
- follow_up_questions: array of strings with maximum 2 items
- confidence: number between 0 and 1
"""


def _tokens(text: str) -> set[str]:
    return set(WORD_RE.findall(text.lower()))


def jaccard_similarity(first: str, second: str) -> float:
    tokens_a = _tokens(first)
    tokens_b = _tokens(second)
    if not tokens_a and not tokens_b:
        return 1.0
    union = tokens_a | tokens_b
    if not union:
        return 0.0
    return len(tokens_a & tokens_b) / len(union)


class ClarifierService:
    DISTINCTNESS_THRESHOLD = 0.72

    def __init__(self) -> None:
        self._backend = os.getenv("PROMPT_COACH_BACKEND", "mock").lower()
        self._model = os.getenv("PROMPT_COACH_MODEL", "gpt-4.1-nano")
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._client: Any | None = None

        if self._backend == "openai" and self._api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)

    def generate(self, message: str, locale: str = "en-US", context: str | None = None) -> ClarifyResponse:
        cleaned_message = message.strip()
        if not cleaned_message:
            raise ValueError("message cannot be empty")

        if self._backend == "openai":
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY is required when PROMPT_COACH_BACKEND=openai")
            try:
                response = self._generate_with_openai(cleaned_message, locale, context)
                self._quality_gate(response)
                return response
            except Exception as exc:  # pragma: no cover - network/runtime branch
                raise ValueError(f"OpenAI clarify failed: {exc}") from exc

        response = self._generate_with_rules(cleaned_message, context)
        self._quality_gate(response)
        return response

    def _generate_with_openai(self, message: str, locale: str, context: str | None) -> ClarifyResponse:
        assert self._client is not None
        context_block = context if context else "none"
        user_input = (
            f"Locale: {locale}\n"
            f"Context: {context_block}\n"
            f"User message: {message}\n"
            "Return JSON only."
        )

        result = self._client.responses.create(
            model=self._model,
            input=[
                {"role": "system", "content": CLARIFIER_SYSTEM_PROMPT},
                {"role": "user", "content": user_input},
            ],
            temperature=0.2,
            max_output_tokens=500,
        )

        text = getattr(result, "output_text", "")
        if not text:
            raise ValueError("empty output from OpenAI")

        parsed = self._parse_json(text)
        response = ClarifyResponse(**parsed)
        response.follow_up_questions = self._normalize_followups(response.follow_up_questions)
        response.assumptions = response.assumptions or ["none"]
        response.ambiguities = response.ambiguities or ["none"]
        return response

    def _parse_json(self, raw_text: str) -> dict[str, Any]:
        raw_text = raw_text.strip()
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            start = raw_text.find("{")
            end = raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise ValueError("invalid JSON output from model")
            return json.loads(raw_text[start : end + 1])

    def _generate_with_rules(self, message: str, context: str | None) -> ClarifyResponse:
        latent_goal = self._infer_latent_goal(message)
        model_interpretations = self._infer_model_interpretations(message, context)
        assumptions = self._infer_assumptions(message, context)
        ambiguities = self._infer_ambiguities(message)

        prompt_a = self._build_prompt_a(message, assumptions)
        prompt_b = self._build_prompt_b(message, latent_goal, force_shift=False)
        prompt_b = self._enforce_distinctness(
            prompt_a,
            prompt_b,
            lambda: self._build_prompt_b(message, latent_goal, force_shift=True),
        )

        if not self._vision_b_valid(prompt_a, prompt_b):
            prompt_b = self._build_prompt_b(message, latent_goal, force_shift=True)

        follow_up_questions = self._normalize_followups(self._follow_up_questions(ambiguities, message))
        confidence = self._confidence(ambiguities, assumptions)

        return ClarifyResponse(
            latent_goal=latent_goal,
            model_interpretations=model_interpretations,
            assumptions=assumptions,
            ambiguities=ambiguities,
            prompt_a_clarified=prompt_a,
            prompt_b_vision=prompt_b,
            why_a="A preserves the stated request while making required assumptions explicit.",
            why_b="B reframes the problem toward the deeper objective to improve decision quality.",
            follow_up_questions=follow_up_questions,
            confidence=confidence,
        )

    def _normalize_followups(self, questions: list[str]) -> list[str]:
        cleaned = [q.strip() for q in questions if q and q.strip()]
        if SELECTION_QUESTION not in cleaned:
            if len(cleaned) >= 2:
                cleaned = [cleaned[0], SELECTION_QUESTION]
            else:
                cleaned.append(SELECTION_QUESTION)
        return cleaned[:2]

    def _infer_latent_goal(self, message: str) -> str:
        lowered = message.lower()
        if any(keyword in lowered for keyword in ["where", "restaurant", "eat", "sushi", "food"]):
            return "Find a high-confidence recommendation that matches context and preferences."
        if any(keyword in lowered for keyword in ["learn", "understand", "explain", "study"]):
            return "Accelerate understanding while reducing confusion and wasted effort."
        if any(keyword in lowered for keyword in ["build", "create", "design", "prototype"]):
            return "Produce a practical output quickly while keeping quality and clarity."
        if any(keyword in lowered for keyword in ["decide", "choose", "pick", "compare"]):
            return "Make a better decision with explicit trade-offs and criteria."
        return "Achieve the intended outcome efficiently with clear constraints and success criteria."

    def _infer_model_interpretations(self, message: str, context: str | None) -> list[str]:
        interpretations: list[str] = [
            "Interpret the user's primary success criterion from limited wording.",
            "Infer missing constraints such as scope, depth, and acceptable format.",
        ]
        lowered = message.lower()
        if not any(hint in lowered for hint in TIME_HINTS):
            interpretations.append("Infer relevant timeframe because timing is not fully specified.")
        if not any(hint in lowered for hint in LOCATION_HINTS):
            interpretations.append("Infer context/location relevance because the prompt does not pin it down.")
        if context:
            interpretations.append("Reconcile explicit context with prompt wording when they differ.")
        return interpretations

    def _infer_assumptions(self, message: str, context: str | None) -> list[str]:
        assumptions: list[str] = []
        lowered = message.lower()

        if not any(hint in lowered for hint in TIME_HINTS):
            assumptions.append("Assuming the request is for a near-term decision unless stated otherwise.")
        if "budget" not in lowered and "price" not in lowered and "cheap" not in lowered:
            assumptions.append("Assuming no strict budget cap was provided.")
        if not context:
            assumptions.append("Assuming no additional prior context should be considered.")

        if not assumptions:
            return ["none"]
        return assumptions

    def _infer_ambiguities(self, message: str) -> list[str]:
        ambiguities: list[str] = []
        lowered = message.lower()
        tokens = _tokens(lowered)

        if any(term in tokens for term in AMBIGUOUS_TERMS):
            ambiguities.append("Quality target is vague (e.g., 'best', 'good', or 'better').")
        if "for" not in tokens and "because" not in tokens:
            ambiguities.append("Underlying objective is not explicitly stated.")
        if "near" not in tokens and "location" not in tokens and "city" not in tokens:
            ambiguities.append("Relevant location or context boundary is unspecified.")

        if not ambiguities:
            return ["none"]
        return ambiguities

    def _build_prompt_a(self, message: str, assumptions: list[str]) -> str:
        assumption_note = ""
        if assumptions and assumptions[0] != "none":
            assumption_note = " Start by listing assumptions explicitly before giving the final answer."
        return (
            f"Clarify and answer this request with explicit constraints: {message}."
            " Keep the response aligned to the user's likely intended goal."
            f"{assumption_note}"
        )

    def _build_prompt_b(self, message: str, latent_goal: str, force_shift: bool) -> str:
        if force_shift:
            return (
                "Reframe the user request into an outcome-first strategy question. "
                f"Original request: {message}. "
                f"Primary objective: {latent_goal} "
                "Compare 2-3 strategic paths, explain trade-offs, then recommend the best path."
            )
        return (
            "Interpret the deeper objective behind this request and solve for that objective directly. "
            f"Original request: {message}. "
            "Before answering, state what success looks like and optimize for that outcome."
        )

    def _enforce_distinctness(self, prompt_a: str, prompt_b: str, regenerate: callable) -> str:
        if jaccard_similarity(prompt_a, prompt_b) > self.DISTINCTNESS_THRESHOLD:
            return regenerate()
        return prompt_b

    def _vision_b_valid(self, prompt_a: str, prompt_b: str) -> bool:
        if not prompt_b.strip():
            return False
        if prompt_a.strip().lower() == prompt_b.strip().lower():
            return False
        return jaccard_similarity(prompt_a, prompt_b) <= self.DISTINCTNESS_THRESHOLD

    def _follow_up_questions(self, ambiguities: list[str], message: str) -> list[str]:
        if ambiguities == ["none"]:
            return []

        questions: list[str] = []
        joined = " ".join(ambiguities).lower()

        if "quality target" in joined:
            questions.append("What does a successful result look like for you: speed, quality, cost, or something else?")
        if "location" in joined:
            questions.append("What location or context boundary should the model optimize for?")
        if "objective" in joined and len(questions) < 2:
            questions.append("What is the deeper goal behind this request (e.g., save time, decide confidently, reduce cost)?")

        if not questions:
            questions.append(f"What specific outcome do you want from: '{message}'?")

        return questions[:2]

    def _confidence(self, ambiguities: list[str], assumptions: list[str]) -> float:
        score = 0.88
        if ambiguities != ["none"]:
            score -= 0.18
        if assumptions != ["none"]:
            score -= min(0.12, 0.03 * len(assumptions))
        return round(max(0.2, min(0.95, score)), 2)

    def _quality_gate(self, response: ClarifyResponse) -> None:
        if not response.latent_goal.strip():
            raise ValueError("quality gate failed: latent_goal must be non-empty")
        if len(response.assumptions) == 0:
            raise ValueError("quality gate failed: assumptions missing")
        if len(response.ambiguities) == 0:
            raise ValueError("quality gate failed: ambiguities missing")
        if not response.prompt_a_clarified.strip() or not response.prompt_b_vision.strip():
            raise ValueError("quality gate failed: prompts missing")
        if len(response.follow_up_questions) > 2:
            raise ValueError("quality gate failed: too many follow-up questions")
        if not self._vision_b_valid(response.prompt_a_clarified, response.prompt_b_vision):
            raise ValueError("quality gate failed: Vision B is not materially distinct")
