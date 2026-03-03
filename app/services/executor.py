from __future__ import annotations

import os
from typing import Any

from app.models import DirectAskResponse, ExecuteRequest, ExecuteResponse, ExecuteRun
from app.services.clarifier import ClarifierService


class ExecutorService:
    def __init__(self, clarifier_service: ClarifierService) -> None:
        self._clarifier_service = clarifier_service
        self._backend = os.getenv("PROMPT_COACH_BACKEND", "mock").lower()
        self._model = os.getenv("PROMPT_COACH_MODEL", "gpt-4.1-nano")
        self._api_key = os.getenv("OPENAI_API_KEY")
        self._client: Any | None = None

        if self._backend == "openai" and self._api_key:
            from openai import OpenAI

            self._client = OpenAI(api_key=self._api_key)

    def execute(self, request: ExecuteRequest) -> ExecuteResponse:
        prompts: dict[str, str] = {}
        if request.prompt_a_clarified:
            prompts["A"] = request.prompt_a_clarified
        if request.prompt_b_vision:
            prompts["B"] = request.prompt_b_vision

        missing_variants = [variant for variant in request.selected if variant not in prompts]
        if missing_variants:
            clarifier_result = self._clarifier_service.generate(request.original_message)
            prompts.setdefault("A", clarifier_result.prompt_a_clarified)
            prompts.setdefault("B", clarifier_result.prompt_b_vision)

        runs: list[ExecuteRun] = []
        for variant in request.selected:
            prompt = prompts[variant]
            answer = self._answer_prompt(prompt, variant)
            runs.append(ExecuteRun(variant=variant, prompt=prompt, answer=answer))

        return ExecuteResponse(runs=runs)

    def ask_direct(self, message: str) -> DirectAskResponse:
        answer = self._answer_prompt(message, "Direct")
        return DirectAskResponse(prompt=message, answer=answer)

    def _answer_prompt(self, prompt: str, variant: str) -> str:
        if self._backend == "openai":
            if not self._api_key:
                raise ValueError("OPENAI_API_KEY is required when PROMPT_COACH_BACKEND=openai")
            assert self._client is not None
            try:
                response = self._client.responses.create(
                    model=self._model,
                    input=prompt,
                    temperature=0.4,
                    max_output_tokens=500,
                )
                output = getattr(response, "output_text", None)
                if output and output.strip():
                    return output.strip()
                raise ValueError("OpenAI returned an empty output")
            except Exception as exc:  # pragma: no cover - network/runtime branch
                raise ValueError(f"OpenAI execution failed: {exc}") from exc

        return (
            f"[Mock execution {variant}] This is where the selected prompt is run against a chat model. "
            f"Prompt used: {prompt}"
        )
