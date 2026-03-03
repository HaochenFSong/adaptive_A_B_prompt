from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class ClarifyRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    locale: str = Field(default="en-US", min_length=2, max_length=20)
    context: str | None = Field(default=None, max_length=2000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty")
        return stripped


class ClarifyResponse(BaseModel):
    latent_goal: str
    model_interpretations: list[str]
    assumptions: list[str]
    ambiguities: list[str]
    prompt_a_clarified: str
    prompt_b_vision: str
    why_a: str
    why_b: str
    follow_up_questions: list[str]
    confidence: float = Field(ge=0.0, le=1.0)


class ExecuteRequest(BaseModel):
    original_message: str = Field(min_length=1, max_length=2000)
    selected: list[Literal["A", "B"]]
    prompt_a_clarified: str | None = Field(default=None, max_length=4000)
    prompt_b_vision: str | None = Field(default=None, max_length=4000)

    @field_validator("original_message")
    @classmethod
    def validate_original_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("original_message cannot be empty")
        return stripped

    @model_validator(mode="after")
    def validate_selected(self) -> "ExecuteRequest":
        unique = list(dict.fromkeys(self.selected))
        if len(unique) == 0 or len(unique) > 2:
            raise ValueError("selected must include one or two variants")
        if len(unique) != len(self.selected):
            self.selected = unique
        if set(self.selected) not in ({"A"}, {"B"}, {"A", "B"}):
            raise ValueError("selected must be ['A'], ['B'], or ['A', 'B']")
        return self


class ExecuteRun(BaseModel):
    variant: Literal["A", "B"]
    prompt: str
    answer: str


class ExecuteResponse(BaseModel):
    runs: list[ExecuteRun]


class DirectAskRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("message cannot be empty")
        return stripped


class DirectAskResponse(BaseModel):
    prompt: str
    answer: str
