# Prompt Clarifier / Prompt Coach MVP

FastAPI + chat-style web UI implementing:
- Clarifier stage (`/v1/clarify`): Prompt A (clarified intent) and Prompt B (Vision B).
- Execution stage (`/v1/execute`): user selects A, B, or both and runs selected prompt(s).
- Direct stage (`/v1/direct`): bypass A/B and ask GPT directly.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Open `.env` and paste your API key. Then switch backend to `openai`.

```env
PROMPT_COACH_BACKEND=openai
PROMPT_COACH_MODEL=gpt-4.1-nano
PROMPT_COACH_ANSWER_MODEL=gpt-4.1-mini
PROMPT_COACH_ENABLE_WEB_SEARCH=true
OPENAI_API_KEY=PASTE_YOUR_OPENAI_API_KEY_HERE
```

## Run

```bash
uvicorn app.main:app --reload
```

Open: `http://127.0.0.1:8000`

## API

### POST `/v1/clarify`

Request:

```json
{
  "message": "where should I go for sushi tonight?",
  "locale": "en-US",
  "context": "optional"
}
```

Returns latent goal, interpretations, assumptions, ambiguities, Prompt A, Prompt B, rationale, follow-ups, and confidence.

### POST `/v1/execute`

Request:

```json
{
  "original_message": "where should I go for sushi tonight?",
  "selected": ["A", "B"]
}
```

Returns one or two runs depending on selection.

### POST `/v1/direct`

Request:

```json
{
  "message": "where should I go for sushi tonight?"
}
```

Returns one direct answer without generating A/B prompts first.

## Notes

- If `PROMPT_COACH_BACKEND=openai` and the key is missing/invalid, the API returns a clear `400` error.
- If backend is `mock`, no external model is called.
- `.env` is local-only and ignored by git. Commit only `.env.example`.
- Live web retrieval is enabled in both `A/B` answer mode and `Direct` mode when `PROMPT_COACH_ENABLE_WEB_SEARCH=true`.

## Test

```bash
pytest
```
