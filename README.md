# Sophistication Evaluator (Internal) + Private Storage

This patch adds an internal (not user-facing) investor sophistication evaluator using **gpt-4.1**
and stores the results in a private SQL table `investor_sophistication`.

## Files added
- `app/agents/sophistication_evaluator.py`  (gpt-4.1 JSON-only evaluator)
- `app/storage/sophistication_store.py`     (SQLAlchemy table + insert helpers)
- `app/services/sophistication_eval.py`     (one function you call from AgentService)

## How to wire into your AgentService

### 1) Startup: ensure table exists
In `AgentService.startup()` after you set `self._engine`:

```python
from app.storage.sophistication_store import ensure_sophistication_table

await ensure_sophistication_table(self._engine)
```

### 2) After each user turn: evaluate + store (internal)
In `AgentService.chat()` after you run your main agent and before returning:

```python
from app.services.sophistication_eval import evaluate_and_store_sophistication

await evaluate_and_store_sophistication(
    engine=self._engine,
    session_id=session_id,
    session=session,
    presence={
        "has_portafolio_promedio": portafolio_promedio is not None,
        "has_portafolio_inversionista": portafolio_inversionista is not None,
        "has_club_deals_concepts": bool(club_deals_concepts),
        "has_club_deals_opinion": bool(club_deals_opinion),
    },
)
```

Notes:
- The evaluator call does **NOT** pass `session=` to Runner, and the Agent uses `store=False`.
  So the evaluation never appears in the chat history.
- The transcript builder skips big `CONTEXTO — ...` blocks to keep tokens low.
