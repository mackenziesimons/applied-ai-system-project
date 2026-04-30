# PawPal+ Model Card

This document describes the AI components built into PawPal+, records the AI collaboration process used during development, addresses potential biases, and summarizes testing results. It fulfills the reflection requirements for AI 110.

---

## 1. System Design

### Initial design

My initial UML design used four classes that separate data modeling, scheduling logic, and app orchestration. `Pet` stores profile information (name, species, age, preferences). `CareTask` represents individual care actions with a due time, priority, and completion status. `Scheduler` handles prioritization, today's plan, and conflict resolution. `PawPalApp` acts as the coordinator — managing collections of pets and tasks and exposing user-facing actions.

### Design changes after AI feedback

Three concrete changes came out of AI-assisted design review:

1. **Explicit pet-task relationship validation** — added app-level checks so a task's `pet_id` must match an existing pet before it is scheduled, preventing orphaned tasks.
2. **Fast ID-based lookups** — identified a list-scan bottleneck and planned pet/task maps for O(1) access while keeping ordered lists for display.
3. **Clearer responsibility split** — `PawPalApp` owns storage and user actions; `Scheduler` owns only prioritization, plan-building, and conflict detection.

### Core user actions

1. Create and manage pet profiles (name, species, age, preferences).
2. Schedule and track daily care tasks (walks, feedings, medication, appointments).
3. Open a "Today" view showing the ordered daily plan with AI care advice.

---

## 2. AI System Components

### Base model

**OpenAI `gpt-4o-mini`** — used for natural-language care advice generation. The model is called with a structured RAG prompt that includes retrieved knowledge-base facts and a few-shot example. It is never called for scheduling logic, which remains fully deterministic.

### Retrieval component

A tag-intersection scorer over two local JSON knowledge sources (`pet_care_facts.json` and `owner_notes.json`). No external model is used for retrieval — scoring is purely deterministic (intersection size / total tags). This means retrieval quality is fully transparent and testable without an API key.

### Confidence scoring

`compute_confidence_score()` returns the average ratio of matched tags to total tags across the top-k retrieved facts. Score is shown in the UI as a green (≥60%), yellow (30–59%), or red (<30%) badge so the owner always knows how grounded the advice is before acting on it.

### Agentic chain

`run_agent_chain()` executes 7 named steps sequentially: profile analysis → task context → gap detection → fact retrieval → confidence scoring → advice generation → self-evaluation. Each step records its input, output, and any structured data so intermediate decisions are fully observable in the UI expander and in `pawpal_advisor.log`.

---

## 3. AI Collaboration

### How I used AI tools

I used GitHub Copilot and GPT-4o throughout the project for:

- **Design brainstorming** — reviewing my initial UML class diagram and identifying the three refinements described above.
- **Code generation** — generating the skeleton of `ai_advisor.py` (RAG retriever, prompt builder, confidence scorer) and iterating on it until the test suite passed.
- **Debugging** — diagnosing a date-dependency bug in `evaluate_schedule()` that caused tests to fail depending on the current date; the fix was making `current_time` an injectable parameter.
- **Documentation** — drafting the README sections and then editing them to match the actual implementation rather than what the AI assumed was implemented.

The most useful prompt pattern was: *"Here is my current function. Here is the test that fails. What is wrong and how do I fix it without changing the test?"* This kept the AI focused on verifying behavior against a specification rather than redesigning freely.

### Judgment calls — where I did not accept suggestions as-is

**Retrieval scoring:** The AI initially suggested using cosine similarity over TF-IDF vectors for retrieval. I rejected this because it would require a dependency (scikit-learn), add non-determinism to tests, and obscure the scoring logic. Tag-intersection scoring is simpler, fully deterministic, and still effective for a small curated knowledge base — a deliberate tradeoff I made consciously rather than defaulting to the AI's more complex suggestion.

**Confidence badge thresholds:** The AI proposed a single green/red threshold at 50%. I changed it to a three-tier system (≥60% green, 30–59% yellow, <30% red) after reviewing the actual score distribution on realistic test scenarios, where most scores landed between 0.30 and 0.55. A binary threshold would have shown nearly everything as red even for well-matched queries.

**Verification process:** After every code suggestion I ran `pytest tests/ -v` and `python run_eval.py` before accepting the change. Any suggestion that broke an existing test was revised or rejected. I also read every generated function before merging it to confirm it matched the intended design (e.g., that `retrieve_facts` was actually filtering by species, not just returning all facts).

---

## 4. Potential Biases and Limitations

### Knowledge base bias

`pet_care_facts.json` was written by hand based on common veterinary guidelines for dogs and cats only. The system has **no knowledge of exotic pets** (rabbits, birds, reptiles, fish). A user with a rabbit will receive no retrieved facts and a near-zero confidence score, and the LLM fallback advice may be generic or inaccurate. This is disclosed in the UI via the confidence badge and the "no relevant facts were retrieved" message.

### Species imbalance

The knowledge base contains 14 dog-specific facts and 8 cat-specific facts (plus 6 owner notes that are pet-agnostic). Dog owners will generally receive higher-confidence, more specific advice. Cat owners receive less coverage. This should be addressed by expanding the knowledge base before any real-world deployment.

### Recency bias in the LLM

`gpt-4o-mini` has a training cutoff and may reflect outdated veterinary guidance. All LLM output should be treated as general information, not professional veterinary advice. The UI includes a disclaimer to this effect.

### Owner notes are not validated

The `owner_notes.json` file is owner-editable with no schema enforcement. A user could add incorrect or contradictory notes that would surface as high-confidence retrieved facts. In a production system, owner notes would require validation against a schema and possibly a review step.

### Conflict detection is time-exact only

The scheduler flags conflicts only for tasks scheduled at the exact same minute. Overlapping tasks (e.g., a 30-minute walk and a 20-minute grooming starting 15 minutes apart) are not detected. This is a known limitation documented in the design tradeoffs section.

---

## 5. Testing and Verification

### What was tested

**17 pytest unit tests** (`tests/test_pawpal.py`):
- Core scheduler: `mark_complete`, `add_task`, `sort_by_time`, `filter_tasks`, `complete_task` (with recurrence), `detect_conflicts`
- AI advisor: `load_knowledge_base`, `retrieve_facts` (relevant + empty KB), `compute_confidence_score` (range, zero case, relevant vs. irrelevant), `evaluate_schedule` (feeding gap, exercise gap, clean schedule)

**17 eval harness checks** (`run_eval.py`):
- Source attribution: all 28 facts labeled with correct source
- 6 predefined scenarios covering: clean dog schedule, feeding-only gap, exercise-only gap, clean cat schedule, two-pet time conflict, medication with owner-note retrieval

No LLM is called by either test suite — all checks are deterministic.

### Results

```
pytest tests/ -v     →  17 passed in 0.03s
python run_eval.py   →  17/17 checks passed  |  Average retrieval confidence: 0.40
```

### Confidence in correctness

I am confident the **deterministic components** (scheduler, retriever, confidence scorer, schedule evaluator) are correct. The unit tests cover every meaningful branch and the eval harness proves the pipeline behaves correctly end-to-end on representative scenarios.

I am **less confident** about the LLM-generated advice quality, for two reasons:
1. Output quality varies with model updates and temperature.
2. The few-shot example constrains format but cannot guarantee factual accuracy.

Mitigation: the confidence badge and schedule observations give the owner independent signal about advice quality before they act on it.

### Edge cases to test next

- Pet with no tasks at all (empty task list)
- Task scheduled exactly at midnight (boundary for `occurs_on`)
- Owner with 10+ pets and 50+ tasks (performance)
- Owner notes file missing or malformed (graceful degradation)
- `OPENAI_API_KEY` set but invalid (error handling)

---

## 6. Scheduling Logic and Tradeoffs

### Constraints considered

The scheduler considers: **time** (tasks are sorted chronologically), **recurrence frequency** (daily vs. weekly determines when the next occurrence is generated), and **exact-time conflicts** (two tasks at the same minute across any pet belonging to the same owner).

Priority and estimated duration were considered in the original UML design but not implemented in the final system. This was a conscious scope decision — for a pet care planner, chronological order and conflict visibility are more useful to an owner than an abstract priority number.

### Key tradeoff

Conflict detection flags exact-time matches only, not overlapping windows. This keeps the detection logic simple and fully testable (the test `test_detect_conflicts_returns_warning_for_same_time_tasks` covers it completely), but it misses cases like a 30-minute walk and a 20-minute grooming that overlap. A time-range conflict detector would require duration data on every task, which was not in the original data model.

---

## 7. Reflection

### What went well

The decision to separate retrieval from generation paid off immediately in testing. Because `retrieve_facts()` is a pure function (no side effects, no API calls), I could write tests for it without mocking anything and catch a species-filtering bug early. The same applies to `compute_confidence_score()` and `evaluate_schedule()` — all three are independently testable, which gave me confidence in the pipeline before the LLM integration was complete.

### What I would improve

If I had another iteration I would:

1. **Expand the knowledge base** to cover exotic pets and increase cat-specific facts to reach parity with dog coverage.
2. **Add duration to `Task`** so the conflict detector can flag overlapping windows, not just exact-time matches.
3. **Cache the knowledge base** in memory rather than reading from disk on every `advise()` call — currently not a problem at 28 facts, but would matter at scale.
4. **Validate owner notes schema** with Pydantic or a JSON Schema file so malformed entries fail loudly at load time.

### Key takeaway

The most important lesson from this project is that **the retrieval step is the quality gate for the whole AI pipeline**. If the wrong facts are retrieved, the LLM advice will be off-topic regardless of model quality. Writing tests for the retriever before integrating the LLM made that bottleneck visible early and gave me a clear acceptance criterion for "good enough" retrieval before spending time on prompt engineering.
