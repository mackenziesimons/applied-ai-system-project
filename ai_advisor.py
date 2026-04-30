"""AI Advisor module for PawPal+.

Stretch features implemented:
  RAG Enhancement     — multi-source retrieval (pet_care_facts + owner_notes),
                        with source attribution on every returned fact.
  Agentic Workflow    — ``run_agent_chain()`` executes 7 named, observable steps
                        and returns all intermediate results for UI display.
  Few-shot Prompt     — ``_build_prompt()`` injects a labelled example that
                        constrains the LLM output to a structured format
                        (numbered tips + explicit GAP flags + confidence note).
  Test Harness        — see ``run_eval.py`` for the evaluation script.

All activity is logged to ``pawpal_advisor.log`` and to stderr.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from pawpal_system import Owner, Pet, Scheduler, Task

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.FileHandler("pawpal_advisor.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("pawpal.advisor")

# ---------------------------------------------------------------------------
# Knowledge sources  (RAG Enhancement: multiple sources with attribution)
# ---------------------------------------------------------------------------

_KB_PATH = Path(__file__).parent / "knowledge_base" / "pet_care_facts.json"
_OWNER_NOTES_PATH = Path(__file__).parent / "knowledge_base" / "owner_notes.json"


def _load_json_source(path: Path, source_label: str) -> list[dict[str, Any]]:
    """Load a JSON array from *path*, tagging each entry with *source_label*."""
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError(f"{path.name} root must be a JSON array.")
        for entry in data:
            entry.setdefault("source", source_label)
        logger.info("Loaded %d entries from '%s' (%s).", len(data), path.name, source_label)
        return data
    except FileNotFoundError:
        logger.warning("Source file not found: %s — skipping.", path)
        return []
    except (json.JSONDecodeError, ValueError) as exc:
        logger.error("Invalid JSON in %s: %s", path.name, exc)
        return []


def load_knowledge_base() -> list[dict[str, Any]]:
    """Load the curated pet care facts knowledge base (source: pet_care_facts)."""
    return _load_json_source(_KB_PATH, "pet_care_facts")


def load_owner_notes() -> list[dict[str, Any]]:
    """Load owner-written supplementary notes (source: owner_notes).

    Owners can edit ``knowledge_base/owner_notes.json`` to add pet-specific
    observations that are not covered by the curated knowledge base.
    """
    return _load_json_source(_OWNER_NOTES_PATH, "owner_notes")


def load_all_sources() -> list[dict[str, Any]]:
    """Merge all knowledge sources into one list for retrieval.

    RAG Enhancement: combining ``pet_care_facts`` (22 curated facts) with
    ``owner_notes`` (owner-customizable additions) lets the retriever surface
    both general best-practice advice and pet-specific context in a single pass.
    Each fact retains a ``source`` field so the UI can show provenance.
    """
    kb = load_knowledge_base()
    notes = load_owner_notes()
    combined = kb + notes
    sources = {f.get("source", "unknown") for f in combined}
    logger.info(
        "All sources loaded: %d total facts across %d source(s): %s.",
        len(combined), len(sources), sorted(sources),
    )
    return combined


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


def retrieve_facts(
    pet: Pet,
    tasks: list[Task],
    knowledge_base: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    """Return the top-k most relevant facts for the given pet and tasks.

    Relevance is scored by counting how many of a fact's tags appear in the
    combined set of tokens from the pet's species and task descriptions.
    """
    if not knowledge_base:
        logger.warning("Knowledge base is empty; no facts to retrieve.")
        return []

    # Build a token set from the pet's species + every task description word
    query_tokens: set[str] = {pet.species.lower()}
    for task in tasks:
        query_tokens.update(word.lower() for word in task.description.split())

    scored: list[tuple[int, dict[str, Any]]] = []
    for fact in knowledge_base:
        tags = {tag.lower() for tag in fact.get("tags", [])}
        score = len(query_tokens & tags)
        if score > 0:
            scored.append((score, fact))

    scored.sort(key=lambda item: item[0], reverse=True)
    results = [fact for _, fact in scored[:top_k]]

    logger.info(
        "Retrieved %d/%d relevant facts for pet '%s' (%d tasks in context).",
        len(results),
        len(knowledge_base),
        pet.name,
        len(tasks),
    )
    return results


def compute_confidence_score(
    pet: Pet,
    tasks: list[Task],
    retrieved_facts: list[dict[str, Any]],
) -> float:
    """Return a 0.0-1.0 confidence score for the retrieval step.

    Each retrieved fact is scored as matched_tags / total_tags (its individual
    precision).  The overall confidence is the average of those per-fact scores,
    clamped to [0.0, 1.0].  A score of 0.0 means no facts were retrieved or
    none of the tags matched; 1.0 means every tag on every retrieved fact was
    present in the query.
    """
    if not retrieved_facts:
        logger.info("Confidence score: 0.0 (no facts retrieved).")
        return 0.0

    query_tokens: set[str] = {pet.species.lower()}
    for task in tasks:
        query_tokens.update(word.lower() for word in task.description.split())

    per_fact_scores: list[float] = []
    for fact in retrieved_facts:
        tags = {tag.lower() for tag in fact.get("tags", [])}
        if not tags:
            continue
        per_fact_scores.append(len(query_tokens & tags) / len(tags))

    if not per_fact_scores:
        return 0.0

    score = round(min(1.0, sum(per_fact_scores) / len(per_fact_scores)), 2)
    logger.info(
        "Confidence score for pet '%s': %.2f (avg over %d facts).",
        pet.name,
        score,
        len(per_fact_scores),
    )
    return score


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


# Few-shot example used to specialise the output format.
# Showing one labelled example measurably changes the LLM output:
# it produces numbered tips, explicit GAP lines, and a confidence note
# instead of free-form prose, making the advice easier to scan and test.
_FEW_SHOT_EXAMPLE = """\
EXAMPLE (do not repeat this — use it only to learn the required format)
------------------------------------------------------------------------
Pet: Buddy (dog, 4 years) | Tasks today: Morning walk 08:00
Retrieved facts:
  [1] Adult dogs do best with two meals per day, 8-12 hours apart.
  [2] Give oral medications at the same time daily; absorb better with food.

Expected output format:
1. Buddy has a morning walk scheduled — great start. Aim for at least 30 minutes
   to meet his daily exercise needs (fact [2] notes consistency is key).
2. Consider pairing any medication with breakfast to improve absorption (fact [2]).
⚠ GAP: No feeding task found. Adult dogs need two meals daily (fact [1]) —
  add a breakfast task around 08:30 and a dinner task around 19:00.
Retrieval confidence: high (facts directly matched dog + walk + medication tags)
------------------------------------------------------------------------
"""


def _build_prompt(
    pet: Pet,
    tasks: list[Task],
    retrieved_facts: list[dict[str, Any]],
) -> str:
    """Compose the few-shot RAG prompt sent to the LLM.

    Few-shot specialisation: a labelled input/output example is prepended so
    the model learns the required structured format — numbered tips, explicit
    GAP warnings, a closing confidence note — without any fine-tuning.
    The source label on each retrieved fact is included so the model (and the
    owner) can see whether advice comes from the curated KB or owner notes.
    """
    if tasks:
        task_lines = "\n".join(
            f"  - {t.description} at {t.time.strftime('%H:%M')} "
            f"({'completed' if t.completed else 'pending'}, {t.frequency})"
            for t in tasks
        )
    else:
        task_lines = "  (no tasks scheduled for today)"

    if retrieved_facts:
        facts_block = "\n".join(
            f"  [{i + 1}] [{fact.get('source', 'kb')}] {fact['fact']}"
            for i, fact in enumerate(retrieved_facts)
        )
    else:
        facts_block = "  (no relevant facts retrieved)"

    return (
        "You are a veterinary-informed pet care assistant. "
        "Use ONLY the retrieved facts listed below when giving advice — "
        "do not invent or assume information not present in the facts.\n\n"
        + _FEW_SHOT_EXAMPLE
        + "\nNow follow the same format for the pet below.\n\n"
        f"Pet profile:\n"
        f"  Name:       {pet.name}\n"
        f"  Species:    {pet.species}\n"
        f"  Age:        {pet.age} year(s)\n"
        f"  Preferences: {pet.preferences if pet.preferences else 'none recorded'}\n\n"
        f"Today's tasks:\n{task_lines}\n\n"
        f"Retrieved care facts (source shown in brackets):\n{facts_block}\n\n"
        "Provide 2-3 numbered tips grounded in the retrieved facts. "
        "For any missing care category (feeding, exercise, medication) add a "
        "line starting with '\u26a0 GAP:'. "
        "End with a one-line 'Retrieval confidence:' note."
    )


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------


def generate_advice(
    pet: Pet,
    tasks: list[Task],
    retrieved_facts: list[dict[str, Any]],
    model: str = "gpt-4o-mini",
) -> str:
    """Call the OpenAI chat API and return AI-generated care advice.

    Returns a graceful fallback string when no API key is set or the call fails.
    Does NOT raise exceptions — all errors are caught, logged, and surfaced as
    human-readable messages so the Streamlit UI never crashes.
    """
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        logger.warning("OPENAI_API_KEY is not set — AI advice will be unavailable.")
        return (
            "AI advice is currently unavailable. "
            "Set the OPENAI_API_KEY environment variable (see .env.example) "
            "to enable personalized recommendations."
        )

    try:
        # Import here so the module loads without error even if openai is missing
        from openai import OpenAI  # noqa: PLC0415

        prompt = _build_prompt(pet, tasks, retrieved_facts)
        logger.info(
            "Requesting advice from model '%s' for pet '%s' (%d facts in context).",
            model,
            pet.name,
            len(retrieved_facts),
        )

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0.3,
        )
        advice = response.choices[0].message.content.strip()
        logger.info(
            "Received %d-character advice response for pet '%s'.",
            len(advice),
            pet.name,
        )
        return advice

    except ImportError:
        logger.error("openai package is not installed. Run: pip install openai")
        return "AI advice unavailable: the 'openai' package is not installed."
    except Exception as exc:  # pylint: disable=broad-except
        logger.error(
            "LLM call failed for pet '%s': %s: %s",
            pet.name,
            type(exc).__name__,
            exc,
        )
        return (
            f"AI advice unavailable right now ({type(exc).__name__}). "
            "Check pawpal_advisor.log for details."
        )


# ---------------------------------------------------------------------------
# Schedule evaluation — agentic self-check
# ---------------------------------------------------------------------------


def evaluate_schedule(
    owner: Owner,
    current_time: datetime | None = None,
) -> list[str]:
    """Evaluate the owner's today-plan and return a list of observations.

    This is the *agentic* step: the advisor reviews the schedule it will act on
    and flags gaps (missing meals, no exercise for dogs) and time conflicts
    before the owner starts their day.  Returns observations as plain strings
    so they can be rendered in any context (UI, terminal, test).
    """
    scheduler = Scheduler()
    plan = scheduler.build_today_plan(owner, current_time=current_time)
    conflicts = scheduler.detect_conflicts(owner)
    observations: list[str] = []

    if not plan:
        observations.append("No tasks are scheduled for today.")
        logger.info("Schedule evaluation: no tasks found for owner '%s'.", owner.name)
        return observations

    for pet in owner.pets:
        pet_tasks = [t for t in plan if t in pet.tasks]
        desc_set = {t.description.lower() for t in pet_tasks}

        # Check for a feeding task
        feeding_kw = {"feed", "feeding", "breakfast", "lunch", "dinner", "meal", "food"}
        if not any(kw in desc for kw in feeding_kw for desc in desc_set):
            observations.append(
                f"⚠ {pet.name} has no feeding task scheduled today."
            )

        # Dogs specifically need exercise
        if pet.species.lower() == "dog":
            exercise_kw = {"walk", "exercise", "run", "jog", "outside", "play"}
            if not any(kw in desc for kw in exercise_kw for desc in desc_set):
                observations.append(
                    f"⚠ {pet.name} (dog) has no walk or exercise task today."
                )

    # Include any time-conflict warnings from the scheduler
    observations.extend(conflicts)

    if not observations:
        observations.append("✓ Schedule looks complete — no issues detected.")

    logger.info(
        "Schedule evaluation for owner '%s': %d observation(s).",
        owner.name,
        len(observations),
    )
    return observations


# ---------------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------------


def advise(
    owner: Owner,
    pet: Pet,
    current_time: datetime | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Run the full RAG + agentic pipeline for a single pet.

    Steps:
      1. Load knowledge base from disk.
      2. Retrieve relevant facts for this pet's today-tasks (RAG retrieval).
      3. Call the LLM with retrieved facts as context (RAG generation).
      4. Evaluate the complete schedule for gaps and conflicts (agentic check).

    Returns a dict with:
      - ``advice``               — LLM-generated care tips (str)
      - ``retrieved_facts``      — facts used as RAG context (list[dict])
      - ``schedule_observations``— agentic evaluation results (list[str])
      - ``confidence_score``     — 0.0-1.0 retrieval confidence (float)

    For the full observable chain (stretch feature), use ``run_agent_chain()``.
    """
    result = run_agent_chain(owner, pet, current_time=current_time, model=model)
    return {
        "advice": result["advice"],
        "retrieved_facts": result["retrieved_facts"],
        "schedule_observations": result["schedule_observations"],
        "confidence_score": result["confidence_score"],
    }


# ---------------------------------------------------------------------------
# Agentic workflow — observable multi-step reasoning chain
# ---------------------------------------------------------------------------


def run_agent_chain(
    owner: Owner,
    pet: Pet,
    current_time: datetime | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """Execute the full advisor pipeline as an observable multi-step chain.

    Agentic Workflow Enhancement: each step is named, logged, and returned
    in ``chain_steps`` so the UI (or test harness) can display every
    intermediate decision, not just the final answer.

    Steps:
      1. profile_analysis    — summarise pet profile
      2. task_context        — collect today's incomplete tasks for this pet
      3. gap_detection       — agentic self-check for missing care categories
      4. fact_retrieval      — multi-source RAG across all knowledge sources
      5. confidence_scoring  — score retrieval quality
      6. advice_generation   — few-shot LLM call with retrieved facts
      7. self_evaluation     — check whether advice addresses detected gaps

    Returns a dict with all outputs plus ``chain_steps`` (list of step dicts).
    """
    chain_steps: list[dict[str, Any]] = []

    def _step(
        name: str,
        input_summary: str,
        output_summary: str,
        data: dict[str, Any] | None = None,
    ) -> None:
        entry: dict[str, Any] = {
            "step": name,
            "input": input_summary,
            "output": output_summary,
        }
        if data:
            entry["data"] = data
        chain_steps.append(entry)
        logger.info(
            "[CHAIN:%s] in=%s | out=%s", name, input_summary, output_summary
        )

    # Step 1 — profile analysis
    pref_keys = list(pet.preferences.keys()) if pet.preferences else ["none"]
    profile_summary = (
        f"{pet.species}, age {pet.age}, preferences: {pref_keys}"
    )
    _step(
        "profile_analysis",
        f"pet={pet.name}",
        profile_summary,
        data={"species": pet.species, "age": pet.age, "preferences": pref_keys},
    )

    # Step 2 — task context
    scheduler = Scheduler()
    today_plan = scheduler.build_today_plan(owner, current_time=current_time)
    pet_tasks = [t for t in today_plan if t in pet.tasks]
    task_descs = [t.description for t in pet_tasks]
    _step(
        "task_context",
        f"owner={owner.name}, date={( current_time or datetime.now()).date()}",
        f"{len(pet_tasks)} task(s): {task_descs}",
        data={"tasks": task_descs},
    )

    # Step 3 — gap detection
    observations = evaluate_schedule(owner, current_time=current_time)
    gaps = [o for o in observations if o.startswith("\u26a0")]
    _step(
        "gap_detection",
        f"{len(pet_tasks)} task(s) evaluated",
        f"{len(gaps)} gap(s) detected",
        data={"gaps": gaps, "all_observations": observations},
    )

    # Step 4 — multi-source fact retrieval (RAG Enhancement)
    all_facts = load_all_sources()
    source_counts = {}
    for f in all_facts:
        s = f.get("source", "unknown")
        source_counts[s] = source_counts.get(s, 0) + 1
    retrieved = retrieve_facts(pet, pet_tasks, all_facts, top_k=5)
    retrieved_sources = sorted({f.get("source", "unknown") for f in retrieved})
    _step(
        "fact_retrieval",
        f"{len(all_facts)} facts across sources: {source_counts}",
        f"{len(retrieved)} fact(s) retrieved from: {retrieved_sources}",
        data={
            "retrieved_facts": [{"fact": f["fact"], "source": f.get("source")} for f in retrieved],
            "sources_used": retrieved_sources,
        },
    )

    # Step 5 — confidence scoring
    confidence = compute_confidence_score(pet, pet_tasks, retrieved)
    _step(
        "confidence_scoring",
        f"{len(retrieved)} retrieved fact(s)",
        f"score={confidence:.2f}",
        data={"confidence_score": confidence},
    )

    # Step 6 — few-shot advice generation
    advice = generate_advice(pet, pet_tasks, retrieved, model=model)
    _step(
        "advice_generation",
        f"model={model}, facts={len(retrieved)}, gaps={len(gaps)}",
        f"{len(advice)} character(s) generated",
        data={"model": model, "advice_preview": advice[:120] + "..." if len(advice) > 120 else advice},
    )

    # Step 7 — self-evaluation: does advice address each detected gap?
    unaddressed: list[str] = []
    for gap in gaps:
        gap_kw = {
            w for w in gap.lower().split()
            if w not in {"a", "an", "the", "has", "no", "for", "is", "today.", "today"}
        }
        if not any(kw in advice.lower() for kw in gap_kw):
            unaddressed.append(gap)

    self_eval = (
        "all detected gaps addressed in advice"
        if not unaddressed
        else f"{len(unaddressed)} gap(s) not explicitly addressed: {unaddressed}"
    )
    _step(
        "self_evaluation",
        f"{len(gaps)} gap(s) to verify",
        self_eval,
        data={"unaddressed_gaps": unaddressed},
    )

    return {
        "advice": advice,
        "retrieved_facts": retrieved,
        "schedule_observations": observations,
        "confidence_score": confidence,
        "chain_steps": chain_steps,
    }
