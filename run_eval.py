"""PawPal+ Evaluation Harness.

Runs the AI advisor pipeline on a fixed set of predefined scenarios and prints
a pass/fail summary table.  Does NOT call the LLM (no API key required) — it
evaluates the deterministic components: multi-source retrieval, confidence
scoring, and agentic schedule evaluation.

Run with:
    python run_eval.py

Exit code 0 if all checks pass, 1 if any fail.
"""

from __future__ import annotations

import sys
from datetime import datetime
from typing import Any

from pawpal_system import Owner, Pet, Scheduler, Task
from ai_advisor import (
    compute_confidence_score,
    evaluate_schedule,
    load_all_sources,
    load_knowledge_base,
    load_owner_notes,
    retrieve_facts,
)

# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

# Fixed date used across all scenarios so results are deterministic
_DATE = datetime(2026, 4, 30, 7, 0)  # 07:00 on the test day


def _make_scenario_1() -> tuple[Owner, Pet]:
    """Dog with a walk AND a feeding — complete schedule, should pass clean."""
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    owner.add_pet(dog)
    dog.add_task(Task(description="Morning walk",  time=_DATE.replace(hour=8),  frequency="daily"))
    dog.add_task(Task(description="Feed breakfast", time=_DATE.replace(hour=9),  frequency="daily"))
    return owner, dog


def _make_scenario_2() -> tuple[Owner, Pet]:
    """Dog with only a walk — feeding gap expected."""
    owner = Owner(name="Alex")
    dog = Pet(name="Rex", species="dog", age=5)
    owner.add_pet(dog)
    dog.add_task(Task(description="Morning walk", time=_DATE.replace(hour=8), frequency="daily"))
    return owner, dog


def _make_scenario_3() -> tuple[Owner, Pet]:
    """Dog with only a feeding — exercise gap expected."""
    owner = Owner(name="Sam")
    dog = Pet(name="Bella", species="dog", age=2)
    owner.add_pet(dog)
    dog.add_task(Task(description="Feed breakfast", time=_DATE.replace(hour=9), frequency="daily"))
    return owner, dog


def _make_scenario_4() -> tuple[Owner, Pet]:
    """Cat with a play session and feeding — complete, should pass clean."""
    owner = Owner(name="Taylor")
    cat = Pet(name="Luna", species="cat", age=5)
    owner.add_pet(cat)
    cat.add_task(Task(description="Play session",  time=_DATE.replace(hour=18), frequency="daily"))
    cat.add_task(Task(description="Feed dinner",   time=_DATE.replace(hour=19), frequency="daily"))
    return owner, cat


def _make_scenario_5() -> tuple[Owner, Pet]:
    """Two pets with same-time tasks — conflict expected, gaps expected."""
    owner = Owner(name="Morgan")
    dog = Pet(name="Buddy", species="dog", age=4)
    cat = Pet(name="Whiskers", species="cat", age=3)
    owner.add_pet(dog)
    owner.add_pet(cat)
    dog.add_task(Task(description="Morning walk",  time=_DATE.replace(hour=8), frequency="daily"))
    cat.add_task(Task(description="Feed breakfast", time=_DATE.replace(hour=8), frequency="daily"))
    return owner, dog  # evaluate from dog's perspective; schedule check covers both


def _make_scenario_6() -> tuple[Owner, Pet]:
    """Dog with a medication task — owner_notes source should be retrieved."""
    owner = Owner(name="Casey")
    dog = Pet(name="Daisy", species="dog", age=7)
    owner.add_pet(dog)
    dog.add_task(Task(description="Give medication", time=_DATE.replace(hour=8), frequency="daily"))
    return owner, dog


# ---------------------------------------------------------------------------
# Check helpers
# ---------------------------------------------------------------------------


def _check(label: str, condition: bool, detail: str = "") -> dict[str, Any]:
    status = "PASS" if condition else "FAIL"
    return {"label": label, "status": status, "detail": detail}


# ---------------------------------------------------------------------------
# Scenario runners
# ---------------------------------------------------------------------------


def run_scenario_1() -> list[dict[str, Any]]:
    owner, dog = _make_scenario_1()
    kb = load_all_sources()
    tasks = [t for t in owner.get_all_tasks() if t in dog.tasks]
    facts = retrieve_facts(dog, tasks, kb)
    score = compute_confidence_score(dog, tasks, facts)
    obs = evaluate_schedule(owner, current_time=_DATE)
    gaps = [o for o in obs if o.startswith("⚠")]

    return [
        _check("S1: retrieval returns ≥3 facts",      len(facts) >= 3,   f"got {len(facts)}"),
        _check("S1: confidence ≥ 0.4",                score >= 0.4,      f"got {score:.2f}"),
        _check("S1: no gap warnings",                  len(gaps) == 0,    f"gaps={gaps}"),
        _check("S1: both sources present in pool",
               any(f.get("source") == "owner_notes" for f in kb),
               f"sources={sorted({f.get('source') for f in kb})}"),
    ]


def run_scenario_2() -> list[dict[str, Any]]:
    owner, dog = _make_scenario_2()
    obs = evaluate_schedule(owner, current_time=_DATE)
    feeding_gaps = [o for o in obs if "feeding" in o.lower()]
    exercise_gaps = [o for o in obs if "walk" in o.lower() or "exercise" in o.lower()]

    return [
        _check("S2: feeding gap flagged",    len(feeding_gaps) >= 1,   f"obs={obs}"),
        _check("S2: no exercise gap",        len(exercise_gaps) == 0,  f"obs={obs}"),
    ]


def run_scenario_3() -> list[dict[str, Any]]:
    owner, dog = _make_scenario_3()
    obs = evaluate_schedule(owner, current_time=_DATE)
    feeding_gaps  = [o for o in obs if "feeding" in o.lower()]
    exercise_gaps = [o for o in obs if "walk" in o.lower() or "exercise" in o.lower()]

    return [
        _check("S3: exercise gap flagged",  len(exercise_gaps) >= 1,  f"obs={obs}"),
        _check("S3: no feeding gap",        len(feeding_gaps) == 0,   f"obs={obs}"),
    ]


def run_scenario_4() -> list[dict[str, Any]]:
    owner, cat = _make_scenario_4()
    kb = load_all_sources()
    tasks = [t for t in owner.get_all_tasks() if t in cat.tasks]
    facts = retrieve_facts(cat, tasks, kb)
    score = compute_confidence_score(cat, tasks, facts)
    obs = evaluate_schedule(owner, current_time=_DATE)
    gaps = [o for o in obs if o.startswith("⚠")]

    # At least one retrieved fact should have a 'cat' tag
    cat_facts = [f for f in facts if "cat" in [t.lower() for t in f.get("tags", [])]]

    return [
        _check("S4: cat-relevant facts retrieved", len(cat_facts) >= 1, f"cat_facts={len(cat_facts)}"),
        _check("S4: confidence ≥ 0.3",             score >= 0.3,        f"got {score:.2f}"),
        _check("S4: no gap warnings",              len(gaps) == 0,      f"gaps={gaps}"),
    ]


def run_scenario_5() -> list[dict[str, Any]]:
    owner, dog = _make_scenario_5()
    scheduler = Scheduler()
    conflicts = scheduler.detect_conflicts(owner)
    obs = evaluate_schedule(owner, current_time=_DATE)
    feeding_gaps = [o for o in obs if "feeding" in o.lower()]

    return [
        _check("S5: time conflict detected",   len(conflicts) >= 1,      f"conflicts={conflicts}"),
        _check("S5: dog feeding gap flagged",  len(feeding_gaps) >= 1,   f"obs={obs}"),
    ]


def run_scenario_6() -> list[dict[str, Any]]:
    """Owner notes should be retrieved for a dog with a medication task."""
    owner, dog = _make_scenario_6()
    kb = load_all_sources()
    tasks = [t for t in owner.get_all_tasks() if t in dog.tasks]
    facts = retrieve_facts(dog, tasks, kb)
    owner_note_facts = [f for f in facts if f.get("source") == "owner_notes"]

    return [
        _check("S6: owner_notes source retrieved",
               len(owner_note_facts) >= 1,
               f"owner_note facts={[f['fact'][:60] for f in owner_note_facts]}"),
    ]


# ---------------------------------------------------------------------------
# Multi-source attribution check
# ---------------------------------------------------------------------------


def run_source_attribution_checks() -> list[dict[str, Any]]:
    """Verify source labels are present and correct on loaded facts."""
    kb = load_knowledge_base()
    notes = load_owner_notes()
    all_facts = load_all_sources()

    kb_labeled    = all(f.get("source") == "pet_care_facts" for f in kb)
    notes_labeled = all(f.get("source") == "owner_notes"    for f in notes)
    total_correct = len(all_facts) == len(kb) + len(notes)

    return [
        _check("ATTR: pet_care_facts all labeled", kb_labeled,    f"{len(kb)} facts"),
        _check("ATTR: owner_notes all labeled",    notes_labeled, f"{len(notes)} notes"),
        _check("ATTR: combined count matches",     total_correct,
               f"expected {len(kb)+len(notes)}, got {len(all_facts)}"),
    ]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> int:
    all_results: list[dict[str, Any]] = []
    all_results += run_source_attribution_checks()
    all_results += run_scenario_1()
    all_results += run_scenario_2()
    all_results += run_scenario_3()
    all_results += run_scenario_4()
    all_results += run_scenario_5()
    all_results += run_scenario_6()

    # --- print summary table ---
    col_label  = max(len(r["label"]) for r in all_results) + 2
    col_status = 6
    print()
    print("PawPal+ Evaluation Harness")
    print("=" * (col_label + col_status + 6))
    print(f"{'Check':<{col_label}} {'Result':<{col_status}}  Detail")
    print("-" * (col_label + col_status + 40))
    for r in all_results:
        icon = "✓" if r["status"] == "PASS" else "✗"
        print(f"{r['label']:<{col_label}} {r['status']:<{col_status}}  {icon} {r['detail']}")
    print("-" * (col_label + col_status + 40))

    passed = sum(1 for r in all_results if r["status"] == "PASS")
    total  = len(all_results)
    print(f"\nResult: {passed}/{total} checks passed", end="")

    # Compute average confidence across scenarios with tasks
    conf_checks = [r for r in all_results if "confidence" in r["label"].lower()]
    if conf_checks:
        # Re-run to collect numeric scores for the summary line
        scores = []
        for owner_fn, pet_fn in [(_make_scenario_1, lambda o: o.pets[0]),
                                  (_make_scenario_4, lambda o: o.pets[0])]:
            owner, pet = owner_fn()
            kb = load_all_sources()
            tasks = [t for t in owner.get_all_tasks() if t in pet.tasks]
            facts = retrieve_facts(pet, tasks, kb)
            scores.append(compute_confidence_score(pet, tasks, facts))
        if scores:
            avg = sum(scores) / len(scores)
            print(f"  |  Average retrieval confidence: {avg:.2f}", end="")

    print()
    print("=" * (col_label + col_status + 6))

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
