from datetime import datetime

from pawpal_system import Owner, Pet, Scheduler, Task
from ai_advisor import (
    compute_confidence_score,
    evaluate_schedule,
    load_knowledge_base,
    retrieve_facts,
)


def test_mark_complete_changes_status():
    task = Task(description="Feed breakfast", time=datetime(2026, 3, 29, 9, 0))
    assert task.completed is False
    task.mark_complete()
    assert task.completed is True


def test_add_task_increases_pet_task_count():
    pet = Pet(name="Mochi", species="dog", age=3)
    assert len(pet.tasks) == 0
    pet.add_task(Task(description="Morning walk", time=datetime(2026, 3, 29, 8, 0)))
    assert len(pet.tasks) == 1


def test_sort_by_time_orders_tasks_chronologically():
    scheduler = Scheduler()
    tasks = [
        Task(description="Breakfast", time=datetime(2026, 3, 29, 9, 0)),
        Task(description="Walk", time=datetime(2026, 3, 29, 8, 0)),
    ]

    sorted_tasks = scheduler.sort_by_time(tasks)

    assert [task.description for task in sorted_tasks] == ["Walk", "Breakfast"]


def test_filter_tasks_by_pet_and_status_returns_matching_tasks():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    cat = Pet(name="Luna", species="cat", age=5)
    owner.add_pet(dog)
    owner.add_pet(cat)

    completed_task = Task(description="Breakfast", time=datetime(2026, 3, 29, 9, 0), completed=True)
    pending_task = Task(description="Walk", time=datetime(2026, 3, 29, 8, 0))
    dog.add_task(completed_task)
    dog.add_task(pending_task)
    cat.add_task(Task(description="Play", time=datetime(2026, 3, 29, 10, 0)))

    filtered_tasks = Scheduler().filter_tasks(owner, pet_name="Mochi", completed=False)

    assert [task.description for task in filtered_tasks] == ["Walk"]


def test_filter_tasks_by_completion_status_returns_all_completed_tasks():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    cat = Pet(name="Luna", species="cat", age=5)
    owner.add_pet(dog)
    owner.add_pet(cat)

    dog.add_task(Task(description="Breakfast", time=datetime(2026, 3, 29, 9, 0), completed=True))
    cat.add_task(Task(description="Brush coat", time=datetime(2026, 3, 29, 7, 30), completed=True))
    cat.add_task(Task(description="Play", time=datetime(2026, 3, 29, 10, 0)))

    filtered_tasks = Scheduler().filter_tasks(owner, completed=True)

    assert [task.description for task in filtered_tasks] == ["Brush coat", "Breakfast"]


def test_complete_recurring_task_creates_next_occurrence():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    owner.add_pet(dog)
    task = Task(
        description="Morning walk",
        time=datetime(2026, 3, 29, 8, 0),
        frequency="daily",
    )
    dog.add_task(task)

    next_task = Scheduler().complete_task(owner, task)

    assert task.completed is True
    assert next_task is not None
    assert next_task.time == datetime(2026, 3, 30, 8, 0)
    assert next_task.completed is False
    assert len(dog.tasks) == 2


def test_mark_task_complete_creates_next_weekly_occurrence():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    owner.add_pet(dog)
    task = Task(
        description="Weekly grooming",
        time=datetime(2026, 3, 29, 15, 0),
        frequency="weekly",
    )
    dog.add_task(task)

    next_task = Scheduler().mark_task_complete(owner, task)

    assert task.completed is True
    assert next_task is not None
    assert next_task.time == datetime(2026, 4, 5, 15, 0)
    assert next_task.frequency == "weekly"
    assert next_task.completed is False
    assert len(dog.tasks) == 2


def test_detect_conflicts_returns_warning_for_same_time_tasks():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    cat = Pet(name="Luna", species="cat", age=5)
    owner.add_pet(dog)
    owner.add_pet(cat)

    dog.add_task(Task(description="Walk", time=datetime(2026, 3, 29, 8, 0)))
    cat.add_task(Task(description="Breakfast", time=datetime(2026, 3, 29, 8, 0)))

    warnings = Scheduler().detect_conflicts(owner)

    assert len(warnings) == 1
    assert "Conflict detected" in warnings[0]


# ---------------------------------------------------------------------------
# AI Advisor tests
# ---------------------------------------------------------------------------


def test_load_knowledge_base_returns_nonempty_list():
    kb = load_knowledge_base()
    assert isinstance(kb, list)
    assert len(kb) > 0
    # Every entry must have 'fact' and 'tags' keys
    for entry in kb:
        assert "fact" in entry
        assert "tags" in entry


def test_retrieve_facts_returns_relevant_facts_for_dog():
    kb = load_knowledge_base()
    pet = Pet(name="Mochi", species="dog", age=3)
    tasks = [Task(description="Morning walk", time=datetime(2026, 3, 29, 8, 0))]

    facts = retrieve_facts(pet, tasks, kb, top_k=3)

    assert len(facts) <= 3
    # All returned facts should have at least one dog- or walk-related tag
    for fact in facts:
        tags = {t.lower() for t in fact["tags"]}
        assert tags & {"dog", "walk", "exercise"}, f"Unexpected fact tags: {tags}"


def test_retrieve_facts_returns_empty_for_empty_knowledge_base():
    pet = Pet(name="Mochi", species="dog", age=3)
    tasks = [Task(description="Feed breakfast", time=datetime(2026, 3, 29, 9, 0))]

    facts = retrieve_facts(pet, tasks, knowledge_base=[], top_k=5)

    assert facts == []


def test_confidence_score_is_between_zero_and_one():
    kb = load_knowledge_base()
    pet = Pet(name="Mochi", species="dog", age=3)
    tasks = [Task(description="Morning walk", time=datetime(2026, 3, 29, 8, 0))]
    facts = retrieve_facts(pet, tasks, kb)

    score = compute_confidence_score(pet, tasks, facts)

    assert 0.0 <= score <= 1.0


def test_confidence_score_is_zero_when_no_facts_retrieved():
    pet = Pet(name="Mochi", species="dog", age=3)
    tasks = [Task(description="Morning walk", time=datetime(2026, 3, 29, 8, 0))]

    score = compute_confidence_score(pet, tasks, retrieved_facts=[])

    assert score == 0.0


def test_confidence_score_higher_with_matching_tasks():
    kb = load_knowledge_base()
    pet = Pet(name="Mochi", species="dog", age=3)

    tasks_relevant = [
        Task(description="Morning walk", time=datetime(2026, 3, 29, 8, 0)),
        Task(description="Feed breakfast", time=datetime(2026, 3, 29, 9, 0)),
    ]
    tasks_irrelevant = [
        Task(description="Arbitrary xyz task", time=datetime(2026, 3, 29, 10, 0)),
    ]

    facts_relevant = retrieve_facts(pet, tasks_relevant, kb)
    facts_irrelevant = retrieve_facts(pet, tasks_irrelevant, kb)

    score_relevant = compute_confidence_score(pet, tasks_relevant, facts_relevant)
    score_irrelevant = compute_confidence_score(pet, tasks_irrelevant, facts_irrelevant)

    assert score_relevant >= score_irrelevant


def test_evaluate_schedule_flags_missing_feeding_for_dog():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    owner.add_pet(dog)
    # Only a walk — no feeding task
    dog.add_task(
        Task(
            description="Morning walk",
            time=datetime(2026, 4, 30, 8, 0),
            frequency="daily",
        )
    )

    observations = evaluate_schedule(owner, current_time=datetime(2026, 4, 30, 7, 0))

    feeding_warnings = [o for o in observations if "feeding" in o.lower()]
    assert len(feeding_warnings) >= 1


def test_evaluate_schedule_flags_missing_exercise_for_dog():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    owner.add_pet(dog)
    # Only a feeding task — no walk/exercise
    dog.add_task(
        Task(
            description="Feed breakfast",
            time=datetime(2026, 4, 30, 9, 0),
            frequency="daily",
        )
    )

    observations = evaluate_schedule(owner, current_time=datetime(2026, 4, 30, 7, 0))

    exercise_warnings = [o for o in observations if "walk" in o.lower() or "exercise" in o.lower()]
    assert len(exercise_warnings) >= 1


def test_evaluate_schedule_passes_when_dog_has_feeding_and_walk():
    owner = Owner(name="Jordan")
    dog = Pet(name="Mochi", species="dog", age=3)
    owner.add_pet(dog)
    dog.add_task(
        Task(description="Feed breakfast", time=datetime(2026, 4, 30, 9, 0), frequency="daily")
    )
    dog.add_task(
        Task(description="Morning walk", time=datetime(2026, 4, 30, 8, 0), frequency="daily")
    )

    observations = evaluate_schedule(owner, current_time=datetime(2026, 4, 30, 7, 0))

    assert any("complete" in o.lower() or o.startswith("\u2713") for o in observations)
