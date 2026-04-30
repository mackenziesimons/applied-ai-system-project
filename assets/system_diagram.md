# PawPal+ System Diagram

```mermaid
flowchart TD
    subgraph INPUT["Input Layer"]
        U["👤 Owner / User"]
        UI["Streamlit UI\n(app.py)"]
        SAMPLE["Sample Data Builder\n(main.py)"]
    end

    subgraph CORE["Core Domain — pawpal_system.py"]
        OWNER["Owner\n• name\n• pets list"]
        PET["Pet\n• name, species, age\n• preferences\n• tasks list"]
        TASK["Task\n• description, time\n• frequency, completed"]
        SCHED["Scheduler\n• build_today_plan()\n• filter_tasks()\n• get_upcoming_tasks()\n• complete_task()\n• detect_conflicts()"]
    end

    subgraph AI["AI Advisor — ai_advisor.py"]
        KB["Knowledge Base\npet_care_facts.json\n(22 tagged facts)"]
        RETRIEVER["Retriever\nretrieve_facts()\n• score tags vs. pet + tasks\n• return top-k facts"]
        LLM["LLM\ngenerate_advice()\n• OpenAI gpt-4o-mini\n• RAG prompt with facts\n• max 400 tokens"]
        AGENT["Agentic Evaluator\nevaluate_schedule()\n• check for missing feedings\n• check dog exercise\n• surface conflicts"]
        LOG["Logger\npawpal_advisor.log\n• every step recorded"]
    end

    subgraph OUTPUT["Output Layer"]
        PLAN["Ordered Daily Plan\n(sorted task list)"]
        ADVICE["AI Care Advice\n(LLM response)"]
        OBSERVATIONS["Schedule Observations\n(gaps + conflicts)"]
        NEXT["Next Recurrence\n(auto-generated Task)"]
    end

    subgraph VERIFICATION["Human / Automated Verification"]
        PYTEST["pytest suite\ntests/test_pawpal.py\n• Task, Pet, Scheduler unit tests"]
        HUMAN["Human Review\nvia Streamlit UI\n• reads advice\n• marks tasks complete"]
    end

    U --> UI
    SAMPLE --> OWNER
    UI --> OWNER
    OWNER --> PET
    PET --> TASK
    OWNER --> SCHED
    SCHED -->|"aggregates & filters"| TASK
    SCHED -->|"chronological output"| PLAN
    TASK -->|"next_occurrence()"| NEXT
    NEXT --> PET

    PLAN -->|"today's pet tasks"| RETRIEVER
    KB -->|"all facts"| RETRIEVER
    RETRIEVER -->|"top-k relevant facts"| LLM
    LLM -->|"care tips"| ADVICE
    SCHED -->|"full daily plan"| AGENT
    AGENT -->|"gap + conflict warnings"| OBSERVATIONS

    PLAN --> UI
    ADVICE --> UI
    OBSERVATIONS --> UI

    AI -->|"all steps logged"| LOG

    PYTEST -->|"unit tests"| TASK
    PYTEST -->|"unit tests"| PET
    PYTEST -->|"unit tests"| SCHED
    HUMAN -->|"reviews advice\nchecks observations"| ADVICE
    HUMAN -->|"marks tasks complete"| SCHED
```

## Component Summary

| Layer | Component | Role |
|---|---|---|
| Input | `app.py` (Streamlit UI) | Owner enters pets and tasks interactively |
| Input | `main.py` | Seeds sample data for testing/demo |
| Core | `Owner` | Holds pets, aggregates all tasks |
| Core | `Pet` | Holds task list and preferences per pet |
| Core | `Task` | Describes a single care event with time/frequency |
| Core | `Scheduler` | Builds daily plan, filters, handles recurrence, detects conflicts |
| AI | Knowledge Base | 22 tagged, vet-informed pet care facts (JSON) |
| AI | Retriever | Scores and retrieves the most relevant facts for the current pet + tasks (RAG) |
| AI | LLM | Generates personalized care advice grounded in retrieved facts only |
| AI | Agentic Evaluator | Self-checks the schedule for missing feedings, exercise, and time conflicts |
| AI | Logger | Writes every pipeline step to `pawpal_advisor.log` |
| Output | Daily Plan | Sorted, filtered task list for the day |
| Output | AI Care Advice | LLM-generated tips, grounded in retrieved facts |
| Output | Schedule Observations | Warnings about gaps or conflicts found by the evaluator |
| Output | Next Recurrence | Auto-generated follow-up task for repeating items |
| Verification | `tests/test_pawpal.py` | Automated pytest unit tests for core domain |
| Verification | Streamlit UI | Human reads advice, verifies observations, marks tasks complete |
