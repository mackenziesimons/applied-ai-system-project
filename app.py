import os

import streamlit as st
from dotenv import load_dotenv

from datetime import datetime

from ai_advisor import advise, run_agent_chain
from pawpal_system import Owner, Pet, Scheduler, Task

# Load environment variables from .env if present (ignored when not found)
load_dotenv()

st.set_page_config(page_title="PawPal+", page_icon="🐾", layout="centered")

st.title("🐾 PawPal+")

st.markdown(
    """
Welcome to the PawPal+ starter app.

This file is intentionally thin. It gives you a working Streamlit app so you can start quickly,
but **it does not implement the project logic**. Your job is to design the system and build it.

Use this app as your interactive demo once your backend classes/functions exist.
"""
)

with st.expander("Scenario", expanded=True):
    st.markdown(
        """
**PawPal+** is a pet care planning assistant. It helps a pet owner plan care tasks
for their pet(s) based on constraints like time, priority, and preferences.

You will design and implement the scheduling logic and connect it to this Streamlit UI.
"""
    )

with st.expander("What you need to build", expanded=True):
    st.markdown(
        """
At minimum, your system should:
- Represent pet care tasks (what needs to happen, how long it takes, priority)
- Represent the pet and the owner (basic info and preferences)
- Build a plan/schedule for a day that chooses and orders tasks based on constraints
- Explain the plan (why each task was chosen and when it happens)
"""
    )

st.divider()

if "owner" not in st.session_state:
    st.session_state.owner = Owner(name="Jordan")

if "scheduler" not in st.session_state:
    st.session_state.scheduler = Scheduler()

owner: Owner = st.session_state.owner
scheduler: Scheduler = st.session_state.scheduler

st.subheader("Owner and Pets")
owner_name = st.text_input("Owner name", value=owner.name)
owner.name = owner_name

with st.form("add_pet_form"):
    st.markdown("### Add a Pet")
    pet_name = st.text_input("Pet name", value="Mochi")
    species = st.selectbox("Species", ["dog", "cat", "other"])
    age = st.number_input("Age", min_value=0, max_value=50, value=3)
    favorite_activity = st.text_input("Favorite activity", value="Walks")
    add_pet_submitted = st.form_submit_button("Add pet")

    if add_pet_submitted:
        try:
            owner.add_pet(
                Pet(
                    name=pet_name.strip(),
                    species=species,
                    age=int(age),
                    preferences={"favorite_activity": favorite_activity.strip()},
                )
            )
            st.success(f"Added {pet_name.strip()} to {owner.name}'s pets.")
        except ValueError as error:
            st.error(str(error))

if owner.pets:
    st.write("Current pets:")
    st.table(
        [
            {
                "name": pet.name,
                "species": pet.species,
                "age": pet.age,
                "favorite_activity": pet.preferences.get("favorite_activity", ""),
            }
            for pet in owner.pets
        ]
    )
else:
    st.info("No pets yet. Add one above.")

st.markdown("### Tasks")
st.caption("Add tasks to a specific pet using your backend classes.")

if owner.pets:
    with st.form("add_task_form"):
        pet_options = [pet.name for pet in owner.pets]
        selected_pet_name = st.selectbox("Pet", pet_options)
        task_description = st.text_input("Task description", value="Morning walk")
        task_date = st.date_input("Due date")
        task_time = st.time_input("Due time")
        frequency = st.selectbox("Frequency", ["once", "daily", "weekly", "monthly"])
        add_task_submitted = st.form_submit_button("Add task")

        if add_task_submitted:
            selected_pet = owner.get_pet(selected_pet_name)
            if selected_pet is None:
                st.error("Selected pet was not found.")
            else:
                due_time = datetime.combine(task_date, task_time)
                selected_pet.add_task(
                    Task(
                        description=task_description.strip(),
                        time=due_time,
                        frequency=frequency,
                    )
                )
                st.success(f"Added task for {selected_pet_name}.")
else:
    st.info("Add a pet before scheduling tasks.")

current_tasks = owner.get_all_tasks(include_completed=True)
if current_tasks:
    st.write("Current tasks:")
    st.table(
        [
            {
                "pet": pet.name,
                "description": task.description,
                "time": task.time.strftime("%Y-%m-%d %H:%M"),
                "frequency": task.frequency,
                "completed": task.completed,
            }
            for pet in owner.pets
            for task in pet.get_tasks(include_completed=True)
        ]
    )
else:
    st.info("No tasks yet. Add one above.")

st.divider()

st.subheader("Build Schedule")
st.caption("This button now calls your scheduling logic and shows today's tasks.")

if st.button("Generate schedule"):
    today_plan = scheduler.build_today_plan(owner)
    if today_plan:
        st.write("Today's plan:")
        st.table(
            [
                {
                    "description": task.description,
                    "time": task.time.strftime("%Y-%m-%d %H:%M"),
                    "frequency": task.frequency,
                }
                for task in today_plan
            ]
        )
    else:
        st.info("No incomplete tasks are scheduled for today.")

st.divider()

# ---------------------------------------------------------------------------
# AI Advisor  (RAG + agentic schedule check)
# ---------------------------------------------------------------------------
st.subheader("🤖 AI Advisor")
st.caption(
    "Retrieves relevant pet care facts from the local knowledge base, "
    "then asks an LLM to generate personalised advice for each pet. "
    "Also runs an agentic check of today's schedule to flag gaps or conflicts."
)

if not owner.pets:
    st.info("Add a pet to get AI-powered care advice.")
else:
    selected_advisor_pet = st.selectbox(
        "Get advice for",
        [pet.name for pet in owner.pets],
        key="advisor_pet_select",
    )

    if st.button("Run AI Advisor"):
        target_pet = owner.get_pet(selected_advisor_pet)
        if target_pet is None:
            st.error("Selected pet could not be found.")
        else:
            with st.spinner("Retrieving facts and generating advice…"):
                result = run_agent_chain(owner, target_pet)

            st.markdown("#### Care Advice")
            confidence = result["confidence_score"]
            if confidence >= 0.6:
                st.success(f"Retrieval confidence: {confidence:.0%}")
            elif confidence >= 0.3:
                st.warning(f"Retrieval confidence: {confidence:.0%} — limited context matched")
            else:
                st.error(f"Retrieval confidence: {confidence:.0%} — few relevant facts found; advice may be generic")
            st.write(result["advice"])

            with st.expander("Agent chain — intermediate steps"):
                for step in result.get("chain_steps", []):
                    st.markdown(f"**{step['step']}**")
                    st.caption(f"In: {step['input']}  →  Out: {step['output']}")
                    if "data" in step:
                        st.json(step["data"])

            with st.expander("Retrieved knowledge base facts used as context"):
                if result["retrieved_facts"]:
                    for fact in result["retrieved_facts"]:
                        source = fact.get("source", "kb")
                        st.markdown(f"- `[{source}]` {fact['fact']}")
                else:
                    st.write("No relevant facts were retrieved.")

            st.markdown("#### Schedule Check")
            for obs in result["schedule_observations"]:
                if obs.startswith("⚠") or "Conflict" in obs:
                    st.warning(obs)
                elif obs.startswith("✓"):
                    st.success(obs)
                else:
                    st.info(obs)
