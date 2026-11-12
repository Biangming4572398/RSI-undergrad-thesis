"""Prompt text and helpers shared by the ALFWorld agents."""

import re

from Jericho.src.openai_helpers import chat_completion_with_retries

ENV_RULES = """You are an expert agent in ALFWorld, a text-based household simulator. Each episode has one household task. The episode is won as soon as the task's goal holds.

How the environment works:
- You stand at one receptacle at a time. "go to X" moves you to X and lists what is on or in it.
- Closed receptacles (fridge, drawer, cabinet, microwave, safe, ...) must be opened before you can see or take what is inside.
- You carry at most one object. "take X from Y" picks it up; "move X to Y" puts the held object on or in Y.
- Process a held object at an appliance: "clean X with sinkbasin", "heat X with microwave", "cool X with fridge". To look at an object under a light, hold it and "use" a desklamp or floorlamp.
- "some X" or "a X" means any object of that type. Tasks with "two" need two objects of that type, placed one at a time.
- Objects may be anywhere. Check the likely receptacles first, then the rest systematically. Do not revisit a receptacle you already checked unless something changed."""


def match_action(action, valid):
    """Fix the format of an action when that makes it an admissible command. Otherwise return it unchanged."""
    if not valid:
        return action
    text = " ".join(word for word in action.lower().strip().strip(".\"'`").split() if word != "the")
    candidates = [text]
    put = re.fullmatch(r"put (.+?) (?:in/on|into|onto|in|on) (.+)", text)
    if put:  # Older ALFWorld prompts use "put X in/on Y"; this version calls it "move X to Y".
        candidates.append(f"move {put[1]} to {put[2]}")
    for candidate in candidates:
        if candidate in valid:
            return candidate
    return action


def format_recent_steps(history, limit=20):
    text = ""
    start = max(0, len(history) - limit)
    for index, entry in enumerate(history[start:], start):
        text += f"Step {index}:\nObservation: {entry.get('state', '')}\nAction: {entry.get('action', '')}\n\n"
    return text


def summarize_progress(game_history, current_state, task, llm_model, inventory=None):
    """Return (summary paragraph, structured sections) for the episode so far.

    Unlike the Jericho summary, progress here means completed subgoals, because
    ALFWorld gives no reward before the task is done.
    """
    trajectory = format_recent_steps(game_history, limit=len(game_history))
    trajectory += f"Step {len(game_history)} (current):\nObservation: {current_state}\n"
    sys_prompt = ("You summarize an agent's progress on an ALFWorld household task. "
                  "Be concise and concrete. Use exact object and receptacle names with their numbers.")
    user_prompt = f"""Task: {task}

Trajectory:
{trajectory}
Write exactly these sections:

[SUMMARY]
Two or three sentences: what the task requires, what is done, and what remains.

[PROGRESS]
One line per completed subgoal, for example "✓ found bowl 1 at countertop 3", "✓ took bowl 1", "✓ cleaned bowl 1 with sinkbasin 1".
Write "No Progress" if no subgoal is complete.

[SEARCHED]
Receptacles already checked, with any task-relevant objects seen there, for example "drawer 1: nothing useful; fridge 1: apple 2".

[LOCATION]
Current receptacle and what the agent is holding."""

    response = chat_completion_with_retries(
        model=llm_model, sys_prompt=sys_prompt, prompt=user_prompt, max_tokens=1000, temperature=0)
    if not (response and getattr(response, "choices", None)):
        print("[Warning] No response from LLM for ALFWorld progress summary")
        return "", f"[INVENTORY]\n{inventory}" if inventory else ""
    text = (response.choices[0].message.content or "").strip()
    summary, structured = "", text
    if "[SUMMARY]" in text:
        summary, _, rest = text.split("[SUMMARY]", 1)[1].partition("[PROGRESS]")
        summary = summary.strip()
        structured = "[PROGRESS]" + rest if rest else ""
    if inventory:
        structured += f"\n\n[INVENTORY]\n{inventory}"
    return summary, structured
