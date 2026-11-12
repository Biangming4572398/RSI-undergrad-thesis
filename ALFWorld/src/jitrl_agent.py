"""JitRL on ALFWorld: Jericho's agent and memory with ALFWorld prompts and success signal."""

import random
import time

from Jericho.src.jitrl_agent import JitRLAgent

from .prompts import ENV_RULES, format_recent_steps, match_action, summarize_progress

DEFAULT_GUIDE = ("Find the objects the task names, process them as the task asks (clean, heat, cool, "
                 "or examine under a lamp), and put them where the task says.")


class ALFWorldJitRLAgent(JitRLAgent):
    def __init__(self, args, guiding_prompt=None):
        super().__init__(args, guiding_prompt=guiding_prompt or DEFAULT_GUIDE)
        self._won = False
        if self.cross_mem is not None:
            # Jericho drops the last step of a won episode. In ALFWorld that step completes the task.
            self.cross_mem.keep_final_step = True

    def end_episode(self, state, score, won=False):
        self._won = won
        super().end_episode(state, score)

    def _detect_victory_from_observation(self, final_observation_text):
        return self._won  # The environment reports success; the text does not.

    def generate_action(self, state_node, info=None):
        history_length = len(self.game_history)
        action, full_response = super().generate_action(state_node, info=info)
        matched = match_action(action, (info or {}).get("valid") if self.args.use_valid_actions else None)
        if matched != action and len(self.game_history) > history_length:
            self.game_history[-1]["action"] = matched
        return matched, full_response

    def get_prompts(self, state_node, info=None):
        info = info or {}
        task = info.get("task", "")
        valid_actions = info.get("valid", []) if self.args.use_valid_actions else None
        inventory = info.get("inv") or None

        if self.game_history:
            summary, memory_text = summarize_progress(
                self.game_history, state_node.state, task, self.args.llm_model, inventory=inventory)
        else:
            summary, memory_text = "", "No actions taken yet."
            if inventory:
                memory_text += f"\n\n[INVENTORY]\n{inventory}"

        sys_prompt = ENV_RULES
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"
        if valid_actions:
            sys_prompt += ("\n\n**CRITICAL CONSTRAINT**: You MUST ONLY choose actions from the ADMISSIBLE ACTIONS list. "
                           "Any other action fails. The list is unordered; position does not indicate quality.")

        valid_actions_text = ""
        if valid_actions:
            valid_actions = valid_actions[:]
            random.Random(time.time_ns()).shuffle(valid_actions)
            valid_actions_text = (f"\nADMISSIBLE ACTIONS (the only valid actions now):\n{valid_actions}\n"
                                  "Copy each option EXACTLY from this list.\n")

        source = "from the ADMISSIBLE ACTIONS list" if valid_actions else "as short ALFWorld commands"
        verbalized = getattr(self.args, "confidence_mode", "logit") == "verbalized"
        fields = []
        for i in range(1, self.args.top_actions + 1):
            fields.append(f'    "reasoning{i}": "Why this action helps",')
            fields.append(f'    "option{i}": "action command",')
            if verbalized:
                fields.append(f'    "confidence{i}": 80,')
        option_format = "\n".join(fields)
        confidence_rules = ""
        if verbalized:
            confidence_rules = (
                f"- The confidence values (confidence1 + ... + confidence{self.args.top_actions}) MUST sum to 100, "
                "with meaningful differences between them\n"
                "- Higher confidence means the action is more likely to move the task forward\n")

        user_prompt = f"""YOUR TASK: {task}

PROGRESS SO FAR:
{summary}
{memory_text}

RECENT STEPS:
{format_recent_steps(self.game_history)}
CURRENT OBSERVATION: {state_node.state}

INSTRUCTIONS:
1. Analyze your progress: which subgoals of the task are done, and what is next?
2. Propose {self.args.top_actions} different actions {source}, with reasoning
{"3. Give each action a confidence from 0 to 100" if verbalized else ""}

RESPONSE FORMAT (JSON):
{{
    "progress_analysis": "What is done and what remains",
    "next_objective": "Your next subgoal",
{option_format}
    "best_action": 1
}}

RULES:
- Every option MUST be {"copied exactly from the ADMISSIBLE ACTIONS list" if valid_actions else "a short ALFWorld command"}
{confidence_rules}- Do not repeat actions that changed nothing, and do not move back and forth between the same places
- Do not re-check receptacles you already searched unless something changed
{valid_actions_text}"""

        if self.prior_memory is not None:
            action_types = self.prior_memory.action_types(state_node.state)
            if action_types:
                user_prompt += (
                    "\nPrior memory action types: " + ", ".join(action_types) + ".\n"
                    "Consider these types when you propose actions. Use objects from the current state. "
                    "If admissible actions are supplied, select only those actions.\n")
        return sys_prompt, user_prompt, memory_text
