"""Memoryless ALFWorld baseline: one LLM call per step with the task, recent steps and admissible actions."""

from Jericho.src.openai_helpers import chat_completion_with_retries

from .prompts import ENV_RULES, format_recent_steps, match_action


class ALFWorldNaiveAgent:
    def __init__(self, args, guiding_prompt=None):
        self.args = args
        self.guiding_prompt = guiding_prompt or ""
        self.game_history = []

    def start_episode(self):
        self.game_history = []

    def end_episode(self, state, score, won=False):
        print(f"Ending episode. Won: {won}.")

    def update_game_history_reward(self, reward, score):
        pass

    def generate_action(self, state_node, info=None):
        info = info or {}
        valid = info.get("valid") if self.args.use_valid_actions else None
        sys_prompt = ENV_RULES
        if self.guiding_prompt:
            sys_prompt += f"\n\nFollow this guide: {self.guiding_prompt}"
        user_prompt = f"""YOUR TASK: {info.get('task', '')}

RECENT STEPS:
{format_recent_steps(self.game_history, limit=10)}
CURRENT OBSERVATION: {state_node.state}
{f"ADMISSIBLE ACTIONS: {valid}" if valid else ""}

Choose the next action{" exactly from the admissible actions" if valid else ""}. Your response MUST strictly follow this format and include nothing else:
REASONING: [one or two sentences]
ACTION: [the action]"""

        response = chat_completion_with_retries(
            model=self.args.llm_model, sys_prompt=sys_prompt, prompt=user_prompt,
            max_tokens=300, temperature=self.args.llm_temperature)
        full_response = ""
        action = "look"
        if response and getattr(response, "choices", None) and response.choices[0].message:
            full_response = response.choices[0].message.content or ""
            for line in full_response.strip().splitlines():
                if line.upper().startswith("ACTION:"):
                    action = line.split(":", 1)[1].strip() or action
        else:
            print("Warning: LLM API call failed or returned nothing. Defaulting to 'look'.")
        action = match_action(action, valid)
        self.game_history.append({"state": state_node.state, "action": action})
        return action, full_response
