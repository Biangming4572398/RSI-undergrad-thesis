import json
import os
import random
import statistics
import time
from collections import defaultdict

import numpy as np

from .env import ALFWorldEnv, collect_games, parse_task_types, select_games, task_type_of


class StateNode:
    def __init__(self, state, reward=0.0):
        self.state = state
        self.reward = reward
        self.response = ""


def make_agent(args):
    if args.agent_type == "jitrl":
        from .jitrl_agent import ALFWorldJitRLAgent
        return ALFWorldJitRLAgent(args)
    if args.agent_type == "naive":
        from .naive_agent import ALFWorldNaiveAgent
        return ALFWorldNaiveAgent(args)
    raise ValueError(f"Unknown agent type: {args.agent_type}.")


def success_rates(episodes, key):
    groups = defaultdict(list)
    for episode in episodes:
        groups[episode[key]].append(episode["won"])
    return {str(name): statistics.mean(values) for name, values in sorted(groups.items())}


class ALFWorldEvaluator:
    """Runs every selected game once per round. One agent (and one memory) sees all episodes in order."""

    def __init__(self, args):
        self.args = args
        # JitRL keeps its memory in output_path/game_name/agent_type/model.
        self.args.game_name = "alfworld"
        if self.args.seed is None:
            self.args.seed = random.randint(0, 100000)
        np.random.seed(self.args.seed)
        random.seed(self.args.seed)

        games = collect_games(self.args.data_path, self.args.split)
        ids = {int(item) for item in self.args.games.split(",")} if self.args.games else None
        self.games = select_games(games, ids=ids, start=self.args.start, end=self.args.end,
                                  task_types=parse_task_types(self.args.task_types))
        if not self.games:
            raise ValueError("No ALFWorld games match the selection.")
        if self.args.shuffle:
            random.Random(self.args.seed).shuffle(self.games)

    def run_evaluation(self, agent=None):
        args = self.args
        model_slug = args.llm_model.replace("/", "_").replace("\\", "_")
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        self.log_dir = os.path.join(args.output_path, args.game_name, args.agent_type, model_slug, timestamp)
        os.makedirs(self.log_dir, exist_ok=True)
        summary_path = os.path.join(self.log_dir, f"evaluation_summary_{args.agent_type}_{model_slug}_{timestamp}.json")

        print("\n--- STARTING ALFWORLD EVALUATION ---")
        print(f"Split: {args.split}, Games: {len(self.games)}, Rounds: {args.rounds}")
        print(f"Agent Type: {args.agent_type}, Agent LLM Model: {args.llm_model}, Seed: {args.seed}")
        print(f"Episode logs and summary will be in: {self.log_dir}")

        agent = agent or make_agent(args)
        episodes = []
        summary = {}
        for round_index in range(1, args.rounds + 1):
            for game_id, game_file in self.games:
                episode_num = len(episodes)
                print(f"\nRound {round_index}/{args.rounds}, episode {episode_num + 1}/{len(self.games) * args.rounds}, "
                      f"game {game_id} ({task_type_of(game_file)})")
                result = self.run_episode(agent, game_id, game_file, episode_num)
                result["round"] = round_index
                episodes.append(result)
                print(f"Game {game_id} finished. Won: {result['won']} in {result['steps']} steps.")

                summary = {
                    "split": args.split,
                    "num_games": len(self.games),
                    "rounds": args.rounds,
                    "episodes_completed": len(episodes),
                    "success_rate": statistics.mean(e["won"] for e in episodes),
                    "success_rate_by_round": success_rates(episodes, "round"),
                    "success_rate_by_task_type": success_rates(episodes, "task_type"),
                    "average_steps": statistics.mean(e["steps"] for e in episodes),
                    "episodes": episodes,
                    "timestamp_evaluation_session_start": timestamp,
                    "timestamp_summary_created": time.strftime("%Y%m%d-%H%M%S"),
                    "all_args": vars(args),
                    "success": True,
                }
                with open(summary_path, "w", encoding="utf-8") as output:
                    json.dump(summary, output, indent=4)
                print(f"Success rate so far: {summary['success_rate']:.3f} "
                      f"({sum(e['won'] for e in episodes)}/{len(episodes)})")

        print("\n--- ALFWORLD EVALUATION SUMMARY ---")
        print(f"Success rate: {summary['success_rate']:.3f}")
        print(f"By round: {summary['success_rate_by_round']}")
        print(f"By task type: {summary['success_rate_by_task_type']}")
        return summary

    def run_episode(self, agent, game_id, game_file, episode_num):
        args = self.args
        log_file_path = os.path.join(self.log_dir, f"episode_{episode_num:04d}_game_{game_id:04d}.txt")
        env = ALFWorldEnv(game_file, step_limit=args.env_step_limit)
        try:
            ob, info = env.reset()
            step = -1
            with open(log_file_path, "w", encoding="utf-8") as output:
                output.write(f"Game {game_id}: {game_file}\nTask type: {info['task_type']}\nTask: {info['task']}\n"
                             f"Model: {args.llm_model}, Episode: {episode_num}\n")
                agent.start_episode()
                output.write(f"Guiding prompt: {agent.guiding_prompt}\n")
                for step in range(args.env_step_limit):
                    action, raw_llm_output = agent.generate_action(StateNode(state=ob), info=info)
                    output.write("==========\n")
                    output.write(f"[STEP] {step}\n")
                    output.write("----------\n")
                    output.write(f"[OBS] {ob}\n")
                    output.write("----------\n")
                    output.write(f"[INV] {info.get('inv', 'N/A')}\n")
                    if args.use_valid_actions:
                        output.write(f"[VALID_ACTIONS] {info['valid']}\n")
                    output.write(f"[RAW_LLM_OUTPUT] {raw_llm_output}\n")
                    output.write(f"[CHOSEN_ACTION] {action}\n")
                    output.write("----------\n")
                    prev_ob = ob
                    ob, reward, done, info = env.step(action)
                    output.write(f"[REWARD] {reward}\n")
                    output.write(f"[CUM_REWARD] {info['score']}\n")
                    output.write("----------\n")
                    agent.update_game_history_reward(reward, info["score"])
                    short = prev_ob[:70].replace("\n", " ").ljust(70)
                    print(f"[STEP {step}] | {short} || Action: {action.ljust(30)} || Reward: {reward}")
                    if done:
                        break
                won = info["won"]
                ending = "Task completed" if won else "Step limit reached"
                output.write(f"\n{ending} at step {step}. Final observation: {ob}\n")
                agent.end_episode(state=ob, score=info["score"], won=won)
        finally:
            env.close()
        return {
            "episode": episode_num,
            "game_id": game_id,
            "task_type": info["task_type"],
            "task": info["task"],
            "won": bool(won),
            "steps": step + 1,
            "score": info["score"],
            "game_file": game_file,
            "log_file": log_file_path,
        }
