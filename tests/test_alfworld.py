import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ALFWorld.src.env import DEFAULT_DATA_PATH, collect_games, parse_task_types, select_games
from ALFWorld.src.prompts import match_action

HAS_ALFWORLD = all(importlib.util.find_spec(name) for name in ("alfworld", "textworld"))
HAS_GAMES = (DEFAULT_DATA_PATH / "json_2.1.1" / "valid_unseen").is_dir()


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def args(output_path, **overrides):
    values = dict(
        agent_type="jitrl", llm_model="test/model", eval_llm_model="test/model", llm_temperature=0,
        top_actions=3, max_memory=30, gamma=0.5, exploration_rate=0.65, use_history_prompt=True,
        retrieval_top_k=10, retrieval_threshold=0.95, enable_cross_mem=True, update_guiding_prompt=False,
        use_valid_actions=True, confidence_mode="verbalized", prior_path=None, prior_kappa=0.0,
        prior_return_scale=1.0, output_path=str(output_path), data_path=str(DEFAULT_DATA_PATH),
        split="valid_unseen", task_types=None, games=None, start=None, end=None, rounds=1,
        shuffle=False, env_step_limit=50, seed=0)
    values.update(overrides)
    return SimpleNamespace(**values)


def completion(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text))])


def action_json(action):
    return completion(json.dumps({
        "progress_analysis": "", "next_objective": "",
        "reasoning1": "", "option1": action, "confidence1": 90,
        "reasoning2": "", "option2": "look", "confidence2": 5,
        "reasoning3": "", "option3": "inventory", "confidence3": 5,
        "best_action": 1}))


def fake_embedding(text, **kwargs):
    seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
    vector = np.random.default_rng(seed).normal(size=1536).astype(np.float32)
    return vector / np.linalg.norm(vector)


def plan(game_file):
    """The game's planner solution, from TextWorld."""
    import textworld
    import textworld.gym
    from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos
    infos = textworld.EnvInfos(policy_commands=True, extras=["gamefile"])
    env = textworld.gym.make(textworld.gym.register_game(
        game_file, infos, wrappers=[AlfredDemangler(shuffle=False), AlfredInfos]))
    try:
        return env.reset()[1]["policy_commands"]
    finally:
        env.close()


class GameSelectionTests(unittest.TestCase):
    def make_split(self, root, games):
        for name, solvable in games.items():
            path = root / "json_2.1.1" / "valid_seen" / name / "trial_1" / "game.tw-pddl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({"solvable": solvable}))

    def test_collect_games_filters_like_alfworld(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_split(root, {
                "pick_and_place_simple-Book-None-Desk-1": True,
                "pick_and_place_simple-Book-None-Desk-2": False,
                "pick_and_place_with_movable_recep-Apple-Bowl-Desk-3": True,
                "pick_cool_then_place_in_recep-AppleSliced-None-Fridge-4": True,
                "look_at_obj_in_light-Book-None-DeskLamp-5": True,
            })
            games = collect_games(root, "valid_seen")
            self.assertEqual([Path(game).parts[-3] for game in games], [
                "look_at_obj_in_light-Book-None-DeskLamp-5", "pick_and_place_simple-Book-None-Desk-1"])
            with self.assertRaises(FileNotFoundError):
                collect_games(root, "valid_unseen")

    def test_select_games_keeps_split_ids(self):
        games = [f"/data/{kind}-X-None-Y-1/trial/game.tw-pddl" for kind in (
            "look_at_obj_in_light", "pick_and_place_simple", "pick_and_place_simple", "pick_two_obj_and_place")]
        self.assertEqual([i for i, _ in select_games(games, task_types={"pick_and_place_simple"})], [1, 2])
        self.assertEqual([i for i, _ in select_games(games, ids={0, 3})], [0, 3])
        self.assertEqual([i for i, _ in select_games(games, start=1, end=3)], [1, 2])

    def test_parse_task_types(self):
        self.assertEqual(parse_task_types("1, pick_cool_then_place_in_recep"),
                         {"pick_and_place_simple", "pick_cool_then_place_in_recep"})
        self.assertIsNone(parse_task_types(None))
        with self.assertRaises(ValueError):
            parse_task_types("7")


class MatchActionTests(unittest.TestCase):
    valid = ["go to fridge 1", "move apple 1 to fridge 1", "look"]

    def test_format_fixes(self):
        self.assertEqual(match_action("Go to the Fridge 1.", self.valid), "go to fridge 1")
        self.assertEqual(match_action("put apple 1 in/on fridge 1", self.valid), "move apple 1 to fridge 1")

    def test_no_guessing(self):
        self.assertEqual(match_action("go to fridge 2", self.valid), "go to fridge 2")
        self.assertEqual(match_action("Go to fridge 2", None), "Go to fridge 2")


class AgentTests(unittest.TestCase):
    def test_prompt_has_task_and_admissible_actions(self):
        from ALFWorld.src.jitrl_agent import ALFWorldJitRLAgent
        with tempfile.TemporaryDirectory() as directory, quiet():
            agent = ALFWorldJitRLAgent(args(directory, enable_cross_mem=False))
            info = {"task": "put a mug in desk.", "valid": ["go to desk 1", "look"], "inv": ""}
            _, prompt, _ = agent.get_prompts(SimpleNamespace(state="You are in the middle of a room."), info)
        self.assertIn("YOUR TASK: put a mug in desk.", prompt)
        self.assertIn("go to desk 1", prompt)
        self.assertIn('"confidence3"', prompt)

    def test_action_is_matched_and_recorded(self):
        from ALFWorld.src import jitrl_agent
        with tempfile.TemporaryDirectory() as directory, quiet():
            agent = jitrl_agent.ALFWorldJitRLAgent(args(directory, enable_cross_mem=False))
            agent.start_episode()
            with patch("Jericho.src.jitrl_agent.chat_completion_with_retries",
                       return_value=action_json("Go to the desk 1.")):
                action, _ = agent.generate_action(SimpleNamespace(state="room"), {"task": "t", "valid": ["go to desk 1"]})
        self.assertEqual(action, "go to desk 1")
        self.assertEqual(agent.game_history[-1]["action"], "go to desk 1")

    def test_failed_api_call_falls_back_to_look(self):
        from ALFWorld.src import jitrl_agent
        with tempfile.TemporaryDirectory() as directory, quiet():
            agent = jitrl_agent.ALFWorldJitRLAgent(args(directory, enable_cross_mem=False))
            agent.start_episode()
            with patch("Jericho.src.jitrl_agent.chat_completion_with_retries", return_value={}):
                action, _ = agent.generate_action(SimpleNamespace(state="room"), {"task": "t", "valid": ["look"]})
        self.assertEqual(action, "look")

    def test_memory_keeps_final_step_only_when_asked(self):
        from Jericho.src.cross_episode_memory import CrossEpisodeMemory
        history = [{"state": f"s{i}", "action": f"a{i}", "reward": 0, "score": 0} for i in range(3)]
        for keep, expected in ((False, 2), (True, 3)):
            with tempfile.TemporaryDirectory() as directory, quiet(), \
                    patch("Jericho.src.cross_episode_memory.evaluate_step_scores_with_llm",
                          return_value=([1, 1, 1], ["", "", ""])), \
                    patch.object(CrossEpisodeMemory, "_store_step_in_vector_db"):
                memory = CrossEpisodeMemory(directory)
                memory.keep_final_step = keep
                memory.add_episode(history, state="won", final_score=1, success=True)
                self.assertEqual(len(memory.load_episodes()[0]["steps"]), expected)


@unittest.skipUnless(HAS_ALFWORLD and HAS_GAMES, "needs the alfworld extra and ALFWorld/data")
class RealGameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.games = collect_games(DEFAULT_DATA_PATH, "valid_unseen")

    def test_env_wins_with_planner_commands(self):
        from ALFWorld.src.env import ALFWorldEnv
        env = ALFWorldEnv(self.games[0], step_limit=50)
        try:
            ob, info = env.reset()
            self.assertTrue(info["task"])
            self.assertNotIn("help", info["valid"])
            self.assertIn(info["task"], info["look"])
            self.assertEqual(info["inv"], "You are not carrying anything.")
            steps = plan(self.games[0])
            for action in steps:
                self.assertIn(action, info["valid"])
                ob, reward, done, info = env.step(action)
            self.assertTrue(done and info["won"])
            self.assertEqual((reward, info["score"]), (1.0, 1.0))
        finally:
            env.close()

    def test_env_step_limit(self):
        from ALFWorld.src.env import ALFWorldEnv
        env = ALFWorldEnv(self.games[0], step_limit=2)
        try:
            env.reset()
            self.assertFalse(env.step("look")[2])
            _, reward, done, info = env.step("look")
            self.assertTrue(done)
            self.assertFalse(info["won"])
            self.assertEqual(reward, 0.0)
        finally:
            env.close()

    def test_jitrl_pipeline_stores_won_episodes(self):
        from ALFWorld.src.evaluation import ALFWorldEvaluator
        game_id = 0
        commands = plan(self.games[game_id])
        actions = iter(commands * 2)
        # random.random=0.99 turns off the exploration bonus, so round 2 follows the remembered plan.
        with tempfile.TemporaryDirectory() as directory, quiet(), patch("random.random", return_value=0.99), \
                patch("Jericho.src.jitrl_agent.chat_completion_with_retries",
                      side_effect=lambda **kw: action_json(next(actions))), \
                patch("ALFWorld.src.prompts.chat_completion_with_retries",
                      return_value=completion("[SUMMARY]\nworking\n[PROGRESS]\nNo Progress")), \
                patch("Jericho.src.cross_episode_memory.get_embedding_with_retries", side_effect=fake_embedding), \
                patch("Jericho.src.cross_episode_memory.evaluate_step_scores_with_llm",
                      side_effect=lambda game_history, **kw: ([5] * len(game_history), [""] * len(game_history))):
            summary = ALFWorldEvaluator(args(directory, games=str(game_id), rounds=2)).run_evaluation()
            self.assertEqual(summary["success_rate"], 1.0)
            self.assertEqual(summary["success_rate_by_round"], {"1": 1.0, "2": 1.0})
            self.assertEqual([e["steps"] for e in summary["episodes"]], [len(commands)] * 2)
            memory_dir = Path(directory) / "alfworld" / "jitrl" / "test_model"
            episodes = [json.loads(line) for line in (memory_dir / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual([len(e["steps"]) for e in episodes], [len(commands)] * 2)
            self.assertTrue(all(e["success"] for e in episodes))
            self.assertEqual(episodes[0]["steps"][-1]["action"], commands[-1])
            self.assertTrue(Path(summary["episodes"][1]["log_file"]).is_file())

    def test_naive_agent_fails_at_step_limit(self):
        from ALFWorld.src.evaluation import ALFWorldEvaluator
        with tempfile.TemporaryDirectory() as directory, quiet(), \
                patch("ALFWorld.src.naive_agent.chat_completion_with_retries",
                      return_value=completion("REASONING: wait\nACTION: look")):
            summary = ALFWorldEvaluator(args(directory, agent_type="naive", games="0,1",
                                             env_step_limit=3)).run_evaluation()
        self.assertEqual(summary["success_rate"], 0.0)
        self.assertEqual([e["steps"] for e in summary["episodes"]], [3, 3])


if __name__ == "__main__":
    unittest.main()
