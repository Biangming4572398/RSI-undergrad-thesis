"""Check WebArena integration without browser or model services."""

import contextlib
import importlib.util
import io
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
CLICK = "click(<button[Submit]>)"
FILL = 'fill(<textbox[Search]>, "query")'


def module(name, **attributes):
    result = types.ModuleType(name)
    result.__dict__.update(attributes)
    return result


class WebArenaPriorTests(unittest.TestCase):
    def setUp(self):
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        package = "webarena_prior_test"
        self.ground_action = Mock(return_value=None)
        # Stub browser, model, and local memory dependencies to test the real agent.
        stubs = {
            package: module(package, __path__=[]),
            "browsergym": module("browsergym"),
            "browsergym.experiments": module("browsergym.experiments", Agent=object, AbstractAgentArgs=object),
            "browsergym.utils": module("browsergym.utils"),
            "numpy": module("numpy"),  # The score method imports NumPy but does not use it.
            "browsergym.utils.obs": module("browsergym.utils.obs", flatten_axtree_to_str=Mock(), flatten_dom_to_str=Mock(), prune_html=Mock()),
            package + ".utils": module(package + ".utils", __path__=[]),
            package + ".utils.openai_helpers": module("helpers", chat_completion_with_retries=Mock(), extract_json_from_response=Mock(), TokenLimitExceededError=RuntimeError, _is_openai_model=lambda _: True),
            package + ".utils.cross_episode_memory": module("memory", CrossEpisodeMemory=Mock()),
            package + ".utils.chat_api": module("chat", ChatModelArgs=object),
            package + ".utils.utils": module("utils", normalize_action=Mock(), calculate_action_similarity=Mock(), denormalize_action=self.ground_action),
            package + ".dynamic_prompting": module("prompting", Flags=object, ActionSpace=lambda _: types.SimpleNamespace(prompt="click(bid)")),
        }
        self.stack.enter_context(patch.dict(sys.modules, stubs))
        spec = importlib.util.spec_from_file_location(package + ".jitrl_agent", ROOT / "WebArena/memory_agents/jitrl_agent.py")
        self.agent_module = importlib.util.module_from_spec(spec)
        self.stack.enter_context(patch.dict(sys.modules, {spec.name: self.agent_module}))
        spec.loader.exec_module(self.agent_module)
        self.args = types.SimpleNamespace(
            llm_model="test", agent_type="memory", eval_runs=1, seed=0,
            top_actions=2, prior_kappa=0, prior_path="missing-prior.json",
        )
        self.agent = self.agent_module.BrowserGymJitRLAgent(self.args)
        self.agent.cross_mem.retrieve_similar.return_value = []
        self.state = self.agent_module.StateNode("[1] button 'Submit'\n[2] textbox 'Search'", "Submit the search")

    def options(self):
        return {
            1: {"action": "click('1')", "normalized_action": CLICK, "normalized_prob": 0.8},
            2: {"action": "fill('2', 'query')", "normalized_action": FILL, "normalized_prob": 0.2},
        }

    def score(self):
        return self.agent.update_scores(self.state, self.options(), 10, 0.8, "")

    def prior(self, key=None):
        return self.agent_module.PrototypeMemory([
            dict(key=key or ["button", "submit"], action_type=kind, value=value, count=1, domain="source")
            for kind, value in [("click", 4), ("fill", 0)]
        ], kappa=2)

    def local_rows(self):
        self.agent.cross_mem.retrieve_similar.return_value = [
            (1, reward, {"action": "click('old')", "normalized_action": CLICK, "discounted_reward": reward})
            for reward in (4, 2)
        ]

    def test_prior_scores_candidates_with_no_local_memory(self):
        prior = self.prior()
        prior.estimate = Mock(wraps=prior.estimate)
        self.agent.prior_memory = prior
        with patch.object(self.agent_module.random, "random", side_effect=AssertionError("Prior must not use exploration")):
            scores = self.score()
        self.assertEqual(scores[1]["avg_reward"], 4)
        self.assertEqual(scores[1]["raw_advantage"], 2)
        self.assertAlmostEqual(scores[1]["corrected_logprob"], 1.8)
        self.assertEqual(scores[2]["action"], "fill('2', 'query')")
        prior.estimate.assert_called_once_with(self.state.state, {}, [CLICK, FILL])

    def test_zero_kappa_keeps_legacy_exploration_and_scores(self):
        self.assertIsNone(self.agent.prior_memory)
        self.assertEqual(self.score(), {})
        self.local_rows()
        with patch.object(self.agent_module.random, "random", return_value=0.02) as rng:
            scores = self.score()
        rng.assert_called_once_with()
        self.assertEqual(scores[1]["avg_reward"], 3)
        self.assertEqual(scores[2]["avg_reward"], 3.5)
        self.assertEqual(scores[1]["raw_advantage"], -0.25)
        self.assertAlmostEqual(scores[1]["corrected_logprob"], -0.2)
        self.assertAlmostEqual(scores[2]["corrected_logprob"], 1.2)

    def test_unmatched_prior_keeps_legacy_result_and_rng_use(self):
        self.local_rows()
        with patch.object(self.agent_module.random, "random", return_value=0.02) as rng:
            expected = self.score()
            self.agent.prior_memory = self.prior(["ocean", "ship"])
            self.assertEqual(self.score(), expected)
            self.assertEqual(rng.call_count, 2)
        self.agent.cross_mem.retrieve_similar.return_value = []
        self.assertEqual(self.score(), {})

    def test_disabled_memory_does_not_load_or_use_prior(self):
        self.args.disable_memory = True
        self.args.prior_kappa = 5
        with patch.object(self.agent_module.PrototypeMemory, "from_args", side_effect=AssertionError("Memory disabled")):
            agent = self.agent_module.BrowserGymJitRLAgent(self.args)
        self.assertIsNone(agent.prior_memory)
        self.assertEqual(agent.update_scores(self.state, self.options(), 10, 0.8, ""), {})

    def test_prior_prompt_uses_action_types_and_current_state(self):
        self.agent.prior_memory = self.prior()
        _, prompt, _ = self.agent.get_prompts(self.state)
        self.assertIn("action types: click, fill", prompt)
        self.assertIn("Use element IDs and arguments from the current state.", prompt)

    def test_unavailable_local_action_is_not_added(self):
        missing_action = "click(<button[Missing]>)"
        self.agent.cross_mem.retrieve_similar.return_value = [
            (1, 5, {"action": "click('old')", "normalized_action": missing_action, "discounted_reward": 5})
        ]
        self.agent.prior_memory = self.prior()
        scores = self.score()
        self.assertEqual(set(scores), {1, 2})
        self.ground_action.assert_called_once_with(missing_action, self.state.state)

    def test_batch_wrapper_passes_prior_arguments_to_run_command(self):
        spec = importlib.util.spec_from_file_location("webarena_prior_batch", ROOT / "WebArena/test_webarena_lite.py")
        batch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(batch)
        args = dict(task_id=0, model="test", max_steps=2, log_file=None, llm_eval=None,
                    prior_path="prior.json", prior_kappa=4, prior_return_scale=0.1)
        with patch.object(batch.subprocess, "run", side_effect=RuntimeError("Stop before browser starts")) as run:
            with self.assertRaisesRegex(RuntimeError, "Stop before browser"):
                batch.run_task_wrapper(args)
        command = run.call_args.args[0]
        for flag, value in [("--prior_path", "prior.json"), ("--prior_kappa", "4"), ("--prior_return_scale", "0.1")]:
            self.assertEqual(command[command.index(flag) + 1], value)


if __name__ == "__main__":
    unittest.main()
