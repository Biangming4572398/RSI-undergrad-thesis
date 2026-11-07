import contextlib
import importlib.util
import io
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from jitrl_prior import PrototypeMemory


class JerichoPriorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Replace API dependencies only; run the actual agent methods.
        modules = {}
        for name, exports in {
            'openai_helpers': ['chat_completion_with_retries'],
            'cross_episode_memory': ['CrossEpisodeMemory'],
            'utils': ['generate_trajectory_summary', 'generate_history_summary'],
        }.items():
            module = ModuleType('Jericho.src.' + name)
            for export in exports:
                setattr(module, export, Mock())
            modules[module.__name__] = module
        path = Path(__file__).resolve().parents[1] / 'Jericho/src/jitrl_agent.py'
        spec = importlib.util.spec_from_file_location('Jericho.src.jitrl_agent', path)
        module = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, modules):
            spec.loader.exec_module(module)
        cls.Agent = module.JitRLAgent

    def prior(self, key=None):
        return PrototypeMemory([
            dict(key=key or ['door', 'locked'], action_type='open', value=1.0, count=1, domain='source'),
            dict(key=key or ['door', 'locked'], action_type='wait', value=0.0, count=1, domain='source'),
        ], kappa=2)

    def agent(self, prior=None, rows=()):
        agent = self.Agent.__new__(self.Agent)
        agent.enable_cross_mem = True
        agent.prior_memory = prior
        agent.cross_mem = SimpleNamespace(
            retrieve_similar=Mock(return_value=list(rows)), current_episode_number=10,
        )
        agent.game_history = []
        agent.exploration_rate = 0.65
        agent.exploration_alpha = 1.0
        return agent

    def score(self, agent, options):
        with contextlib.redirect_stdout(io.StringIO()):
            return agent.update_scores(SimpleNamespace(state='locked door'), options, 10, 0.9, '')

    def test_prior_corrects_cold_start_without_exploration(self):
        agent = self.agent(self.prior())
        options = {
            1: {'action': 'open door', 'normalized_prob': 0.1},
            2: {'action': 'wait', 'normalized_prob': 0.9},
        }
        with patch('random.random', side_effect=AssertionError('No prior exploration bonus')):
            result = self.score(agent, options)
        self.assertAlmostEqual(result[1]['avg_reward'], 1.0)
        self.assertAlmostEqual(result[1]['raw_advantage'], 0.5)
        self.assertGreater(result[1]['corrected_logprob'], result[2]['corrected_logprob'])
        self.assertEqual({v['action'] for v in result.values()}, {'open door', 'wait'})

    def test_zero_kappa_and_no_match_keep_legacy_scores(self):
        rows = [(1, reward, {'action': action, 'discounted_reward': reward})
                for action, reward in [('take key', 2), ('take key', 4), ('open door', 0)]]
        disabled = PrototypeMemory.from_args(SimpleNamespace(
            prior_kappa=0, prior_path='/missing/prior.json', prior_return_scale=10,
        ))
        self.assertIsNone(disabled)
        for prior in (disabled, self.prior(['ocean', 'ship'])):
            with self.subTest(prior=prior):
                options = {
                    1: {'action': 'take key', 'normalized_prob': 0.1},
                    2: {'action': 'open door', 'normalized_prob': 0.8},
                    3: {'action': 'look', 'normalized_prob': 0.1},
                }
                with patch('random.random', return_value=0.99):
                    result = self.score(self.agent(prior, rows), options)
                for number, expected in [(1, 1.2), (2, 0.25), (3, -0.45)]:
                    self.assertAlmostEqual(result[number]['corrected_logprob'], expected)

    def test_no_evidence_keeps_base_choice(self):
        options = {1: {'action': 'wait', 'normalized_prob': 1}}
        self.assertEqual(self.score(self.agent(), options), {})
        self.assertEqual(self.score(self.agent(self.prior(['ocean'])), options), {})

    def test_memory_disable_does_not_load_prior(self):
        agent = self.Agent(SimpleNamespace(enable_cross_mem=False, prior_kappa=2, prior_path='/missing'))
        self.assertIsNone(agent.prior_memory)
        self.assertEqual(self.score(agent, {}), {})

    def test_positive_local_action_is_still_a_candidate(self):
        rows = [(1, 1, {'action': 'take key', 'discounted_reward': 1})]
        result = self.score(self.agent(self.prior(), rows), {
            1: {'action': 'open door', 'normalized_prob': 1},
        })
        self.assertEqual({v['action'] for v in result.values()}, {'open door', 'take key'})

    def test_prompt_types_use_current_objects(self):
        agent = self.agent(self.prior())
        agent.guiding_prompt = ''
        agent.args = SimpleNamespace(use_valid_actions=True, confidence_mode='verbalized', top_actions=2)
        _, prompt, _ = agent.get_prompts(SimpleNamespace(state='locked door'))
        self.assertIn('Prior memory action types:', prompt)
        self.assertIn('Use objects from the current state.', prompt)


if __name__ == '__main__':
    unittest.main()
