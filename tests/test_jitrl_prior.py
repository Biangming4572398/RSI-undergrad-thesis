import argparse
import json
from pathlib import Path
import tempfile
import unittest

from jitrl_prior import (PrototypeMemory, action_type, add_prior_arguments,
                         build_prior, save_prior, shared_key)


def row(kind, value, count=1, domain="source"):
    return {"key": ["door", "room"], "action_type": kind,
            "value": value, "count": count, "domain": domain}


def episode(steps, webarena=False):
    result = {"timestamp": "test", "num_steps": len(steps), "final_score": 0,
              "success": True, "steps": steps}
    if webarena:
        result["task_goal"] = "Open the door"
        result["llm_analysis"] = ""
        for step in steps:
            step["normalized_action"] = step["action"]
    return result


class PrototypeMemoryTests(unittest.TestCase):
    def test_cold_start_and_action_hints(self):
        memory = PrototypeMemory([row("open", 1, 3), row("look", 0)], kappa=2)
        values, state = memory.estimate("room door", {}, ["open door", "look", "wait"])
        self.assertEqual(values, {"open door": 1, "look": 0, "wait": 0.75})
        self.assertEqual(state, 0.75)
        self.assertEqual(memory.action_types("room door"), ["look", "open"])

    def test_state_and_action_counts_are_separate(self):
        memory = PrototypeMemory([row("open", 1), row("look", 0)], kappa=2)
        values, state = memory.estimate("room", {"open door": [0, 0], "look": [1]},
                                        ["open door", "open window", "look"])
        self.assertAlmostEqual(state, 0.4)
        self.assertAlmostEqual(values["open door"], 0.5)
        self.assertEqual(values["open window"], 1)
        self.assertAlmostEqual(values["look"], 1 / 3)

    def test_missing_action_keeps_local_mean_or_zero_advantage(self):
        memory = PrototypeMemory([row("open", 1)], kappa=2)
        values, state = memory.estimate("door", {"wait": [0.2]}, ["wait", "look"])
        self.assertAlmostEqual(values["wait"], 0.2)
        self.assertEqual(values["look"], state)

    def test_prior_weight_falls_with_local_evidence(self):
        memory = PrototypeMemory([row("open", 1)], kappa=2)
        early, _ = memory.estimate("door", {"open door": [0]}, ["open door"])
        late, _ = memory.estimate("door", {"open door": [0] * 20}, ["open door"])
        self.assertAlmostEqual(early["open door"], 2 / 3)
        self.assertAlmostEqual(late["open door"], 2 / 22)

    def test_calibration(self):
        memory = PrototypeMemory([row("open", 1)], kappa=2, return_scale=10)
        values, state = memory.estimate("door", {"open door": [10]}, ["open door"])
        self.assertEqual(values["open door"], 1)
        self.assertEqual(state, 1)

    def test_keys_ignore_urls_case_and_digits(self):
        self.assertEqual(shared_key("ROOM door 12 https://site.test/private/abc"), {"room", "door"})
        self.assertEqual(action_type('click(<button[Submit]>)'), "click")
        self.assertEqual(action_type('fill("42", "private text")'), "fill")

    def test_unicode_state_keys_roundtrip(self):
        key = sorted(shared_key('Room 10 m²'))
        self.assertEqual(key, ['m', 'room'])
        data = {'version': 1, 'gamma': 0.95, 'rows': [{**row('open', 1), 'key': key}]}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'prior.json'
            save_prior(data, target)
            memory = PrototypeMemory.from_args(argparse.Namespace(prior_path=target, prior_kappa=2))
            self.assertEqual(memory.action_types('Room 20 m²'), ['open'])

    def test_no_match_and_disabled_prior(self):
        memory = PrototypeMemory([row("open", 1)], kappa=2)
        self.assertIsNone(memory.estimate("ocean", {}, ["open door"]))
        self.assertEqual(memory.action_types("ocean"), [])
        args = argparse.Namespace(prior_kappa=0, prior_path="does-not-exist")
        self.assertIsNone(PrototypeMemory.from_args(args))
        self.assertIsNone(PrototypeMemory.from_args(argparse.Namespace()))

    def test_invalid_configuration_and_rows(self):
        for kappa in (-1, float("nan"), float("inf")):
            with self.subTest(kappa=kappa), self.assertRaises(ValueError):
                PrototypeMemory([row("open", 1)], kappa=kappa)
        for scale in (0, -1, float("nan"), float("inf")):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                PrototypeMemory([row("open", 1)], kappa=1, return_scale=scale)
        for invalid in (row("open", float("nan")), row("open", 1, 0), row("open", 1, True),
                        {**row("open", 1), "key": ["https://example.test"]},
                        {**row("open", 1), "action_type": "open(private)"}):
            with self.subTest(row=invalid), self.assertRaises(ValueError):
                PrototypeMemory([invalid], kappa=1)
        with self.assertRaises(ValueError):
            PrototypeMemory.from_args(argparse.Namespace(prior_kappa=1))
        with self.assertRaises(ValueError):
            PrototypeMemory([row("open", 1)], kappa=1).estimate(
                "door", {"open": [float("inf")]}, ["open"])

    def test_parser_defaults(self):
        parser = argparse.ArgumentParser()
        add_prior_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.prior_kappa, 0)
        self.assertEqual(args.prior_return_scale, 1)

    def test_native_return_rules_roundtrip_consolidation_and_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            jericho = Path(directory) / "jericho.jsonl"
            webarena = Path(directory) / "webarena.jsonl"
            steps = [{"state": "room door", "action": "open door", "llm_step_score": 2, "reward": 7},
                     {"state": "room door", "action": "look", "llm_step_score": 0, "reward": 4},
                     {"state": "room door", "action": "wait", "llm_step_score": 0, "reward": 0}]
            jericho.write_text(json.dumps(episode(steps)) + "\n" + json.dumps(episode(steps)) + "\n")
            webarena.write_text(json.dumps(episode(steps, webarena=True)) + "\n")
            sources = [("game", 2, jericho), ("site", 2, webarena)]
            data = build_prior(sources, gamma=0.5)
            values = {(item["domain"], item["action_type"]): item for item in data["rows"]}
            self.assertEqual(values["game", "open"]["value"], 2.25)
            self.assertEqual(values["game", "open"]["count"], 2)
            self.assertEqual(values["site", "open"]["value"], 1.75)
            self.assertEqual(values["site", "look"]["value"], 0)
            heldout = build_prior(sources, gamma=0.5, exclude_domain="game")
            self.assertTrue(all(item["domain"] == "site" for item in heldout["rows"]))
            self.assertEqual([item["domain"] for item in heldout["sources"]], ["site"])
            target = Path(directory) / "prior.json"
            save_prior(data, target, [jericho, webarena])
            args = argparse.Namespace(prior_path=str(target), prior_kappa=2, gamma=0.5)
            loaded = PrototypeMemory.from_args(args)
            self.assertEqual(len(loaded.rows), 6)
            args.gamma = 0.1
            with self.assertRaisesRegex(ValueError, "gamma"):
                PrototypeMemory.from_args(args)
            del args.gamma
            with self.assertRaisesRegex(ValueError, "gamma"):
                PrototypeMemory.from_args(args)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                save_prior(data, jericho, [jericho])
            self.assertEqual(len(jericho.read_text().splitlines()), 2)

    def test_builder_rejects_unknown_schema_and_nonfinite_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episodes.jsonl"
            for data in ({"steps": []}, episode([{"state": "room", "action": "look",
                                                  "reward": float("nan")}])):
                source.write_text(json.dumps(data) + "\n")
                with self.assertRaises(ValueError):
                    build_prior([("game", 1, source)])

    def test_webarena_null_scores_and_unused_environment_rewards(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "episodes.jsonl"
            data = episode([{"state": "room", "action": "click('1')",
                             "llm_step_score": None, "reward": None}], webarena=True)
            source.write_text(json.dumps(data) + "\n")
            result = build_prior([("site", 1, source)])
            self.assertEqual(result["rows"][0]["value"], 0)


if __name__ == "__main__":
    unittest.main()
