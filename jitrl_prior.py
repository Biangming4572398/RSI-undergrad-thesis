"""A fixed, JSON-backed value prior for JitRL.

Returns are divided by a positive domain scale. Keys and action types are
deterministic prototype rules, not learned representations. Local action keys
remain exact. A missing prior action keeps its local mean, or gets the mixed
state value when there is no local evidence (zero advantage).
"""

import argparse
import json
import math
import os
from pathlib import Path
import re
import tempfile


def _finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _positive(value, name):
    if _finite(value, name) <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def shared_key(state):
    """Remove URLs and numbers, then ignore case for state retrieval."""
    if not isinstance(state, str):
        raise ValueError("state must be text")
    # shortcut: lexical keys are fixed; learn shared keys after transfer evaluation.
    state = re.sub(r"(?:https?://|www\.)\S+", " ", state.casefold())
    return frozenset(''.join(char if char.isalpha() else ' ' for char in state).split())


def action_type(action):
    if not isinstance(action, str):
        raise ValueError("action must be text")
    # shortcut: first verbs define action types; learn mappings after transfer evaluation.
    match = re.match(r"\s*([a-z_]+)", action.casefold())
    return match.group(1) if match else ""


def add_prior_arguments(parser):
    parser.add_argument("--prior_path", help="Path to a version 1 prototype prior JSON file")
    parser.add_argument("--prior_kappa", type=float, default=0.0,
                        help="Prior strength; zero keeps plain JitRL")
    parser.add_argument("--prior_return_scale", type=float, default=1.0,
                        help="Divide local returns by this positive domain scale")


class PrototypeMemory:
    def __init__(self, rows, kappa, return_scale=1.0, top_k=5):
        self.kappa = _finite(kappa, "prior_kappa")
        if kappa < 0:
            raise ValueError("prior_kappa must be nonnegative")
        self.return_scale = _positive(return_scale, "prior_return_scale")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer")
        self.top_k = top_k
        if not isinstance(rows, list):
            raise ValueError("prior rows must be a list")
        self.rows = []
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError("each prior row must be an object")
            key = row.get("key")
            kind = row.get("action_type")
            count = row.get("count")
            if not isinstance(key, list) or not key or any(
                not isinstance(token, str) or not token.isalpha()
                or token != token.casefold() for token in key
            ):
                raise ValueError("row key must contain lowercase word tokens")
            if not isinstance(kind, str) or not re.fullmatch(r"[a-z_]+", kind):
                raise ValueError("row action_type must be a lowercase verb or function name")
            if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
                raise ValueError("row count must be a positive integer")
            if not isinstance(row.get("domain"), str) or not row["domain"].strip():
                raise ValueError("row domain must be nonempty text")
            self.rows.append({**row, "key": frozenset(key),
                              "value": _finite(row.get("value"), "row value")})

    @classmethod
    def from_args(cls, args):
        kappa = _finite(getattr(args, "prior_kappa", 0.0), "prior_kappa")
        scale = _positive(getattr(args, "prior_return_scale", 1.0), "prior_return_scale")
        if kappa < 0:
            raise ValueError("prior_kappa must be nonnegative")
        if kappa == 0:
            return None
        path = getattr(args, "prior_path", None)
        if not path:
            raise ValueError("positive prior_kappa requires prior_path")
        with open(path, encoding="utf-8") as source:
            data = json.load(source)
        if not isinstance(data, dict) or type(data.get("version")) is not int or data["version"] != 1:
            raise ValueError("prior must use JSON version 1")
        gamma = _finite(data.get("gamma"), "prior gamma")
        if not 0 <= gamma <= 1:
            raise ValueError("prior gamma must be between zero and one")
        if gamma != _finite(getattr(args, "gamma", 0.95), "gamma"):
            raise ValueError("prior gamma must equal the local memory gamma")
        return cls(data.get("rows"), kappa, scale, getattr(args, "retrieval_top_k", 5))

    def _neighbours(self, state):
        key = shared_key(state)
        ranked = []
        for row in self.rows:
            overlap = key & row["key"]
            if overlap:
                ranked.append((len(overlap) / len(key | row["key"]), row))
        ranked.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in ranked[:self.top_k]]

    def action_types(self, state):
        """Return type hints only; callers must bind them to the current state."""
        return sorted({row["action_type"] for row in self._neighbours(state)})

    @staticmethod
    def _prior_mean(rows):
        count = sum(row["count"] for row in rows)
        return math.fsum(row["value"] * (row["count"] / count) for row in rows)

    def _mix(self, rewards, prior):
        if not rewards:
            return prior
        weight = self.kappa / (len(rewards) + self.kappa)
        mean = math.fsum(value / len(rewards) for value in rewards)
        return (1 - weight) * mean + weight * prior

    def estimate(self, state, action_rewards, candidate_actions):
        """Mix exact local-action evidence with abstract prior-action values."""
        if self.kappa == 0:
            return None
        rows = self._neighbours(state)
        if not rows:
            return None
        local = {}
        for action, rewards in action_rewards.items():
            action_type(action)
            local[action] = [_finite(_finite(value, "local return") / self.return_scale,
                                     "scaled local return") for value in rewards]
        all_rewards = [value for rewards in local.values() for value in rewards]
        state_value = self._mix(all_rewards, self._prior_mean(rows))
        action_values = {}
        for action in candidate_actions:
            kind = action_type(action)
            rewards = local.get(action, [])
            matches = [row for row in rows if row["action_type"] == kind]
            if matches:
                action_values[action] = self._mix(rewards, self._prior_mean(matches))
            elif rewards:
                action_values[action] = math.fsum(value / len(rewards) for value in rewards)
            else:
                action_values[action] = state_value
        return action_values, state_value


def _episode_rows(episode, domain, scale, gamma):
    """Match the native memory return rules, including their zero-score fallback."""
    if not isinstance(episode, dict) or not isinstance(episode.get("steps"), list):
        raise ValueError("source must contain native episode objects with a steps list")
    if not {"timestamp", "num_steps", "final_score", "success"} <= episode.keys():
        raise ValueError("source must use the native Jericho or WebArena episode schema")
    steps = episode["steps"]
    webarena = "task_goal" in episode
    if any(not isinstance(step, dict) for step in steps):
        raise ValueError("episode steps must be objects")
    if any(("normalized_action" in step) != webarena for step in steps):
        raise ValueError("source episode has mixed or unknown memory schemas")
    scores = []
    future_rewards = []
    for step in steps:
        score = step.get("llm_step_score", 0)
        if webarena and score is None:
            score = 0
        score = _finite(score, "llm_step_score")
        scores.append(score)
        future_rewards.append(score if webarena or score != 0 else
                              _finite(step.get("reward", 0), "reward"))
    rows = []
    for index, step in enumerate(steps):
        state = step.get("state")
        key = shared_key(state)
        action = step.get("action")
        if action is None or action == "":
            continue
        kind = action_type(action)
        if not key or not kind:
            continue
        value = math.fsum((gamma ** offset) * (reward if reward != 0 else scores[index])
                          for offset, reward in enumerate(future_rewards[index:])) / scale
        rows.append({"key": sorted(key), "action_type": kind,
                     "value": _finite(value, "source return"), "count": 1, "domain": domain})
    return rows, "webarena" if webarena else "jericho"


def build_prior(sources, gamma=0.5, exclude_domain=None):
    """Build rows from (domain, positive return scale, native episodes JSONL) tuples."""
    if not 0 <= _finite(gamma, "gamma") <= 1:
        raise ValueError("gamma must be between zero and one")
    groups = {}
    metadata = []
    for domain, scale, path in sources:
        if not isinstance(domain, str) or not domain.strip():
            raise ValueError("source domain must be nonempty text")
        scale = _positive(float(scale), "source return scale")
        if domain == exclude_domain:
            continue
        formats = set()
        with open(path, encoding="utf-8") as source:
            for line_number, line in enumerate(source, 1):
                if not line.strip():
                    continue
                try:
                    rows, source_format = _episode_rows(json.loads(line), domain, scale, gamma)
                except (ValueError, TypeError, OverflowError) as error:
                    raise ValueError(f"{path}:{line_number}: {error}") from error
                formats.add(source_format)
                for row in rows:
                    key = (tuple(row["key"]), row["action_type"], domain)
                    groups.setdefault(key, []).append(row["value"])
        if len(formats) > 1:
            raise ValueError(f"{path}: a source file must use one native memory schema")
        metadata.append({"domain": domain, "return_scale": scale,
                         "format": next(iter(formats), "empty")})
    rows = [{"key": list(key), "action_type": kind, "domain": domain,
             "value": math.fsum(value / len(values) for value in values), "count": len(values)}
            for (key, kind, domain), values in sorted(groups.items())]
    return {"version": 1, "gamma": gamma, "sources": metadata, "rows": rows}


def save_prior(data, output, source_paths=()):
    """Replace the output atomically, but never replace a source episode file."""
    output = Path(output)
    for source in map(Path, source_paths):
        if output.resolve() == source.resolve() or (
            output.exists() and source.exists() and output.samefile(source)
        ):
            raise ValueError("output must not overwrite a source episode file")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", delete=False) as target:
            temporary = target.name
            json.dump(data, target, indent=2, allow_nan=False)
            target.write("\n")
            target.flush()
            os.fsync(target.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description="Build a fixed JitRL-P prior from native episode JSONL files")
    parser.add_argument("--source", action="append", nargs=3, required=True,
                        metavar=("DOMAIN", "SCALE", "EPISODES_JSONL"))
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--output", required=True)
    parser.add_argument("--exclude_domain", help="Exclude this source domain for a hold-out trial")
    args = parser.parse_args()
    try:
        data = build_prior(args.source, args.gamma, args.exclude_domain)
        save_prior(data, args.output, [source[2] for source in args.source])
    except (OSError, ValueError, TypeError, OverflowError) as error:
        parser.error(str(error))
    print(f"Saved {len(data['rows'])} prototype rows to {args.output}")


if __name__ == "__main__":
    main()
