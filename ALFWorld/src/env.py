"""Text-only ALFWorld games, wrapped to look like the Jericho environment."""

import json
from pathlib import Path

# Same ids as alfworld.agents.environment.alfred_tw_env.TASK_TYPES.
TASK_TYPES = {
    1: "pick_and_place_simple",
    2: "look_at_obj_in_light",
    3: "pick_clean_then_place_in_recep",
    4: "pick_heat_then_place_in_recep",
    5: "pick_cool_then_place_in_recep",
    6: "pick_two_obj_and_place",
}
# valid_unseen is ALFWorld's out-of-distribution eval set (134 games), valid_seen the in-distribution one (140).
SPLITS = ("valid_unseen", "valid_seen", "valid_train", "train")
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[1] / "data"
WELCOME = "-= Welcome to TextWorld, ALFRED! =-"


def task_type_of(game_file):
    # Game folders are named like pick_and_place_simple-Book-None-Desk-310/trial_.../game.tw-pddl
    return Path(game_file).parent.parent.name.split("-", 1)[0]


def parse_task_types(text):
    """Turn '1,pick_cool_then_place_in_recep' into a set of task type names."""
    if not text:
        return None
    names = set()
    for item in text.split(","):
        item = item.strip()
        name = TASK_TYPES.get(int(item)) if item.isdigit() else item
        if name not in TASK_TYPES.values():
            raise ValueError(f"Unknown ALFWorld task type: {item!r}. Use 1-6 or one of {sorted(TASK_TYPES.values())}")
        names.add(name)
    return names


def collect_games(data_path, split):
    """Sorted solvable game files of a split, filtered like ALFWorld's AlfredTWEnv."""
    root = Path(data_path) / "json_2.1.1" / split
    if not root.is_dir():
        raise FileNotFoundError(
            f"No ALFWorld games at {root}. Download json_2.1.2_tw-pddl.zip into {data_path} (see README).")
    games = []
    for path in sorted(root.glob("*/*/game.tw-pddl")):
        relative = path.relative_to(root).as_posix()
        if "movable" in relative or "Sliced" in relative:
            continue
        with open(path, encoding="utf-8") as source:
            if not json.load(source).get("solvable", False):
                continue
        games.append(str(path))
    return games


def select_games(games, ids=None, start=None, end=None, task_types=None):
    """Return (game_id, path) pairs. A game id is its index in the sorted split, so ids stay stable."""
    selected = []
    for game_id, path in enumerate(games):
        if ids is not None and game_id not in ids:
            continue
        if start is not None and game_id < start:
            continue
        if end is not None and game_id >= end:
            continue
        if task_types and task_type_of(path) not in task_types:
            continue
        selected.append((game_id, path))
    return selected


def parse_task(observation):
    marker = "Your task is to: "
    return observation.split(marker, 1)[1].strip() if marker in observation else ""


class ALFWorldEnv:
    """One ALFWorld game. reset/step return Jericho-style (ob, info) and (ob, reward, done, info).

    info['valid'] holds the admissible commands, info['look'] the task, the agent's
    last location and its inventory. JitRL retrieval reads info['look'] as the state description.
    """

    def __init__(self, game_file, step_limit=None):
        import textworld
        import textworld.gym
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self.game_file = str(game_file)
        self.task_type = task_type_of(game_file)
        self.step_limit = step_limit
        request_infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile"])
        env_id = textworld.gym.register_game(
            self.game_file, request_infos, max_episode_steps=None,
            wrappers=[AlfredDemangler(shuffle=False), AlfredInfos])
        self.env = textworld.gym.make(env_id)
        self.task = ""
        self.location = ""
        self.score = 0.0
        self.steps = 0

    def reset(self):
        ob, raw = self.env.reset()
        ob = ob.replace(WELCOME, "").strip()
        self.task = parse_task(ob)
        self.location = ob.split("\n\n")[0]
        self.score = 0.0
        self.steps = 0
        return ob, self._info(raw, done=False)

    def step(self, action):
        ob, score, done, raw = self.env.step(action)
        ob = ob.strip()
        self.steps += 1
        if ob.startswith(("You arrive at", "You open the")):
            self.location = ob
        score = float(score or 0)
        reward = score - self.score
        self.score = score
        if self.step_limit and self.steps >= self.step_limit:
            done = True
        return ob, reward, done, self._info(raw, done)

    def _info(self, raw, done):
        # TextWorld does not report the inventory for these games, and "inventory" changes nothing.
        inventory = "" if done else self.env.step("inventory")[0].strip()
        return {
            "valid": [command for command in raw["admissible_commands"] if command != "help"],
            "inv": inventory,
            "look": "\n".join(part for part in (f"Your task is to: {self.task}", self.location, inventory) if part),
            "task": self.task,
            "task_type": self.task_type,
            "won": bool(raw["won"]),
            "score": self.score,
            "gamefile": self.game_file,
        }

    def close(self):
        self.env.close()
