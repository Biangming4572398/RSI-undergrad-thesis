# JitRL: Just-in-Time Reinforcement Learning

<p align="center">
  <b>Continual Learning in LLM Agents Without Gradient Updates</b>
</p>

<p align="center">
  🎉 <b>Accepted at ICML 2026 (Spotlight)</b> •
  <a href="https://arxiv.org/abs/2601.18510">arXiv:2601.18510</a>
</p>

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#quick-start">Quick Start</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#results">Results</a> •
  <a href="#citation">Citation</a>
</p>

---

## Overview

While LLM agents excel at general tasks, they struggle with **continual adaptation** due to frozen weights after deployment. Conventional RL offers a solution but incurs prohibitive computational costs and the risk of catastrophic forgetting.

<p align="center">
  <img src="figures/intro1.png" alt="JitRL vs Standard RL" width="600">
</p>
<p align="center">
  <em>Standard RL performs gradient updates during training; JitRL operates at test time by retrieving relevant trajectories to estimate advantages and refine output logits.</em>
</p>

**JitRL** is a training-free framework that enables **test-time policy optimization without any gradient updates**. Instead of updating parameters, JitRL:

- Maintains a **dynamic, non-parametric memory** that stores experience trajectories as `<state, action, reward>` triplets
- **Retrieves relevant trajectories** given the current state to estimate action advantages on-the-fly
- **Directly modulates the LLM's output logits** based on these advantage estimates

We theoretically prove that this additive update rule is the **exact closed-form solution** to the KL-constrained policy optimization objective.

<p align="center">
  <img src="figures/framework4.png" alt="JitRL Framework" width="800">
</p>
<p align="center">
  <em>JitRL Framework: The inference stream retrieves experiences from memory to estimate advantages and adjust action logits; the memory update stream stores evaluated trajectories with step-wise returns.</em>
</p>

### Supported Environments

| Environment | Domain | Tasks | Description |
|-------------|--------|-------|-------------|
| **Jericho** | Text Adventure Games | 16+ games | Interactive fiction games (Zork, Library, etc.) |
| **WebArena** | Web Automation | 812 tasks | Real-world web tasks (shopping, admin, maps, etc.) |
| **ALFWorld** | Household Tasks | 134 unseen + 140 seen games | Text-only household tasks (find, clean, heat, cool, place objects) |

---

## Key Features

- **Jaccard Similarity Matching**: N-gram based similarity for trajectory retrieval
- **LLM-based Step Scoring**: Automatic evaluation of action quality
- **Dynamic Prompt Generation**: Adaptive prompts based on task and history
- **Multi-model Support**: OpenAI, Anthropic, Google, and open-source models via OpenRouter

---

## Installation

### Prerequisites

- Python 3.11 is the default. The dependency files support Python 3.10 to 3.12.
- Install [uv](https://docs.astral.sh/uv/getting-started/installation/).
- On Linux, install GCC, G++, and Make to build Jericho.
- Supply API keys for the selected remote models.

Run the installation commands from the repository root.
`pyproject.toml` lists the dependencies. `uv.lock` records the resolved versions.
The default installation includes Jericho. The `webarena` extra adds browser tasks and evaluation.

BrowserGym 0.14.3 requires Playwright 1.44.0. Its Greenlet dependency limits this setup to Python 3.12 or earlier.
The agent also uses legacy LangChain imports, so the dependency files select the 0.3 series.
See the [BrowserGym metadata](https://pypi.org/pypi/browsergym-core/0.14.3/json)
and [LangChain migration guide](https://docs.langchain.com/oss/python/migrate/langchain-v1).

The installation excludes PyTorch, Accelerate, and vLLM. Use remote API models on this device.
WebArena still imports Transformers and Hugging Face classes. These packages do not install LLM weights.

### Jericho Setup (Text Adventure Games)

```bash
uv sync --locked
source .venv/bin/activate
python -m spacy download en_core_web_sm
```

Jericho uses the small spaCy model to parse game text. It does not use it for LLM inference.
Place game ROM files in `Jericho/jericho-games/`, or set `--rom_path`.
FAISS is included because Jericho memory retrieval requires it.
See the [Jericho setup guide](https://ms-jericho.readthedocs.io/en/latest/tutorial_quick.html).

### WebArena Setup (Web Automation)

```bash
uv sync --locked --extra webarena
source .venv/bin/activate
python -m playwright install chromium
python -m nltk.downloader punkt_tab
```

Install the required browser system libraries if Playwright reports missing libraries.
Start the WebArena website services and set their URLs before you run a task.
Python packages do not include these services. See the
[BrowserGym WebArena setup guide](https://github.com/ServiceNow/BrowserGym/tree/main/browsergym/webarena).

### ALFWorld Setup (Household Tasks)

```bash
uv sync --locked --extra alfworld
source .venv/bin/activate
```

A sync installs only the extras you name. To keep WebArena as well, run `uv sync --locked --extra alfworld --extra webarena`.
ALFWorld needs `fast-downward-textworld`, which builds from source with CMake and a C++ compiler.
On macOS, a build error such as `'memory' file not found` means an old Command Line Tools header folder
hides the SDK headers. Clear the failed build and point the compiler at the SDK:

```bash
uv cache clean fast-downward-textworld
CXXFLAGS="-isystem $(xcrun --show-sdk-path)/usr/include/c++/v1" uv sync --locked --extra alfworld
```

The pipeline reads the text-only game files from `ALFWorld/data/`. Git ignores this folder.
Download them once (36 MB, 336 MB unpacked):

```bash
mkdir -p ALFWorld/data
curl -L -o ALFWorld/data/tw-pddl.zip https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip
unzip -q ALFWorld/data/tw-pddl.zip -d ALFWorld/data && rm ALFWorld/data/tw-pddl.zip
```

If you already ran `alfworld-download`, pass `--data_path ~/.cache/alfworld` instead.
See the [ALFWorld repository](https://github.com/alfworld/alfworld) for the dataset.

After a later `uv sync`, repeat the spaCy model installation if you use Jericho.
The model is a separate download and is not in the lock file.
Use the activated environment for the commands below. This prevents `uv run` from removing an extra during a sync.

For a pip installation, first create and activate a Python 3.11 environment. Then export the selected packages:

```bash
uv export --locked --extra webarena --no-emit-project --format requirements-txt --output-file /tmp/jitrl-requirements.txt
python -m pip install -r /tmp/jitrl-requirements.txt
```

Omit `--extra webarena` for Jericho only.
`WebArena/autoeval/requirements.txt` covers the standalone evaluator. It does not install the browser agent.

The existing Codex SDK dependency is available with `uv sync --locked --extra codex`.
Codex and WebArena require incompatible versions of `packaging`. Use separate environments for these extras.
Do not use `--all-extras`.

Check the prior-memory tests without starting a game or browser:

```bash
python -m unittest discover -s tests -v
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# WebArena: OpenRouter models
OPENROUTER_API_KEY=your_openrouter_key
# WebArena: direct OpenAI models
OPENAI_API_KEY=your_openai_key

# WebArena URLs (if running locally)
WA_SHOPPING=http://localhost:7770
WA_SHOPPING_ADMIN=http://localhost:7780/admin
WA_REDDIT=http://localhost:9999
WA_GITLAB=http://localhost:8023
WA_MAP=http://localhost:3000
WA_WIKIPEDIA=http://localhost:8888
WA_HOMEPAGE=http://localhost:4399
```

Jericho and ALFWorld use different key names. Set `OPENAI_API_KEY` to the OpenRouter key for their runs.
Set `OPENAI_API_KEY2` to a direct OpenAI key for its memory embeddings.
For WebArena, also export the `WA_*` variables in the shell, or put them in `WebArena/env_setup.txt`.
BrowserGym reads these variables before the agent loads the root `.env` file.

---

## Quick Start

### Jericho: Text Adventure Games

```bash
cd Jericho

# Run JitRL agent on Zork 1 for 10 episodes
python main.py --game_name zork1 --agent_type jitrl --eval_runs 10

# Run with cross-episode memory disabled (baseline)
python main.py --game_name zork1 --agent_type jitrl --no-enable_cross_mem

# Use different LLM model
python main.py --game_name zork1 --llm_model openai/gpt-4o --eval_runs 10
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--game_name` | `library` | Game to play |
| `--agent_type` | `jitrl` | Agent: `jitrl`, `naive`, `awm` |
| `--eval_runs` | `50` | Number of episodes |
| `--llm_model` | `google/gemini-2.5-flash` | LLM model |
| `--env_step_limit` | `50` | Max steps per episode |
| `--enable_cross_mem` | `True` | Enable cross-episode memory |
| `--gamma` | `0.5` | Discount factor for rewards |

> **Note on reproducibility:** The Gemini-2.5-Flash results in the paper were obtained with the
> `google/gemini-2.5-flash-preview-09-2025` snapshot on OpenRouter, which has since been retired.
> The defaults now point to the stable `google/gemini-2.5-flash`; reproduced numbers may differ
> slightly due to the model version change.

### WebArena: Web Automation

```bash
cd WebArena

# Run single task with memory agent
python test_webarena_lite.py --tasks 0 --repeat 5 --model gpt-4o

# Run multiple tasks in parallel
python test_webarena_lite.py --start 0 --end 10 --workers 4 --repeat 3

# Run with screenshots for vision-capable models
python test_webarena_lite.py --tasks 68 --model gpt-4o \
    --use_screenshot_action --use_screenshot_eval

# Run without memory (baseline)
python test_webarena_lite.py --tasks 0 --repeat 10 --disable_memory
```

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--tasks` | - | Comma-separated task IDs (e.g., `0,1,2`) |
| `--start/--end` | - | Task ID range |
| `--repeat` | `1` | Episodes per task |
| `--workers` | `1` | Parallel workers |
| `--model` | `gpt-4o` | LLM model |
| `--max_steps` | `30` | Max steps per episode |
| `--disable_memory` | `False` | Disable memory system |

### ALFWorld: Household Tasks

```bash
cd ALFWorld

# Run JitRL once on each of the 134 unseen games
python main.py --agent_type jitrl

# Three passes over the games; memory from earlier passes guides later ones
python main.py --rounds 3

# One task type (ids 1-6 or names), first 20 game ids
python main.py --task_types pick_heat_then_place_in_recep --start 0 --end 20 --rounds 3

# Baselines: JitRL prompts without memory, and a memoryless single-call agent
python main.py --no-enable_cross_mem
python main.py --agent_type naive
```

The ALFWorld agent reuses the Jericho JitRL agent and memory. It changes the prompts, takes success from the
environment, and stores the winning step of each episode. The default retrieval threshold of `0.95` mostly
matches steps from the same game. Lower `--retrieval_threshold` to share experience across different games.
Memory is kept in `output/alfworld/<agent>/<model>/` across games, rounds, and runs. Use a new `--output_path`
to start with empty memory. Each run writes episode logs and a summary with the success rate overall,
per round, and per task type.

**Key Arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--split` | `valid_unseen` | `valid_unseen` (134 games), `valid_seen` (140), `valid_train`, `train` |
| `--task_types` | all | Comma-separated task types, as ids 1-6 or names |
| `--games` / `--start` / `--end` | all | Game ids are indices in the sorted split |
| `--rounds` | `1` | Passes over the selected games |
| `--shuffle` | `True` | Shuffle the game order once with `--seed`; `--no-shuffle` keeps id order |
| `--agent_type` | `jitrl` | Agent: `jitrl`, `naive` |
| `--env_step_limit` | `50` | Max steps per episode |
| `--data_path` | `ALFWorld/data` | Folder with `json_2.1.1/<split>/` |

The memory and LLM options match the Jericho runner.

---

## Architecture

### Project Structure

```
JitRL/
├── Jericho/                          # Text Adventure Games
│   ├── main.py                       # Entry point
│   ├── console_play.py               # Interactive play mode
│   └── src/
│       ├── jitrl_agent.py            # JitRL agent: retrieval + advantage + logit modulation (850 lines)
│       ├── naive_agent.py            # Simple baseline agent
│       ├── awm_agent.py              # Agent Workflow Memory
│       ├── cross_episode_memory.py   # Core memory system (900 lines)
│       ├── evaluation.py             # Game evaluator
│       ├── env.py                    # Jericho environment wrapper
│       ├── utils.py                  # LLM utilities (1150 lines)
│       ├── openai_helpers.py         # API helpers
│       └── prompt_update_with_history.py  # Prompt optimization
│
├── WebArena/                         # Web Automation
│   ├── test_webarena_lite.py         # Main test script
│   ├── run.py                        # Legacy entry point
│   ├── memory_agents/
│   │   ├── jitrl_agent.py            # JitRL BrowserGym agent (1450 lines)
│   │   ├── dynamic_prompting.py      # Prompt generation (760 lines)
│   │   └── utils/
│   │       ├── cross_episode_memory.py  # Memory system (1200 lines)
│   │       ├── openai_helpers.py     # LLM utilities
│   │       ├── llm_utils.py          # Parsing utilities
│   │       └── utils.py              # General utilities (2100 lines)
│   ├── autoeval/                     # Evaluation system
│   │   ├── evaluator.py              # LLM-based evaluator
│   │   ├── enhanced_evaluator.py     # Rule-based evaluator
│   │   └── live_evaluator.py         # Real-time evaluation
│   ├── config_files/                 # Full WebArena (812 tasks)
│   └── config_files_lite/            # WebArena-Lite (165 tasks)
│
├── ALFWorld/                         # Household Tasks
│   ├── main.py                       # Entry point
│   ├── data/                         # Game files (downloaded, not in git)
│   └── src/
│       ├── jitrl_agent.py            # Jericho JitRL agent with ALFWorld prompts
│       ├── naive_agent.py            # Memoryless baseline agent
│       ├── prompts.py                # Prompt text, progress summary, action matching
│       ├── evaluation.py             # Game selection, episode loop, summary
│       └── env.py                    # TextWorld ALFWorld wrapper
│
└── README.md
```

---

## Supported Models

JitRL supports any model available through OpenRouter:

| Provider | Models | Notes |
|----------|--------|-------|
| **OpenAI** | `gpt-4o`, `gpt-4-turbo`, `gpt-4o-mini` | Best for structured output |
| **Anthropic** | `claude-3.5-sonnet`, `claude-3-opus` | Strong reasoning |
| **Google** | `gemini-2.5-flash`, `gemini-2.5-pro` | Cost-effective |
| **Meta** | `llama-3.1-70b`, `llama-3.1-405b` | Open-source |
| **Mistral** | `mistral-large`, `mixtral-8x22b` | Fast inference |

---

## Results

### Jericho Text Adventures

| Game | Naive Agent | Memory Agent | Improvement |
|------|-------------|--------------|-------------|
| Zork1 | 35.2 | 52.8 | +50% |
| Library | 18.5 | 28.3 | +53% |
| Detective | 180 | 265 | +47% |

*Scores averaged over 50 episodes*

### WebArena-Lite

| Model | Without Memory | With Memory | Improvement |
|-------|----------------|-------------|-------------|
| GPT-4o | 32.5% | 41.2% | +27% |
| Claude-3.5 | 28.8% | 38.5% | +34% |
| Gemini-2.5 | 25.3% | 34.7% | +37% |

*Success rate on 165 WebArena-Lite tasks*

---

## Advanced Usage

### Prototype prior memory (JitRL-P)

Both agents can use a separate prior memory, `M₀`. This prototype follows the value mix in
`index.html` and `jitrl_p_visual (1).html`. It does not train the LLM.

For each candidate action, the unit calculates:

```text
Q = (sum(local action returns) + kappa * prior action value) / (action count + kappa)
V = (sum(all local returns) + kappa * prior state value) / (state count + kappa)
advantage = Q - V
```

The agent then uses its existing advantage normalization and score correction.
The current code adds normalized advantages to confidence scores. It does not implement
the raw-logit equation shown in the HTML files.

Build a prior from saved `episodes.jsonl` files. Run this command from the repository root:

```bash
python jitrl_prior.py \
  --source source_game 10 path/to/source_game/episodes.jsonl \
  --source other_game 20 path/to/other_game/episodes.jsonl \
  --gamma 0.5 --output prior.json
```

Each `--source` takes a domain name, a positive return scale, and a file path.
ALFWorld memory uses the Jericho episode format, so its `episodes.jsonl` also works as a source.
The builder divides returns by that scale. For example, scale `10` maps return `10` to value `1`.
Choose source and target scales that give values the same meaning.
The builder preserves the return calculation of each environment.
It combines rows with the same domain, state key, and action type.
Use `--exclude_domain NAME` to leave a source domain out of the build.
Keep target-domain data out of the source files.

To check the builder with synthetic data:

```bash
python jitrl_prior.py --source toy 1 examples/prior_source.jsonl \
  --gamma 0.5 --output /tmp/jitrl-toy-prior.json
python -m unittest discover -s tests -v
```

The example contains invented values. It is not a trained prior or a benchmark result.

Run Jericho with a prior:

```bash
cd Jericho
python main.py --game_name zork1 --agent_type jitrl \
  --prior_path ../prior.json --prior_kappa 2 --prior_return_scale 10 \
  --gamma 0.5 --output_path output/prior-trial
```

Run WebArena with a prior built with the same discount factor:

```bash
cd WebArena
python test_webarena_lite.py --tasks 0 --repeat 3 \
  --prior_path ../web-prior.json --prior_kappa 2 --prior_return_scale 1
```

The WebArena runner uses `gamma=0.1` by default. Build `web-prior.json` with `--gamma 0.1`.
The prior loader rejects a different discount factor.

| Argument | Default | Purpose |
|----------|---------|---------|
| `--prior_path` | None | Read the prior JSON file. |
| `--prior_kappa` | `0` | Set the prior strength. Zero keeps the existing JitRL path. |
| `--prior_return_scale` | `1` | Divide local returns by this positive value before the mix. |

Local episode files keep their existing format. Calibration occurs when values are read.
Disabling cross-episode memory also disables the prior.
An absent prior match uses the existing JitRL path, including its exploration rule.
For an unmatched action type, the unit uses the local mean when local evidence exists.
Otherwise, it uses the mixed state value, which gives zero advantage.
Prior action types guide the LLM proposals. Source object names and browser element IDs
are not added as executable actions.
The LLM can omit a suggested type, so this does not guarantee the candidate union in the HTML design.

This prototype uses fixed text keys and action types. It does not learn keys, calibration,
or `kappa`. The domain hold-out search in the HTML design remains a separate research step.
The count `n` is the number of retrieved local rows, not the episode count.
A fixed retrieval limit can prevent the prior weight from reaching zero.
The existing advantage normalization can also retain a score bias when value differences become small.
No improvement in task success is claimed.

### Clearing Memory

```python
from src.cross_episode_memory import CrossEpisodeMemory

memory = CrossEpisodeMemory("output/game/agent/model")
memory.clear_memory(save_to_disk=True)
```

### Custom Retrieval Parameters

```bash
# Adjust retrieval sensitivity
python main.py --game_name zork1 \
    --retrieval_top_k 10 \
    --retrieval_threshold 0.8 \
    --gamma 0.95
```

### Prompt Optimization

```bash
# Enable automatic prompt updates
python main.py --game_name zork1 \
    --update_guiding_prompt \
    --use_history_prompt
```

---

## Troubleshooting

### Common Issues

1. **OpenRouter rate limits**: Use exponential backoff (built-in)
2. **Memory errors**: Reduce `--retrieval_top_k` or `--max_memory`
3. **WebArena connection**: Ensure Docker containers are running

### Logging

Logs are saved to:
- Jericho: `output/{game}/{agent}/{model}/{timestamp}/`
- WebArena: `results/{task_id}/` or `--log_dir`

---

## Citation

If you use JitRL in your research, please cite:

```bibtex
@article{li2026just,
  title={Just-In-Time Reinforcement Learning: Continual Learning in LLM Agents Without Gradient Updates},
  author={Li, Yibo and Lin, Zijie and Deng, Ailin and Zhang, Xuan and He, Yufei and Ji, Shuo and Cao, Tri and Hooi, Bryan},
  journal={arXiv preprint arXiv:2601.18510},
  year={2026}
}
```

---

## Acknowledgments

- [Jericho](https://github.com/microsoft/jericho) - Text adventure game framework by Microsoft Research
- [BrowserGym](https://github.com/ServiceNow/BrowserGym) - Web automation framework by ServiceNow
- [ALFWorld](https://github.com/alfworld/alfworld) - Text-based household task benchmark
- [WebArena](https://webarena.dev/) - Web agent benchmark by CMU

---

## License

This project is for research purposes only. See [LICENSE](LICENSE) for details.
