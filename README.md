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

---

## Key Features

- **Jaccard Similarity Matching**: N-gram based similarity for trajectory retrieval
- **LLM-based Step Scoring**: Automatic evaluation of action quality
- **Dynamic Prompt Generation**: Adaptive prompts based on task and history
- **Multi-model Support**: OpenAI, Anthropic, Google, and open-source models via OpenRouter

---

## Installation

### Prerequisites

- Python 3.10+
- OpenRouter API key (for LLM inference)

### Clone Repository

```bash
git clone https://github.com/your-username/JitRL.git
cd JitRL
```

### Jericho Setup (Text Adventure Games)

```bash
cd Jericho

# Install dependencies
pip install jericho openai tiktoken numpy python-dotenv

# Download game ROMs (required)
# Place .z5/.z8 files in jericho-games/ directory
# Games available: zork1, zork3, library, detective, etc.
```

### WebArena Setup (Web Automation)

```bash
cd WebArena

# Install dependencies
pip install browsergym-core browsergym-experiments
pip install openai tiktoken numpy langchain pillow
```

### Environment Configuration

Create a `.env` file in the project root:

```bash
# OpenRouter API (for LLM inference)
OPENROUTER_API_KEY=your_openrouter_key

# WebArena URLs (if running locally)
WA_SHOPPING=http://localhost:7770
WA_SHOPPING_ADMIN=http://localhost:7780/admin
WA_REDDIT=http://localhost:9999
WA_GITLAB=http://localhost:8023
WA_MAP=http://localhost:3000
WA_WIKIPEDIA=http://localhost:8888
```

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
- [WebArena](https://webarena.dev/) - Web agent benchmark by CMU

---

## License

This project is for research purposes only. See [LICENSE](LICENSE) for details.
