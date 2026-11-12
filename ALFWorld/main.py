import argparse
import os

from src.env import DEFAULT_DATA_PATH, SPLITS
from src.evaluation import ALFWorldEvaluator
from jitrl_prior import add_prior_arguments  # importable after src adds the repository root to sys.path


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run JitRL or a baseline agent on text-only ALFWorld.")
    add_prior_arguments(parser)

    # Games
    parser.add_argument('--data_path', default=str(DEFAULT_DATA_PATH), type=str,
                        help="Folder that holds json_2.1.1/<split>/.../game.tw-pddl.")
    parser.add_argument('--split', default='valid_unseen', choices=SPLITS,
                        help="valid_unseen: 134 out-of-distribution games; valid_seen: 140 in-distribution games.")
    parser.add_argument('--task_types', default=None, type=str,
                        help="Comma-separated task types, as ids 1-6 or names (e.g. '1,pick_cool_then_place_in_recep').")
    parser.add_argument('--games', default=None, type=str,
                        help="Comma-separated game ids (indices in the sorted split).")
    parser.add_argument('--start', default=None, type=int, help="First game id (inclusive).")
    parser.add_argument('--end', default=None, type=int, help="Last game id (exclusive).")
    parser.add_argument('--rounds', default=1, type=int,
                        help="Passes over the selected games. Memory carries over between games and rounds.")
    parser.add_argument('--shuffle', default=True, action=argparse.BooleanOptionalAction,
                        help="Shuffle the game order once with --seed. Sorted order groups games by task type.")

    parser.add_argument('--output_path', default='output', type=str, help="Base directory for logs and memory.")
    parser.add_argument('--env_step_limit', default=50, type=int, help="Maximum number of steps per episode.")
    parser.add_argument('--seed', default=0, type=int, help="Random seed.")

    # LLM
    parser.add_argument('--llm_model', default='google/gemini-2.5-flash', type=str, help="LLM model for the agent.")
    parser.add_argument('--top_actions', default=3, type=int, help="Number of candidate actions per step.")
    parser.add_argument('--llm_temperature', default=0.8, type=float, help="Temperature for the agent's LLM.")
    parser.add_argument('--max_memory', default=30, type=int, help="Maximum number of past states kept by the agent.")
    parser.add_argument('--gamma', default=0.5, type=float, help="Discount factor for returns in cross-episode memory.")
    parser.add_argument('--exploration_rate', default=0.65, type=float, help="Exploration parameter for action confidence.")
    parser.add_argument('--use_history_prompt', default=True, action=argparse.BooleanOptionalAction,
                        help="Compare with top-scoring episodes when updating the guiding prompt.")

    parser.add_argument('--agent_type', type=str, default='jitrl', choices=['jitrl', 'naive'], help="Agent to evaluate.")

    # Memory
    parser.add_argument('--retrieval_top_k', type=int, default=10, help="Number of similar steps to retrieve from memory.")
    parser.add_argument('--retrieval_threshold', type=float, default=0.95,
                        help="Similarity threshold for retrieved steps. Lower it to share experience across different games.")
    parser.add_argument('--enable_cross_mem', default=True, action=argparse.BooleanOptionalAction,
                        help="Enable cross-episode memory.")
    parser.add_argument('--update_guiding_prompt', default=False, action=argparse.BooleanOptionalAction,
                        help="Update the guiding prompt at the end of each episode.")
    parser.add_argument('--use_valid_actions', default=True, action=argparse.BooleanOptionalAction,
                        help="Give the agent ALFWorld's admissible commands.")
    parser.add_argument('--confidence_mode', type=str, default='verbalized', choices=['logit', 'verbalized'],
                        help='"logit" reads logprobs (OpenAI only); "verbalized" asks for confidence percentages.')
    parser.add_argument('--eval_llm_model', type=str, default='google/gemini-2.5-flash',
                        help="LLM model that scores steps for cross-episode memory.")

    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.output_path, exist_ok=True)
    results = ALFWorldEvaluator(args).run_evaluation()
    if results.get("success", False):
        print("Evaluation completed successfully!")
        exit(0)
    print("Evaluation failed!")
    exit(1)
