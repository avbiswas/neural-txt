from __future__ import annotations

import argparse
import builtins
from contextlib import contextmanager
import json
import random
from dataclasses import dataclass
import time
from time import perf_counter

from rich import box
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from neuraltxt import NeuralTxt, NeuralTxtReward
from neuraltxt.tasks import SYSTEM_PROMPT


DATASET_REPO = "paperbd/paper_instructions_300K-v1"
DATASET_FILE = "test.jsonl"


@dataclass(frozen=True)
class DatasetRow:
    index: int
    instruction: str
    input: str
    output: str


@dataclass(frozen=True)
class RolloutResult:
    index: int
    response: str
    reward: float


def compact(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


@contextmanager
def suppress_raw_backend_prints():
    original_print = builtins.print

    def filtered_print(*args, **kwargs):
        if args and isinstance(args[0], str) and "[RAW OUTPUT]" in args[0]:
            return
        original_print(*args, **kwargs)

    builtins.print = filtered_print
    try:
        yield
    finally:
        builtins.print = original_print


def load_test_rows(limit: int | None = None) -> list[DatasetRow]:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        raise ImportError(
            "This demo needs huggingface_hub to fetch the dataset. Install with:\n\n"
            "  pip install neural-txt"
        ) from None

    path = hf_hub_download(DATASET_REPO, DATASET_FILE, repo_type="dataset")
    rows: list[DatasetRow] = []
    with open(path) as f:
        for index, line in enumerate(f):
            item = json.loads(line)
            rows.append(
                DatasetRow(
                    index=index,
                    instruction=item["instruction"],
                    input=item["input"],
                    output=item["output"],
                )
            )
            if limit is not None and len(rows) >= limit:
                break
    return rows


def select_rows(
    rows: list[DatasetRow], count: int, row_index: int | None, seed: int
) -> list[DatasetRow]:
    if row_index is not None:
        for row in rows:
            if row.index == row_index:
                return [row]
        raise ValueError(f"row index {row_index} was not found in the loaded rows")

    rng = random.Random(seed)
    if count >= len(rows):
        return rows
    return rng.sample(rows, count)


def build_source_panel(row: DatasetRow) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(style="bold cyan", no_wrap=True)
    table.add_column(ratio=1)
    table.add_row("Dataset", f"{DATASET_REPO} / test row {row.index}")
    table.add_row("System", compact(SYSTEM_PROMPT, 520))
    table.add_row("Instruction", compact(row.instruction, 520))
    table.add_row("Input", compact(row.input, 900))
    table.add_row("Reference", compact(row.output, 900))
    return Panel(table, title="Dataset example", border_style="blue")


def build_live_table(
    responses: list[str],
    rewards: list[float | None],
    phase: str,
    active_index: int | None = None,
    scoring_indices: set[int] | None = None,
) -> Table:
    table = Table(
        title="Responses + reward scores",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
    )
    table.add_column("Rollout", justify="right", no_wrap=True, width=7)
    table.add_column("Streaming response", ratio=1, overflow="fold")
    table.add_column("Reward", justify="right", no_wrap=True, width=8)
    table.add_column("Advantage", justify="right", no_wrap=True, width=10)

    scoring_indices = scoring_indices or set()
    average_reward = (
        sum(reward for reward in rewards if reward is not None) / len(rewards)
        if rewards and all(reward is not None for reward in rewards)
        else None
    )
    for index, response in enumerate(responses):
        advantage = Text("...")
        if rewards[index] is not None and average_reward is not None:
            value = rewards[index] - average_reward
            style = "bold green" if value > 0 else "bold red" if value < 0 else "dim"
            advantage = Text(f"{value:+.4f}", style=style)
        reward = (
            Text(f"{rewards[index]:.4f}", style="bold cyan")
            if rewards[index] is not None
            else Text("⠋ scoring", style="bold magenta")
            if phase == "reward" and index in scoring_indices
            else Text("...")
        )
        table.add_row(str(index + 1), compact(response, 1200) or "...", reward, advantage)
    return table


def render_sleep(seconds: float) -> None:
    if seconds > 0:
        time.sleep(seconds)


def generate_and_score_rollouts(
    model: NeuralTxt,
    reward_model: NeuralTxtReward,
    row: DatasetRow,
    rollouts: int,
    temperature: float,
    max_new_tokens: int,
    batch_size: int,
    render_interval: float,
) -> tuple[list[RolloutResult], float, float]:
    # Use NeuralTxt's normal prompt path:
    # system prompt + exact dataset instruction + dataset input.
    prompt = model._build_prompt(row.instruction, model._preprocess(row.input))
    responses = [""] * rollouts
    rewards: list[float | None] = [None] * rollouts
    start = perf_counter()
    with Live(
        build_live_table(responses, rewards, "generate"),
        refresh_per_second=12,
    ) as live:
        for index in range(rollouts):
            live.update(build_live_table(responses, rewards, "generate", active_index=index))
            with suppress_raw_backend_prints():
                for chunk in model._backend.stream(
                    prompt,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                ):
                    responses[index] += chunk
                    live.update(build_live_table(responses, rewards, "generate", active_index=index))
            live.update(build_live_table(responses, rewards, "generate"))

        generation_s = perf_counter() - start
        reward_start = perf_counter()
        for start_index in range(0, rollouts, batch_size):
            batch_responses = responses[start_index : start_index + batch_size]
            batch_end = start_index + len(batch_responses)
            scoring_indices = set(range(start_index, batch_end))
            live.update(
                build_live_table(
                    responses,
                    rewards,
                    "reward",
                    scoring_indices=scoring_indices,
                )
            )
            batch_rewards = reward_model.batch_score(
                batch_responses,
                [row.output] * len(batch_responses),
                batch_size=batch_size,
            )
            for offset, reward in enumerate(batch_rewards):
                rewards[start_index + offset] = reward
            live.update(
                build_live_table(
                    responses,
                    rewards,
                    "reward",
                    scoring_indices=scoring_indices,
                )
            )
        reward_s = perf_counter() - reward_start
        live.update(build_live_table(responses, rewards, "done"))

    results = [
        RolloutResult(index=index, response=response, reward=reward or 0.0)
        for index, (response, reward) in enumerate(zip(responses, rewards), start=1)
    ]
    return results, generation_s, reward_s


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate multiple NeuralTxt rollouts from the paper instructions test "
            "split and score them with the NeuralTxt reward model."
        )
    )
    parser.add_argument("--rows", type=int, default=1, help="Number of dataset rows to sample.")
    parser.add_argument("--row-index", type=int, default=None, help="Use one exact test row index.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for row sampling.")
    parser.add_argument("--load-limit", type=int, default=250, help="Number of test rows to load before sampling.")
    parser.add_argument("--rollouts", type=int, default=4, help="Responses to generate per row.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature for generation.")
    parser.add_argument("--max-new-tokens", type=int, default=180, help="Max generated tokens per rollout.")
    parser.add_argument("--batch-size", type=int, default=64, help="Reward batch size.")
    parser.add_argument(
        "--render-interval",
        type=float,
        default=0.5,
        help="Artificial pause after each prompt-response-reward group renders, in seconds.",
    )
    parser.add_argument(
        "--compact-prompts",
        action="store_true",
        help="Show compact Prompt #N dividers instead of the full dataset panel.",
    )
    parser.add_argument("--reward-backend", choices=("mlx", "hf"), default="mlx", help="Reward model backend.")
    args = parser.parse_args()

    console = Console(width=120)
    console.print(Text("$ pip install neural-txt[mlx]", style="bold green"))
    console.print(Rule("NeuralTxt rollouts + reward scoring"))

    progress = Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TimeElapsedColumn(),
        console=console,
    )

    with Live(progress, console=console, refresh_per_second=8):
        task = progress.add_task("Loading dataset test split", total=4)
        rows = load_test_rows(args.load_limit)
        selected_rows = select_rows(rows, args.rows, args.row_index, args.seed)
        progress.advance(task)

        progress.update(task, description="Loading NeuralTxt MLX model")
        model = NeuralTxt(backend="mlx")
        progress.advance(task)

        progress.update(task, description=f"Loading reward {args.reward_backend} model")
        reward_model = NeuralTxtReward(backend=args.reward_backend)
        progress.advance(task)

        progress.update(task, description="Ready")
        progress.advance(task)

    for row_number, row in enumerate(selected_rows, start=1):
        console.print()
        if args.compact_prompts:
            console.print(Rule(f"Prompt #{row_number}"))
        else:
            console.print(Rule(f"Example {row_number}/{len(selected_rows)}"))
            console.print(build_source_panel(row))

        console.print(Text(f"Generating {args.rollouts} rollouts from a live batch", style="bold cyan"))
        generate_and_score_rollouts(
            model,
            reward_model,
            row,
            args.rollouts,
            args.temperature,
            args.max_new_tokens,
            args.batch_size,
            args.render_interval,
        )
        render_sleep(args.render_interval)


if __name__ == "__main__":
    main()
