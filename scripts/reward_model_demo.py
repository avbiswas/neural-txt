from __future__ import annotations

import argparse
from dataclasses import dataclass
from time import perf_counter
import tracemalloc

from rich import box
from rich.console import Console
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from neuraltxt import NeuralTxtReward


@dataclass(frozen=True)
class Example:
    label: str
    response: str
    reference: str
    expected_match: bool
    confound: str | None = None
    response_highlights: tuple[str, ...] = ()
    reference_highlights: tuple[str, ...] = ()


@dataclass(frozen=True)
class Result:
    example: Example
    score: float
    latency_ms: float
    peak_mb: float


EXAMPLES = [
    Example(
        label="Paraphrase",
        reference=(
            "Each token can directly use information from all other tokens through "
            "self-attention."
        ),
        response=(
            "Self-attention lets each token attend to every other token in the "
            "sequence."
        ),
        expected_match=True,
    ),
    Example(
        label="Low-overlap match",
        reference="The capital of France is Paris.",
        response="Paris is France's capital city.",
        expected_match=True,
    ),
    Example(
        label="Hard confound",
        reference="The capital of France is Paris.",
        response="The capital of France is Lyon.",
        expected_match=False,
        confound="entity",
        response_highlights=("Lyon",),
        reference_highlights=("Paris",),
    ),
    Example(
        label="Number exact",
        reference="The treatment improved accuracy from 60% to 82%.",
        response="The treatment improved accuracy from 60% to 82.",
        expected_match=True,
    ),
    Example(
        label="Number swap",
        reference="The treatment improved accuracy from 60% to 82%.",
        response="The treatment improved accuracy from 60% to 68%.",
        expected_match=False,
        confound="number",
        response_highlights=("68%",),
        reference_highlights=("82%",),
    ),
    Example(
        label="Dataset match",
        reference="The model was evaluated on paperbd/paper_answers_reward.",
        response="Evaluation used the paperbd/paper_answers_reward dataset.",
        expected_match=True,
    ),
    Example(
        label="Dataset swap",
        reference="The model was evaluated on paperbd/paper_answers_reward.",
        response="The model was evaluated on sentence-transformers/stsb.",
        expected_match=False,
        confound="dataset",
        response_highlights=("sentence-transformers/stsb",),
        reference_highlights=("paperbd/paper_answers_reward",),
    ),
    Example(
        label="Model size swap",
        reference="The reward model has about 22M parameters.",
        response="The reward model has about 135M parameters.",
        expected_match=False,
        confound="number",
        response_highlights=("135M",),
        reference_highlights=("22M",),
    ),
    Example(
        label="Metric swap",
        reference="The model reached 93% accuracy on answer equivalence.",
        response="The model reached 63% accuracy on answer equivalence.",
        expected_match=False,
        confound="metric",
        response_highlights=("63%",),
        reference_highlights=("93%",),
    ),
    Example(
        label="Method match",
        reference=(
            "Meanmax pooling concatenates masked mean and masked max token pooling."
        ),
        response=(
            "Meanmax pooling joins the masked token average with the masked token "
            "maximum."
        ),
        expected_match=True,
    ),
    Example(
        label="Incomplete",
        reference=(
            "Meanmax pooling concatenates masked mean pooling with masked max pooling."
        ),
        response="Meanmax pooling uses mean pooling.",
        expected_match=False,
        confound="missing",
        response_highlights=("mean pooling",),
        reference_highlights=("max pooling",),
    ),
    Example(
        label="Unrelated",
        reference="Gradient checkpointing trades compute for lower activation memory.",
        response="The Eiffel Tower was completed in 1889 for the World's Fair.",
        expected_match=False,
    ),
]


def short(text: str, width: int = 72) -> str:
    if len(text) <= width:
        return text
    return text[: width - 1].rstrip() + "..."


def highlighted_cell(
    text: str, terms: tuple[str, ...], style: str, width: int = 88
) -> Text:
    cell = Text(short(text, width=width))
    for term in terms:
        start = cell.plain.find(term)
        if start >= 0:
            cell.stylize(style, start, start + len(term))
    return cell


def score_style(score: float) -> str:
    if score >= 0.75:
        return "bold green"
    if score >= 0.45:
        return "yellow"
    return "red"


def verdict(result: Result, threshold: float) -> Text:
    predicted_match = result.score >= threshold
    got_expected = predicted_match == result.example.expected_match
    return Text(
        "✅" if got_expected else "❌",
        style="bold green" if got_expected else "bold red",
    )


def expected_label(result: Result, threshold: float) -> Text:
    predicted_match = result.score >= threshold
    text = "MATCH" if predicted_match else "NO MATCH"
    style = "bold green" if predicted_match else "bold red"
    return Text(text, style=style)


def bar(value: float, maximum: float, width: int = 24, style: str = "cyan") -> Text:
    filled = 0 if maximum <= 0 else round((value / maximum) * width)
    filled = max(0, min(width, filled))
    text = Text()
    text.append("█" * filled, style=style)
    text.append("░" * (width - filled), style="grey37")
    return text


def profile_score(
    reward: NeuralTxtReward, example: Example, batch_size: int
) -> Result:
    tracemalloc.start()
    start = perf_counter()
    score = reward.batch_score(
        [example.response], [example.reference], batch_size=batch_size
    )[0]
    latency_ms = (perf_counter() - start) * 1000
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return Result(
        example=example,
        score=score,
        latency_ms=latency_ms,
        peak_mb=peak_bytes / (1024 * 1024),
    )


def build_score_table(results: list[Result], threshold: float) -> Table:
    table = Table(
        title="NeuralTxt Reward Tiny: reference vs model output",
        box=box.SIMPLE_HEAVY,
        show_lines=True,
    )
    table.add_column("Case", style="bold", no_wrap=True, max_width=15)
    table.add_column("Comparison", overflow="fold", ratio=1)
    table.add_column("Decision", justify="center", no_wrap=True, width=9)
    table.add_column("Score", justify="right", no_wrap=True, width=6)
    table.add_column("OK", justify="center", no_wrap=True, width=4)

    previous_expected_match: bool | None = None
    for result in results:
        if previous_expected_match is True and not result.example.expected_match:
            table.add_section()
        previous_expected_match = result.example.expected_match

        score = Text(f"{result.score:.3f}", style=score_style(result.score))
        label = Text(result.example.label)
        if result.example.confound:
            label = Text("⚠ ", style="bold yellow") + label
        row_style = "bold" if result.example.confound else None
        comparison = Text("REF  ", style="bold green")
        comparison.append_text(
            highlighted_cell(
                result.example.reference,
                result.example.reference_highlights,
                "bold black on green",
            )
        )
        comparison.append("\nOUT  ", style="bold red")
        comparison.append_text(
            highlighted_cell(
                result.example.response,
                result.example.response_highlights,
                "bold white on red",
            )
        )
        table.add_row(
            label,
            comparison,
            expected_label(result, threshold),
            score,
            verdict(result, threshold),
            style=row_style,
        )
    return table


def build_score_section(results: list[Result], threshold: float) -> Group:
    legend = Text()
    legend.append("Legend: ", style="bold")
    legend.append("green highlight", style="bold black on green")
    legend.append(" = reference fact, ")
    legend.append("red highlight", style="bold white on red")
    legend.append(" = changed output fact, ")
    legend.append("⚠ = adversarial confound", style="bold yellow")
    return Group(legend, build_score_table(results, threshold))


def build_metric_table(results: list[Result]) -> Table:
    max_latency = max(result.latency_ms for result in results)
    max_memory = max(result.peak_mb for result in results)

    table = Table(
        title="Runtime profile per scored pair",
        box=box.SIMPLE_HEAVY,
        show_edge=False,
    )
    table.add_column("Case", style="bold", no_wrap=True, max_width=16)
    table.add_column("Latency", no_wrap=True, width=14)
    table.add_column("ms", justify="right", no_wrap=True, width=7)
    table.add_column("Peak mem", no_wrap=True, width=14)
    table.add_column("MB", justify="right", no_wrap=True, width=7)

    for result in results:
        table.add_row(
            result.example.label,
            bar(result.latency_ms, max_latency, width=14, style="cyan"),
            f"{result.latency_ms:.1f}",
            bar(result.peak_mb, max_memory, width=14, style="magenta"),
            f"{result.peak_mb:.2f}",
        )
    return table


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rich demo for the NeuralTxt reward model."
    )
    parser.add_argument(
        "--backend",
        choices=("hf", "mlx"),
        default="hf",
        help="Reward backend to use. Use mlx on Apple Silicon after installing neural-txt[mlx].",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size passed to NeuralTxtReward.batch_score.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Score threshold used for the artificial correct/incorrect verdict.",
    )
    args = parser.parse_args()

    console = Console()
    console.print(
        Panel.fit(
            (
                "[bold]NeuralTxt Reward Tiny[/bold]\n"
                "Scores whether a response is correct and complete against a "
                "reference answer. The examples include paraphrases, low lexical "
                "overlap, number swaps, dataset-name swaps, and unrelated text."
            ),
            border_style="blue",
        )
    )

    with console.status(f"Loading reward model with backend={args.backend!r}..."):
        reward = NeuralTxtReward(backend=args.backend)

    with console.status("Scoring examples and profiling runtime..."):
        results = [
            profile_score(reward, example, args.batch_size) for example in EXAMPLES
        ]
    results = sorted(results, key=lambda result: not result.example.expected_match)

    console.print(build_score_section(results, args.threshold))
    console.print(build_metric_table(results))
    console.print(
        "[dim]Peak memory is measured with tracemalloc during scoring after model "
        "load, so it reflects Python-side allocation for each scored pair.[/dim]"
    )


if __name__ == "__main__":
    main()
