"""
Minimal NeuralTxt reasoning model usage.

Run:
    HF_HOME=.hf-cache uv run python scripts/reasoning_usage.py

Use MLX:
    HF_HOME=.hf-cache uv run python scripts/reasoning_usage.py --mlx
"""
import argparse

from neuraltxt import NeuralTxt, ReasonedOutput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx", action="store_true", help="Use the MLX reasoning model")
    parser.add_argument("--json", action="store_true", help="Use constrained JSON mode")
    args = parser.parse_args()

    model = NeuralTxt(
        backend="mlx" if args.mlx else "hf",
        reasoning=True,
    )

    passage = (
        "Transformers use self-attention to process tokens in parallel. "
        "The original Transformer architecture was introduced by Vaswani et al. in 2017."
    )

    result = model.generate_question(
        passage,
        json=args.json,
        return_reasoning=True,
    )

    if not isinstance(result, ReasonedOutput):
        raise TypeError(f"expected ReasonedOutput, got {type(result).__name__}")

    print("=== Reasoning ===")
    print(result.reasoning)
    print("\n=== Output ===")
    print(result.output.model_dump_json(indent=2) if args.json else result.output)


if __name__ == "__main__":
    main()
