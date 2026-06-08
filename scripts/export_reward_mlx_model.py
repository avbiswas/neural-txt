from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SOURCE_REPO = "paperbd/neuraltxt-reward-tiny"
DEFAULT_OUTPUT_DIR = "dist/neuraltxt-reward-tiny-mlx"


def resolve_file(model_path: str, filename: str) -> str:
    local_path = Path(model_path) / filename
    if local_path.exists():
        return str(local_path)

    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=model_path, filename=filename)


def write_head_safetensors(source: str, output_dir: Path) -> None:
    import torch
    import mlx.core as mx

    head_path = resolve_file(source, "head_weights.pt")
    try:
        state_dict = torch.load(head_path, map_location="cpu", weights_only=True)
    except TypeError:
        state_dict = torch.load(head_path, map_location="cpu")

    mx.save_safetensors(
        str(output_dir / "reward_head.safetensors"),
        {
            "weight": mx.array(state_dict["1.weight"].numpy()),
            "bias": mx.array(state_dict["1.bias"].numpy()),
        },
        metadata={"format": "mlx"},
    )


def write_config(source: str, output_dir: Path) -> None:
    config_path = resolve_file(source, "config.json")
    with open(config_path) as f:
        config = json.load(f)
    config["architectures"] = ["NeuralTxtRewardMLX"]
    config["library_name"] = "neural-txt"
    config["base_model"] = source
    config["reward_head_file"] = "reward_head.safetensors"
    config["model_file"] = "model.safetensors"
    config["reward_pooling"] = "meanmax"
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
        f.write("\n")


def write_readme(output_dir: Path, repo_id: str) -> None:
    readme = f"""---
license: mit
language: en
library_name: neural-txt
pipeline_tag: text-classification
tags:
  - mlx
  - reward-model
  - answer-equivalence
  - question-answering
base_model: {SOURCE_REPO}
---

# NeuralTxt Reward Model MLX

MLX-ready package for [`{SOURCE_REPO}`](https://huggingface.co/{SOURCE_REPO}).

This repo contains:

- `model.safetensors`: MiniLM/BERT encoder weights readable by MLX.
- `reward_head.safetensors`: clamped linear reward head in MLX safetensors format.
- tokenizer files copied from the source reward model.

## Usage

```python
from neuraltxt import NeuralTxtReward

reward = NeuralTxtReward(backend="mlx")
score = reward.score(
    response="Paris is the capital of France.",
    reference="The capital of France is Paris.",
)
```

To load this repo explicitly:

```python
reward = NeuralTxtReward("{repo_id}", backend="mlx")
```
"""
    (output_dir / "README.md").write_text(readme)


def export(source: str, output_dir: Path, repo_id: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_file(source, "model.safetensors")
    shutil.copy2(model_path, output_dir / "model.safetensors")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(source)
    tokenizer.save_pretrained(output_dir)

    write_head_safetensors(source, output_dir)
    write_config(source, output_dir)
    write_readme(output_dir, repo_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export NeuralTxt reward model for MLX.")
    parser.add_argument("--source", default=SOURCE_REPO)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repo-id", default="paperbd/neuraltxt-reward-tiny-mlx")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    export(args.source, output_dir, args.repo_id)
    print(f"Wrote MLX reward model to {output_dir}")


if __name__ == "__main__":
    main()
