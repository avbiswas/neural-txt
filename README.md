# neural-txt

Structured NLP tasks powered by a fine-tuned 135M parameter language model. Extract bullets, generate Q&A pairs, build knowledge graphs, and more — all running locally. Narrow vertical local intelligence that runs super cheaply in resource constrained envs.

## Install

```bash
# Base (tokenizer only, no inference backend)
uv pip install -e .

# With HuggingFace backend (torch)
uv pip install -e ".[hf]"

# With MLX backend (Apple Silicon)
uv pip install -e ".[mlx]"
```

## Quick start

```python
from neuraltxt import NeuralTxt

model = NeuralTxt(backend="mlx")  # or backend="hf"

passage = """
Transformers have revolutionized NLP by introducing the self-attention
mechanism. Unlike RNNs, transformers process all tokens in parallel,
leading to significant training speedups.
"""

# Extract key points
bullets = model.extract_bullets(passage)

# Generate question-answer pairs
pairs = model.generate_qa_pairs(passage)

# Extract knowledge graph triplets
triplets = model.extract_triplets(passage)
```

## API

| Method | Input | Output |
|---|---|---|
| `extract_bullets(passage)` | passage | `list[str]` |
| `generate_qa_pairs(passage)` | passage | `list[QAPair]` |
| `generate_question(passage)` | passage | `str` |
| `extract_fact(passage)` | passage | `str` |
| `answer(question, passage)` | question + passage | `str` |
| `rephrase(passage)` | passage | `str` |
| `continue_from(passage)` | passage start | `str` |
| `extract_triplets(passage)` | passage | `list[Triplet]` |
| `compare(passage_a, passage_b)` | two passages | `str` |
| `find_relevant(question, passages)` | question + passage list | `RetrievalResult` |

## Models

| Backend | Default model |
|---|---|
| `hf` | `paperbd/smollm_135M_neuraltxt_v1` |
| `mlx` | `paperbd/smollm_135M_neuraltxt_mlx_v1` |

Pass a custom path: `NeuralTxt("path/to/model", backend="hf")`

## Gradio demo

```bash
# MLX
uv run app.py paperbd/smollm_135M_neuraltxt_mlx_v1 --mlx

# HuggingFace
uv run app.py paperbd/smollm_135M_neuraltxt_v1

# Options
#   --temperature 0.4    sampling temperature (default 0.4)
#   --n 2                parallel generations, 1-4 (default 2)
```
