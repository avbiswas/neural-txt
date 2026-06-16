# neural-txt

Structured NLP tasks powered by a fine-tuned 135M parameter language model. Extract bullets, generate Q&A pairs, build knowledge graphs, and more — all running locally. Narrow vertical local intelligence that runs super cheaply in resource constrained envs.

https://github.com/user-attachments/assets/04774af0-dc51-42e7-b2a6-d6f50bf4e258



## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)


## Install

```bash
# Base (no inference backend)
pip install neural-txt

# With HuggingFace backend (torch)
pip install neural-txt[hf]

# With MLX backend (Apple Silicon)
pip install neural-txt[mlx]
```

`NeuralTxtReward` works with either backend: install `neural-txt[hf]` for the
Hugging Face / torch scorer, or `neural-txt[mlx]` for Apple Silicon MLX.

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

Use the reasoning model variant with `reasoning=True`:

```python
model = NeuralTxt(backend="mlx", reasoning=True)  # or backend="hf"
answer = model.answer("What mechanism do transformers use?", passage)
```

Reasoning models emit `<think>...</think>{answer}` internally. NeuralTxt strips
the leading reasoning block for plain-text methods. In JSON mode, NeuralTxt
generates the reasoning block first, then uses Outlines constrained decoding for
the JSON answer.
`NeuralTxt(reasoning=True)` also switches to the reasoning model system prompt,
which explicitly asks for `<think>...</think>` reasoning followed by only the
requested final response.

To keep the reasoning trace, pass `return_reasoning=True`:

```python
model = NeuralTxt(backend="mlx", reasoning=True, return_reasoning=True)
result = model.answer("What mechanism do transformers use?", passage)

print(result.output)
print(result.reasoning)
```

With `return_reasoning=True`, generation methods return `ReasonedOutput`
objects containing the normal output, reasoning text, and raw model text. With
`rollouts > 1`, they return a list of `ReasonedOutput` objects. You can also
pass `return_reasoning=True` to a single method call.

There is also a short runnable example:

```bash
HF_HOME=.hf-cache uv run python scripts/reasoning_usage.py
HF_HOME=.hf-cache uv run python scripts/reasoning_usage.py --mlx --json
```

## Reward scoring

`NeuralTxtReward` scores generated responses against a reference answer with
[`paperbd/neuraltxt-reward-tiny`](https://huggingface.co/paperbd/neuraltxt-reward-tiny).
Use it to score one answer, score a batch, or rank candidate responses.

```python
from neuraltxt import NeuralTxtReward

rm = NeuralTxtReward(backend="mlx")  # or backend="hf"

score = rm.score(
    response="Attention is all you need.",
    reference="All you need is attention.",
)

print(score)
# 0.860448
```

You can also score batches and rank responses:

```python
reference = "Attention is all you need."
responses = [
    "All you need is attention.",
    "You do not need attention.",
]

scores = rm.batch_score(responses, reference)
ranked = rm.rank(responses, reference)

print(scores)
# [0.885680, 0.396632]

for item in ranked:
    print(item.index, item.score, item.response)

# 0 0.885680 All you need is attention.
# 1 0.396632 You do not need attention.
```

`batch_score()` scores responses in chunks of 64 by default. Pass
`batch_size=` to tune memory use. Pass a list of references to score
corresponding `(response, reference)` pairs; the list length must match
`responses`. `rank()` preserves the original response index and sorts highest score first.
Pass a local model directory with `NeuralTxtReward("path/to/reward-model")`.

## Multiple rollouts

Every generation method accepts `rollouts`. The default is `1`, which preserves
the usual single-output API. Set `rollouts > 1` to get a list of parsed outputs.

```python
answers = model.answer(
    question="What mechanism do transformers use?",
    passage=passage,
    temperature=0.7,
    rollouts=4,
)

for answer in answers:
    print(answer)
```

`num_beams` is still available as a decoding strategy. Use `rollouts` when you
want multiple returned outputs; use `num_beams` when you want beam search.

## JSON mode

Every method supports `json=True` for guaranteed structured output via [outlines](https://github.com/dottxt-ai/outlines):

```python
# Returns a BulletsOutput pydantic model
bullets = model.extract_bullets(passage, json=True)
print(bullets.bullets)  # list[str]

# Returns a QAPairsOutput pydantic model
qa = model.generate_qa_pairs(passage, json=True)
for pair in qa.pairs:
    print(pair.question, pair.answer)

# Returns a TripletsOutput pydantic model
triplets = model.extract_triplets(passage, json=True)
for t in triplets.triplets:
    print(t.subject, t.relation, t.object)
```

## API

### Generation API

| Method | Input | Output | JSON Output |
|---|---|---|---|
| `extract_bullets(passage)` | passage | `list[str]` | `BulletsOutput` |
| `generate_qa_pairs(passage)` | passage | `list[QAPair]` | `QAPairsOutput` |
| `generate_question(passage)` | passage | `str` | `QuestionOutput` |
| `generate_questions_list(passage)` | passage | `list[str]` | `QuestionsListOutput` |
| `extract_fact(passage)` | passage | `str` | `FactOutput` |
| `answer(question, passage)` | question + passage | `str` | `AnswerOutput` |
| `rephrase(passage)` | passage | `str` | `RephraseOutput` |
| `continue_from(passage)` | passage start | `str` | `ContinuationOutput` |
| `extract_triplets(passage)` | passage | `list[Triplet]` | `TripletsOutput` |
| `compare(passage_a, passage_b)` | two passages | `str` | `ComparisonOutput` |
| `find_relevant(question, passages)` | question + passage list | `RetrievalResult` | `RetrievalOutput` |

### Reward API

| Method | Input | Output |
|---|---|---|
| `score(response, reference)` | one response + reference answer | `float` |
| `batch_score(responses, reference, batch_size=64)` | response list + one reference or paired references | `list[float]` |
| `rank(responses, reference)` | response list + one reference or paired references | `list[RankedResponse]` |

`NeuralTxtReward` accepts `backend="hf"` or `backend="mlx"`.

## Models

| Interface | Default model |
|---|---|
| `NeuralTxt(backend="hf")` | [`paperbd/neuraltxt-v1-135M`](https://huggingface.co/paperbd/neuraltxt-v1-135M) |
| `NeuralTxt(backend="mlx")` | [`paperbd/neuraltxt-v1-135M-mlx`](https://huggingface.co/paperbd/neuraltxt-v1-135M-mlx) |
| `NeuralTxt(backend="hf", reasoning=True)` | [`paperbd/neuraltxt-v1-135M-reasoning`](https://huggingface.co/paperbd/neuraltxt-v1-135M-reasoning) |
| `NeuralTxt(backend="mlx", reasoning=True)` | [`paperbd/neuraltxt-v1-135M-reasoning-mlx`](https://huggingface.co/paperbd/neuraltxt-v1-135M-reasoning-mlx) |
| `NeuralTxtReward(backend="hf")` | [`paperbd/neuraltxt-reward-tiny`](https://huggingface.co/paperbd/neuraltxt-reward-tiny) |
| `NeuralTxtReward(backend="mlx")` | [`paperbd/neuraltxt-reward-tiny-mlx`](https://huggingface.co/paperbd/neuraltxt-reward-tiny-mlx) |

Pass a custom path: `NeuralTxt("path/to/model", backend="hf")`

- Training dataset: [`paperbd/paper_instructions_300K-v1`](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1)
- Synthetic data generation: [`text-albumentations`](https://github.com/avbiswas/text-albumentations)

## Gradio demo

```bash
pip install neural-txt[app]

# HuggingFace (default)
python app.py

# MLX (Apple Silicon)
python app.py --mlx

# Reasoning model
python app.py --reasoning
python app.py --mlx --reasoning

# Options
#   --temperature 0.4    sampling temperature (default 0.4)
#   --num-beams 2        beam candidates, 1-4 (default 1)
```

When the Gradio app runs with `--reasoning`, each output candidate shows the
model's reasoning trace in a light italic gray block above the final output.

## Terminal UI (TUI)

A keyboard-driven terminal app (built with [Textual](https://textual.textualize.io/))
that mirrors the Gradio demo — task grid, live token streaming, color-coded
reasoning trace, and token/throughput/memory stats.

```bash
pip install neural-txt[tui]

# MLX (default, Apple Silicon)
python tui.py

# HuggingFace
python tui.py --hf

# Reasoning model
python tui.py --reasoning

# Options
#   --temperature 0.4    sampling temperature (default 0.4)
#   -n 2                 candidates to generate, 1-4 (default 1)
```

- Pick a task with the **arrow keys**; `answer`/`comparison` reveal a second input.
- **Enter** (or `Ctrl+R`) generates, `f` toggles text/JSON, `Ctrl+L` clears, `Esc` unfocuses the editor.
- Text and JSON both stream token-by-token; reasoning-model output shows the
  `<think>…</think>` trace dimmed above the answer.

For quick manual testing without the UI, edit and run `playground.py`.
