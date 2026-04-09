# neural-txt

Structured NLP tasks powered by a fine-tuned 135M parameter language model. Extract bullets, generate Q&A pairs, build knowledge graphs, and more — all running locally. Narrow vertical local intelligence that runs super cheaply in resource constrained envs.

https://github.com/user-attachments/assets/04774af0-dc51-42e7-b2a6-d6f50bf4e258



## Support

If you find this helpful, consider supporting on Patreon — it hosts all code, projects, slides, and write-ups from the YouTube channel.

[<img src="https://c5.patreon.com/external/logo/become_a_patron_button.png" alt="Become a Patron!" width="200">](https://www.patreon.com/NeuralBreakdownwithAVB)


## Install

```bash
# Base (no inference backend)
pip install neuraltxt

# With HuggingFace backend (torch)
pip install neuraltxt[hf]

# With MLX backend (Apple Silicon)
pip install neuraltxt[mlx]
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

## Models

| Backend | Default model |
|---|---|
| `hf` | [`paperbd/smollm_135M_neuraltxt_v1`](https://huggingface.co/paperbd/smollm_135M_neuraltxt_v1) |
| `mlx` | [`paperbd/smollm_135M_neuraltxt_mlx_v1`](https://huggingface.co/paperbd/smollm_135M_neuraltxt_mlx_v1) |

Pass a custom path: `NeuralTxt("path/to/model", backend="hf")`

- Training dataset: [`paperbd/paper_instructions_300K-v1`](https://huggingface.co/datasets/paperbd/paper_instructions_300K-v1)
- Synthetic data generation: [`text-albumentations`](https://github.com/avbiswas/text-albumentations)

## Gradio demo

```bash
pip install neuraltxt[app]

# HuggingFace (default)
python app.py

# MLX (Apple Silicon)
python app.py --mlx

# Options
#   --temperature 0.4    sampling temperature (default 0.4)
#   -n 2                 parallel generations, 1-4 (default 2)
```
