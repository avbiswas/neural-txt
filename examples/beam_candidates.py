"""
Using num_beams with NeuralTxt.

Install:
    uv pip install -e ".[hf]"

Notes:
    Public task methods still return one parsed result. With the HuggingFace
    backend, num_beams runs beam search internally and returns the top candidate.
    Use _backend.generate_many only when you explicitly want to inspect every
    generated candidate.
"""

from neuraltxt import NeuralTxt
from neuraltxt.tasks import BULLETS_INSTRUCTION, SYSTEM_PROMPT


model = NeuralTxt(backend="hf")

passage = """
Transformers use self-attention to compare tokens across a sequence. This lets
the model process tokens in parallel and capture long-range dependencies more
efficiently than recurrent architectures.
"""

# Public API: returns one parsed result, even when num_beams > 1.
bullets = model.extract_bullets(passage, num_beams=4)
print("=== Top Beam Parsed as Bullets ===")
for bullet in bullets:
    print(f"  - {bullet}")

# Lower-level inspection: return all beam candidates as raw text.
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": f"{BULLETS_INSTRUCTION}\n\n{passage}"},
]
prompt = model._backend.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

candidates = model._backend.generate_many(prompt, num_beams=4)
print("\n=== Raw Beam Candidates ===")
for i, candidate in enumerate(candidates, start=1):
    print(f"\n--- Candidate {i} ---")
    print(candidate.strip())
