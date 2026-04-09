"""
JSON mode — guaranteed structured output via outlines.
Works with both HF and MLX backends.

Install:
    pip install neuraltxt[hf]   # or neuraltxt[mlx]
"""

from neuraltxt import NeuralTxt

model = NeuralTxt(backend="hf")

passage = """
Transformers have revolutionized natural language processing by introducing
the self-attention mechanism, which allows models to weigh the importance of
different parts of the input sequence. Unlike recurrent neural networks,
transformers process all tokens in parallel, leading to significant speedups
during training. The original transformer architecture was introduced in the
paper "Attention Is All You Need" by Vaswani et al. in 2017.
"""

# ── Bullets ──────────────────────────────────────────────────────────────────
result = model.extract_bullets(passage, json=True)
print("=== Bullets (JSON) ===")
print(f"  Type: {type(result).__name__}")
for b in result.bullets:
    print(f"  - {b}")

# ── QA Pairs ─────────────────────────────────────────────────────────────────
result = model.generate_qa_pairs(passage, json=True)
print("\n=== QA Pairs (JSON) ===")
for pair in result.pairs:
    print(f"  Q: {pair.question}")
    print(f"  A: {pair.answer}\n")

# ── Triplets ─────────────────────────────────────────────────────────────────
result = model.extract_triplets(passage, json=True)
print("=== Triplets (JSON) ===")
for t in result.triplets:
    print(f"  ({t.subject}, {t.relation}, {t.object})")

# ── Single-value outputs ─────────────────────────────────────────────────────
q = model.generate_question(passage, json=True)
print(f"\n=== Question (JSON) ===\n  {q.question}")

f = model.extract_fact(passage, json=True)
print(f"\n=== Fact (JSON) ===\n  {f.fact}")

a = model.answer("What mechanism do transformers use?", passage, json=True)
print(f"\n=== Answer (JSON) ===\n  {a.answer}")

# ── Comparison ───────────────────────────────────────────────────────────────
passage_a = "Neural networks learn representations through backpropagation."
passage_b = "Decision trees split data based on feature thresholds."

c = model.compare(passage_a, passage_b, json=True)
print(f"\n=== Comparison (JSON) ===\n  {c.comparison}")
