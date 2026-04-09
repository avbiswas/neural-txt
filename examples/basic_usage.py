"""
Basic usage of NeuralTxt with the HuggingFace backend.

Install:
    uv pip install -e ".[hf]"
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

# Extract bullet points
bullets = model.extract_bullets(passage)
print("=== Bullets ===")
for b in bullets:
    print(f"  - {b}")

# Generate QA pairs
pairs = model.generate_qa_pairs(passage)
print("\n=== QA Pairs ===")
for pair in pairs:
    print(f"  Q: {pair.question}")
    print(f"  A: {pair.answer}\n")

# Generate a question
question = model.generate_question(passage)
print(f"=== Generated Question ===\n  {question}")

# Answer a question
answer = model.answer("What mechanism do transformers use?", passage)
print(f"\n=== Answer ===\n  {answer}")

# Extract knowledge graph triplets
triplets = model.extract_triplets(passage)
print("\n=== Triplets ===")
for t in triplets:
    print(f"  {t}")

# Rephrase
rephrased = model.rephrase(passage)
print(f"\n=== Rephrased ===\n  {rephrased}")
