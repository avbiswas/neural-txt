"""
Using NeuralTxt with the MLX backend (Apple Silicon).

Install:
    uv pip install -e ".[mlx]"
"""

from neuraltxt import NeuralTxt

model = NeuralTxt(backend="mlx")

passage = """
Large language models are trained on massive datasets of text from the internet.
They learn statistical patterns in language which allow them to generate coherent
text, answer questions, and perform a wide variety of NLP tasks. Fine-tuning these
models on domain-specific data can significantly improve their performance on
specialized tasks.
"""

# Extract bullets
bullets = model.extract_bullets(passage)
print("=== Bullets ===")
for b in bullets:
    print(f"  - {b}")

# Compare two passages
passage_a = "Neural networks learn representations through backpropagation."
passage_b = "Decision trees split data based on feature thresholds."

comparison = model.compare(passage_a, passage_b)
print(f"\n=== Comparison ===\n{comparison}")

# Find which passage answers a question
result = model.find_relevant(
    question="How do neural networks learn?",
    passages=[passage_a, passage_b],
)
print(f"\n=== Retrieval ===")
print(f"  Passage index: {result.index}")
print(f"  Reasoning: {result.reasoning}")
