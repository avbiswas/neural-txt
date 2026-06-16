"""
Simple manual test script (MLX + reasoning). Edit the variables below and run:
    uv run playground.py
"""
from neuraltxt import NeuralTxt
import sys
# ── edit these ───────────────────────────────────────────────────────────────
MODE = sys.argv[1]     # bullets, qa_pairs, question, questions_list, fact,
                      # answer, rephrase, continuation, triplets, comparison
JSON = False if len(sys.argv) > 2 else True
ROLLOUTS = 1          # number of candidates ("multiple responses")
TEMPERATURE = 0.5

PASSAGE = """MiniMax-M1 is trained using large-scale reinforcement learning (RL) on diverse problems including sandbox-based, real-world software engineering environments. In addition to M1's inherent efficiency advantage for RL training, we propose CISPO, a novel RL algorithm to further enhance RL efficiency. CISPO clips importance sampling weights rather than token updates, outperforming other competitive RL variants. Combining hybrid-attention and CISPO enables MiniMax-M1's full RL training on 512 H800 GPUs to complete in only three weeks, with a rental cost of just $534,700. We release two versions of MiniMax-M1 models with 40K and 80K thinking budgets respectively, where the 40K model represents an intermediate phase of the 80K training. Experiments on standard benchmarks show that our models are comparable or superior to strong open-weight models such as the original DeepSeek-R1 and Qwen3-235B, with particular strengths in complex software engineering, tool utilization, and long-context tasks"""

PASSAGE_2 = "What is CISPO?"

METHOD = {
    "bullets": "extract_bullets", 
    "qa_pairs": "generate_qa_pairs",
    "question": "generate_question",
    "questions_list": "generate_questions_list",
    "fact": "extract_fact", 
    "answer": "answer", 
    "rephrase": "rephrase",
    "continuation": "continue_from", 
    "triplets": "extract_triplets",
    "comparison": "compare",
}

r = NeuralTxt(backend="mlx", reasoning=True)
method = getattr(r, METHOD[MODE])
kwargs = dict(json=JSON, rollouts=ROLLOUTS, temperature=TEMPERATURE,
              return_reasoning=True)

if MODE == "answer":
    result = method(PASSAGE_2, PASSAGE, **kwargs)   # answer(question, passage)
elif MODE == "comparison":
    result = method(PASSAGE, PASSAGE_2, **kwargs)   # compare(a, b)
else:
    result = method(PASSAGE, **kwargs)

items = result if ROLLOUTS > 1 else [result]
for i, item in enumerate(items):
    print(f"\n===== candidate [{i}] =====")
    print(repr(item))

print(f"\nstats: {getattr(r._backend, '_last_stats', {})}")
