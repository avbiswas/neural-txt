# Exact instruction strings from training data (text_albumentations tasks).
# These must not be changed — the model was trained on these exact prompts.

SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
    """

REASONING_SYSTEM_PROMPT = """You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.
You are an expert in AI, deep learning, and machine learning research and its applications.
Your answers are concise and helps directly solve any user query truthfully.
If you do not know the answer, you will inform the user that you do not know instead of making answers up.
Generate your reasoning first inside <think> and </think> tags. After </think>, generate only the requested final response.
When a structured format such as JSON is requested, the content after </think> must contain only that format, without Markdown fences or additional commentary.
    """

# ── Bullets ───────────────────────────────────────────────────────────────────

BULLETS_INSTRUCTION = (
    "Extract the important points from this passage as markdown bullet points."
)

# ── QA pairs ──────────────────────────────────────────────────────────────────

QA_PAIRS_INSTRUCTION = (
    "\nGiven this passage of text, generate a list of important question answer pairs.\n    "
)

QUESTION_FROM_PASSAGE_INSTRUCTION = "Generate a question from this passage"

QUESTIONS_LIST_INSTRUCTION = (
    "Generate a set of questions from this passage in markdown format."
)

FACT_FROM_PASSAGE_INSTRUCTION = (
    "Generate an important fact or piece of information from this passage"
)

# ── QA answering ──────────────────────────────────────────────────────────────

QA_ANSWER_INSTRUCTION = "Answer the user's question given the provided passage"

def build_qa_answer_input(passage: str, question: str) -> str:
    return f"Passage: {passage}\n\nQuestion: {question}\nWhat is the answer?"

# ── Rephrase ──────────────────────────────────────────────────────────────────

REPHRASE_INSTRUCTION = (
    "\nGiven this passage, rephrase it. Elaborate on the sentences by explaining the meaning. "
    "Only present content that is strictly present in the passage, do not introduce new concepts "
    "outside the scope of this input. Do not re-quote the original. Only generate answers.\n    "
)

# ── Continuation ──────────────────────────────────────────────────────────────

CONTINUATION_INSTRUCTION = (
    "You are given the beginning of a passage. "
    "Continue the passage by generating all remaining text after the provided beginning. "
    "Do not repeat the provided beginning."
)

# ── Triplets ──────────────────────────────────────────────────────────────────

TRIPLETS_INSTRUCTION = (
    "Extract knowledge graph triplets from this passage in markdown format."
)

# ── Comparison ────────────────────────────────────────────────────────────────

COMPARISON_INSTRUCTION = (
    "\nGiven 2 passages of text, generate a detailed comparison of the two\n    "
)

def build_comparison_input(passage_a: str, passage_b: str) -> str:
    return f"Passage 1:\n{passage_a}\n\nPassage 2:\n{passage_b}"

# ── Retrieval ─────────────────────────────────────────────────────────────────

RETRIEVAL_INSTRUCTION = (
    "Read the passages and identify which passage answers the question. "
    "Return the passage number and a short justification in markdown."
)

def build_retrieval_input(question: str, passages: list[str]) -> str:
    formatted = "\n\n".join(
        f"Passage {i+1}:\n{p}" for i, p in enumerate(passages)
    )
    return f"{formatted}\n\nQuestion: {question}"


# ── JSON-mode instruction variants ──────────────────────────────────────────
# These must be the EXACT default instructions the model was trained on
# (paperbd/paper_instructions_300K-v1). The string "Respond in JSON" never
# appears in the training data — this is a prompt-sensitive SLM, so the JSON
# instruction must convey the task using a phrasing the model actually saw;
# structural validity is enforced separately by outlines constrained decoding.
#
# Tasks that have a dedicated JSON training form use it. Tasks whose JSON
# adapter reused the text instruction (question, answer) keep the text default.
# bullets has no JSON form but does have a structured "Python list" form.
# rephrase / continuation / comparison are free-text tasks with NO structured
# form in training; JSON mode is not supported for them (callers should refuse),
# so their constants mirror the text default and must not be relied upon.

BULLETS_INSTRUCTION_JSON = (
    "Extract the important points from this passage as a Python list of strings."
)

QA_PAIRS_INSTRUCTION_JSON = (
    "\nGiven this passage of text, generate a list of important question answer pairs.\n    Generate as a list of json containing 'question' and 'answer' keys"
)

QUESTION_FROM_PASSAGE_INSTRUCTION_JSON = (
    "Generate a question from this passage"
)

QUESTIONS_LIST_INSTRUCTION_JSON = (
    "Generate a list of questions from this passage. Return a JSON array of strings."
)

FACT_FROM_PASSAGE_INSTRUCTION_JSON = (
    "Generate an important fact or piece of information from this passage"
)

QA_ANSWER_INSTRUCTION_JSON = (
    "Answer the user's question given the provided passage"
)

# No structured form in training — JSON mode unsupported (mirrors text default).
REPHRASE_INSTRUCTION_JSON = REPHRASE_INSTRUCTION

# No structured form in training — JSON mode unsupported (mirrors text default).
CONTINUATION_INSTRUCTION_JSON = CONTINUATION_INSTRUCTION

TRIPLETS_INSTRUCTION_JSON = (
    "Extract knowledge graph triplets from this passage and return them as JSON."
)

# No structured form in training — JSON mode unsupported (mirrors text default).
COMPARISON_INSTRUCTION_JSON = COMPARISON_INSTRUCTION

RETRIEVAL_INSTRUCTION_JSON = (
    "Read the passages and identify which passage answers the question. "
    "Return the passage number and a short justification in JSON."
)
