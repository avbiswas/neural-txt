from __future__ import annotations

from .backends import load_backend, Backend
from .tasks import (
    SYSTEM_PROMPT,
    BULLETS_INSTRUCTION,
    QA_PAIRS_INSTRUCTION,
    QUESTION_FROM_PASSAGE_INSTRUCTION,
    QUESTIONS_LIST_INSTRUCTION,
    FACT_FROM_PASSAGE_INSTRUCTION,
    QA_ANSWER_INSTRUCTION,
    REPHRASE_INSTRUCTION,
    CONTINUATION_INSTRUCTION,
    TRIPLETS_INSTRUCTION,
    COMPARISON_INSTRUCTION,
    RETRIEVAL_INSTRUCTION,
    BULLETS_INSTRUCTION_JSON,
    QA_PAIRS_INSTRUCTION_JSON,
    QUESTION_FROM_PASSAGE_INSTRUCTION_JSON,
    QUESTIONS_LIST_INSTRUCTION_JSON,
    FACT_FROM_PASSAGE_INSTRUCTION_JSON,
    QA_ANSWER_INSTRUCTION_JSON,
    REPHRASE_INSTRUCTION_JSON,
    CONTINUATION_INSTRUCTION_JSON,
    TRIPLETS_INSTRUCTION_JSON,
    COMPARISON_INSTRUCTION_JSON,
    RETRIEVAL_INSTRUCTION_JSON,
    build_qa_answer_input,
    build_comparison_input,
    build_retrieval_input,
)
from .parsing import (
    parse_bullets,
    parse_questions_list,
    parse_qa_pairs,
    parse_triplets,
    parse_retrieval,
)
from .types import (
    QAPair, Triplet, RetrievalResult,
    BulletsOutput, QAPairsOutput, QuestionOutput, QuestionsListOutput, FactOutput,
    AnswerOutput, RephraseOutput, ContinuationOutput,
    TripletsOutput, ComparisonOutput, RetrievalOutput,
)

DEFAULT_HF_MODEL = "paperbd/smollm_135M_neuraltxt_v1"
DEFAULT_MLX_MODEL = "paperbd/smollm_135M_neuraltxt_mlx_v1"


class NeuralTxt:
    """
    Clean interface to the neural-txt model.
    All prompt formatting is handled internally.

    Args:
        model_path: Path to a merged HF model or MLX model directory.
                    Defaults based on the chosen backend.
        backend: "mlx" for Apple Silicon MLX, "hf" for HuggingFace Transformers.
                 Defaults to "hf".
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "hf",
    ):
        if backend not in ("hf", "mlx"):
            raise ValueError(f"backend must be 'hf' or 'mlx', got {backend!r}")

        if model_path is None:
            model_path = DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_HF_MODEL

        self._backend: Backend = load_backend(model_path, mlx=(backend == "mlx"))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        tokenizer = self._backend.tokenizer
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\n{user_input}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _preprocess(self, text: str) -> str:
        return " ".join(text.split())

    def _run(self, instruction: str, user_input: str, **kwargs) -> str:
        prompt = self._build_prompt(instruction, self._preprocess(user_input))
        return self._backend.generate(prompt, **kwargs)

    def _run_json(self, instruction_json: str, user_input: str, output_type, **kwargs) -> str:
        prompt = self._build_prompt(instruction_json, self._preprocess(user_input))
        return self._backend.generate_json(prompt, output_type, **kwargs)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_bullets(self, passage: str, json: bool = False, **kwargs) -> list[str] | BulletsOutput:
        """Extract key points from a passage as a list of strings."""
        if json:
            raw = self._run_json(BULLETS_INSTRUCTION_JSON, passage, BulletsOutput, **kwargs)
            return BulletsOutput.model_validate_json(raw)
        raw = self._run(BULLETS_INSTRUCTION, passage, **kwargs)
        return parse_bullets(raw)

    def generate_qa_pairs(self, passage: str, json: bool = False, **kwargs) -> list[QAPair] | QAPairsOutput:
        """Generate question-answer pairs from a passage."""
        if json:
            raw = self._run_json(QA_PAIRS_INSTRUCTION_JSON, passage, QAPairsOutput, **kwargs)
            return QAPairsOutput.model_validate_json(raw)
        raw = self._run(QA_PAIRS_INSTRUCTION, passage, **kwargs)
        return parse_qa_pairs(raw)

    def generate_question(self, passage: str, json: bool = False, **kwargs) -> str | QuestionOutput:
        """Generate a single question from a passage."""
        raw = self._run(QUESTION_FROM_PASSAGE_INSTRUCTION, passage, **kwargs)
        if json:
            return QuestionOutput(question=raw.strip())
        return raw

    def generate_questions_list(self, passage: str, json: bool = False, **kwargs) -> list[str] | QuestionsListOutput:
        """Generate a list of questions from a passage."""
        if json:
            import json as json_mod
            raw = self._run_json(QUESTIONS_LIST_INSTRUCTION_JSON, passage, list[str], **kwargs)
            return QuestionsListOutput(questions=json_mod.loads(raw))
        raw = self._run(QUESTIONS_LIST_INSTRUCTION, passage, **kwargs)
        return parse_questions_list(raw)

    def extract_fact(self, passage: str, json: bool = False, **kwargs) -> str | FactOutput:
        """Extract a single important fact from a passage."""
        raw = self._run(FACT_FROM_PASSAGE_INSTRUCTION, passage, **kwargs)
        if json:
            return FactOutput(fact=raw.strip())
        return raw

    def answer(self, question: str, passage: str, json: bool = False, **kwargs) -> str | AnswerOutput:
        """Answer a question given a supporting passage."""
        user_input = build_qa_answer_input(self._preprocess(passage), self._preprocess(question))
        prompt = self._build_prompt(QA_ANSWER_INSTRUCTION, user_input)
        raw = self._backend.generate(prompt, **kwargs)
        if json:
            return AnswerOutput(answer=raw.strip())
        return raw

    def rephrase(self, passage: str, json: bool = False, **kwargs) -> str | RephraseOutput:
        """Rephrase and elaborate a passage."""
        raw = self._run(REPHRASE_INSTRUCTION, passage, **kwargs)
        if json:
            return RephraseOutput(text=raw.strip())
        return raw

    def continue_from(self, passage_start: str, json: bool = False, **kwargs) -> str | ContinuationOutput:
        """Generate a continuation from the beginning of a passage."""
        raw = self._run(CONTINUATION_INSTRUCTION, passage_start, **kwargs)
        if json:
            return ContinuationOutput(text=raw.strip())
        return raw

    def extract_triplets(self, passage: str, json: bool = False, **kwargs) -> list[Triplet] | TripletsOutput:
        """Extract knowledge graph (subject, relation, object) triplets."""
        if json:
            raw = self._run_json(TRIPLETS_INSTRUCTION_JSON, passage, TripletsOutput, **kwargs)
            return TripletsOutput.model_validate_json(raw)
        raw = self._run(TRIPLETS_INSTRUCTION, passage, **kwargs)
        return parse_triplets(raw)

    def compare(self, passage_a: str, passage_b: str, json: bool = False, **kwargs) -> str | ComparisonOutput:
        """Generate a detailed comparison of two passages."""
        user_input = build_comparison_input(self._preprocess(passage_a), self._preprocess(passage_b))
        prompt = self._build_prompt(COMPARISON_INSTRUCTION, user_input)
        raw = self._backend.generate(prompt, **kwargs)
        if json:
            return ComparisonOutput(comparison=raw.strip())
        return raw

    def find_relevant(
        self, question: str, passages: list[str], json: bool = False, **kwargs
    ) -> RetrievalResult | RetrievalOutput:
        """
        Identify which passage answers the question.
        Returns a RetrievalResult with .index (0-based) and .reasoning.
        .index is None if no passage answers the question.
        """
        user_input = build_retrieval_input(self._preprocess(question), [self._preprocess(p) for p in passages])
        if json:
            prompt = self._build_prompt(RETRIEVAL_INSTRUCTION_JSON, user_input)
            raw = self._backend.generate_json(prompt, RetrievalOutput, **kwargs)
            return RetrievalOutput.model_validate_json(raw)
        prompt = self._build_prompt(RETRIEVAL_INSTRUCTION, user_input)
        raw = self._backend.generate(prompt, **kwargs)
        return parse_retrieval(raw, num_passages=len(passages))
