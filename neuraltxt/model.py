from __future__ import annotations

from .backends import load_backend, Backend
from .tasks import (
    SYSTEM_PROMPT,
    REASONING_SYSTEM_PROMPT,
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
    split_thinking,
    strip_thinking,
    parse_bullets,
    parse_questions_list,
    parse_qa_pairs,
    parse_triplets,
    parse_retrieval,
)
from .types import (
    QAPair, Triplet, RetrievalResult, ReasonedOutput,
    ShortText,
    BulletsOutput, QAPairsOutput, QuestionOutput, QuestionsListOutput, FactOutput,
    AnswerOutput, RephraseOutput, ContinuationOutput,
    TripletsOutput, ComparisonOutput, RetrievalOutput,
)

DEFAULT_HF_MODEL = "paperbd/neuraltxt-v1-135M"
DEFAULT_MLX_MODEL = "paperbd/neuraltxt-v1-135M-mlx"
DEFAULT_HF_REASONING_MODEL = "paperbd/neuraltxt-v1-135M-reasoning"
DEFAULT_MLX_REASONING_MODEL = "paperbd/neuraltxt-v1-135M-reasoning-mlx"


class NeuralTxt:
    """
    Clean interface to the neural-txt model.
    All prompt formatting is handled internally.

    Args:
        model_path: Path to a merged HF model or MLX model directory.
                    Defaults based on the chosen backend.
        backend: "mlx" for Apple Silicon MLX, "hf" for HuggingFace Transformers.
                 Defaults to "hf".
        reasoning: Use the reasoning model variant. Plain-text outputs have
                   the leading <think>...</think> block stripped.
        return_reasoning: Return ReasonedOutput objects containing parsed
                          output, reasoning text, and raw model output.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "hf",
        reasoning: bool = False,
        return_reasoning: bool = False,
    ):
        if backend not in ("hf", "mlx"):
            raise ValueError(f"backend must be 'hf' or 'mlx', got {backend!r}")
        if return_reasoning and not reasoning:
            raise ValueError("return_reasoning=True requires reasoning=True")

        if model_path is None:
            if reasoning:
                model_path = DEFAULT_MLX_REASONING_MODEL if backend == "mlx" else DEFAULT_HF_REASONING_MODEL
            else:
                model_path = DEFAULT_MLX_MODEL if backend == "mlx" else DEFAULT_HF_MODEL

        self.reasoning = reasoning
        self.return_reasoning = return_reasoning
        self._backend: Backend = load_backend(model_path, mlx=(backend == "mlx"))

    # ── Internal ──────────────────────────────────────────────────────────────

    def _build_prompt(self, instruction: str, user_input: str) -> str:
        tokenizer = self._backend.tokenizer
        system_prompt = REASONING_SYSTEM_PROMPT if self.reasoning else SYSTEM_PROMPT
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{instruction}\n\n{user_input}"},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _preprocess(self, text: str) -> str:
        return " ".join(text.split())

    def _run(self, instruction: str, user_input: str, **kwargs) -> str:
        prompt = self._build_prompt(instruction, self._preprocess(user_input))
        return self._postprocess(self._backend.generate(prompt, **kwargs))

    def _postprocess(self, text: str) -> str:
        if self.reasoning:
            return strip_thinking(text)
        return text

    def _postprocess_many(self, texts: list[str]) -> list[str]:
        return [self._postprocess(text) for text in texts]

    def _split_raws(self, raws: list[str]) -> tuple[list[str], list[str]]:
        parts = [split_thinking(raw) for raw in raws]
        return [answer for answer, _ in parts], [reasoning for _, reasoning in parts]

    def _get_return_reasoning(self, kwargs: dict) -> bool:
        return_reasoning = kwargs.pop("return_reasoning", self.return_reasoning)
        if return_reasoning and not self.reasoning:
            raise ValueError("return_reasoning=True requires reasoning=True")
        return bool(return_reasoning)

    def _maybe_return_reasoned(
        self,
        outputs,
        raws: list[str],
        reasonings: list[str],
        rollouts: int,
        return_reasoning: bool,
    ):
        if not return_reasoning:
            return outputs if rollouts > 1 else outputs[0]
        output_list = outputs if rollouts > 1 else [outputs[0]]
        wrapped = [
            ReasonedOutput(output=output, reasoning=reasoning, raw=raw)
            for output, reasoning, raw in zip(output_list, reasonings, raws)
        ]
        return wrapped if rollouts > 1 else wrapped[0]

    def _ensure_json_supported(self) -> None:
        return None

    def _get_rollouts(self, kwargs: dict) -> int:
        rollouts = kwargs.pop("rollouts", 1)
        try:
            rollouts = int(rollouts)
        except (TypeError, ValueError):
            raise ValueError(f"rollouts must be an integer, got {rollouts!r}") from None
        if rollouts < 1:
            raise ValueError(f"rollouts must be >= 1, got {rollouts}")
        return rollouts

    def _run_many(self, instruction: str, user_input: str, rollouts: int, **kwargs) -> list[str]:
        prompt = self._build_prompt(instruction, self._preprocess(user_input))
        return self._postprocess_many(
            self._backend.generate_many(prompt, num_return_sequences=rollouts, **kwargs)
        )

    def _run_many_with_reasoning(
        self, instruction: str, user_input: str, rollouts: int, **kwargs
    ) -> tuple[list[str], list[str], list[str]]:
        prompt = self._build_prompt(instruction, self._preprocess(user_input))
        raws = self._backend.generate_many(prompt, num_return_sequences=rollouts, **kwargs)
        answers, reasonings = self._split_raws(raws) if self.reasoning else (raws, [""] * len(raws))
        return answers, reasonings, raws

    def _run_json(self, instruction_json: str, user_input: str, output_type, **kwargs) -> str:
        prompt = self._build_prompt(instruction_json, self._preprocess(user_input))
        if self.reasoning:
            return self._backend.generate_reasoned_json(prompt, output_type, **kwargs)
        return self._backend.generate_json(prompt, output_type, **kwargs)

    def _run_json_many(
        self, instruction_json: str, user_input: str, output_type, rollouts: int, **kwargs
    ) -> list[str]:
        prompt = self._build_prompt(instruction_json, self._preprocess(user_input))
        if self.reasoning:
            return [
                self._backend.generate_reasoned_json(prompt, output_type, **kwargs)
                for _ in range(rollouts)
            ]
        return [
            self._backend.generate_json(prompt, output_type, **kwargs)
            for _ in range(rollouts)
        ]

    def _json_answers_from_raws(self, raws: list[str]) -> tuple[list[str], list[str]]:
        if self.reasoning:
            return self._split_raws(raws)
        return raws, [""] * len(raws)

    # ── Public API ────────────────────────────────────────────────────────────

    def extract_bullets(self, passage: str, json: bool = False, **kwargs) -> list[str] | BulletsOutput:
        """Extract key points from a passage as a list of strings."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            # The JSON instruction asks for a bare "Python list of strings" (the
            # in-distribution form), so constrain to list[str] and wrap into
            # BulletsOutput — matching the schema to the prompt. Forcing the
            # {"bullets": [...]} object schema here makes the model hallucinate
            # label-like junk entries.
            import json as json_mod
            raws = self._run_json_many(BULLETS_INSTRUCTION_JSON, passage, list[ShortText], rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [BulletsOutput(bullets=json_mod.loads(answer)) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(BULLETS_INSTRUCTION, passage, rollouts, **kwargs)
        outputs = [parse_bullets(answer) for answer in answers]
        return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)

    def generate_qa_pairs(self, passage: str, json: bool = False, **kwargs) -> list[QAPair] | QAPairsOutput:
        """Generate question-answer pairs from a passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            raws = self._run_json_many(QA_PAIRS_INSTRUCTION_JSON, passage, QAPairsOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [QAPairsOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(QA_PAIRS_INSTRUCTION, passage, rollouts, **kwargs)
        outputs = [parse_qa_pairs(answer) for answer in answers]
        return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)

    def generate_question(self, passage: str, json: bool = False, **kwargs) -> str | QuestionOutput:
        """Generate a single question from a passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            raws = self._run_json_many(QUESTION_FROM_PASSAGE_INSTRUCTION_JSON, passage, QuestionOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [QuestionOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(
            QUESTION_FROM_PASSAGE_INSTRUCTION, passage, rollouts, **kwargs
        )
        return self._maybe_return_reasoned(answers, raws, reasonings, rollouts, return_reasoning)

    def generate_questions_list(self, passage: str, json: bool = False, **kwargs) -> list[str] | QuestionsListOutput:
        """Generate a list of questions from a passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            import json as json_mod
            raws = self._run_json_many(QUESTIONS_LIST_INSTRUCTION_JSON, passage, list[ShortText], rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [QuestionsListOutput(questions=json_mod.loads(answer)) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(QUESTIONS_LIST_INSTRUCTION, passage, rollouts, **kwargs)
        outputs = [parse_questions_list(answer) for answer in answers]
        return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)

    def extract_fact(self, passage: str, json: bool = False, **kwargs) -> str | FactOutput:
        """Extract a single important fact from a passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            raws = self._run_json_many(FACT_FROM_PASSAGE_INSTRUCTION_JSON, passage, FactOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [FactOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(FACT_FROM_PASSAGE_INSTRUCTION, passage, rollouts, **kwargs)
        return self._maybe_return_reasoned(answers, raws, reasonings, rollouts, return_reasoning)

    def answer(self, question: str, passage: str, json: bool = False, **kwargs) -> str | AnswerOutput:
        """Answer a question given a supporting passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        user_input = build_qa_answer_input(self._preprocess(passage), self._preprocess(question))
        if json:
            raws = self._run_json_many(QA_ANSWER_INSTRUCTION_JSON, user_input, AnswerOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [AnswerOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        prompt = self._build_prompt(QA_ANSWER_INSTRUCTION, user_input)
        raws = self._backend.generate_many(prompt, num_return_sequences=rollouts, **kwargs)
        answers, reasonings = self._split_raws(raws) if self.reasoning else (raws, [""] * len(raws))
        return self._maybe_return_reasoned(answers, raws, reasonings, rollouts, return_reasoning)

    def rephrase(self, passage: str, json: bool = False, **kwargs) -> str | RephraseOutput:
        """Rephrase and elaborate a passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            raws = self._run_json_many(REPHRASE_INSTRUCTION_JSON, passage, RephraseOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [RephraseOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(REPHRASE_INSTRUCTION, passage, rollouts, **kwargs)
        return self._maybe_return_reasoned(answers, raws, reasonings, rollouts, return_reasoning)

    def continue_from(self, passage_start: str, json: bool = False, **kwargs) -> str | ContinuationOutput:
        """Generate a continuation from the beginning of a passage."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            raws = self._run_json_many(CONTINUATION_INSTRUCTION_JSON, passage_start, ContinuationOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [ContinuationOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(CONTINUATION_INSTRUCTION, passage_start, rollouts, **kwargs)
        return self._maybe_return_reasoned(answers, raws, reasonings, rollouts, return_reasoning)

    def extract_triplets(self, passage: str, json: bool = False, **kwargs) -> list[Triplet] | TripletsOutput:
        """Extract knowledge graph (subject, relation, object) triplets."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        if json:
            raws = self._run_json_many(TRIPLETS_INSTRUCTION_JSON, passage, TripletsOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [TripletsOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        answers, reasonings, raws = self._run_many_with_reasoning(TRIPLETS_INSTRUCTION, passage, rollouts, **kwargs)
        outputs = [parse_triplets(answer) for answer in answers]
        return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)

    def compare(self, passage_a: str, passage_b: str, json: bool = False, **kwargs) -> str | ComparisonOutput:
        """Generate a detailed comparison of two passages."""
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        user_input = build_comparison_input(self._preprocess(passage_a), self._preprocess(passage_b))
        if json:
            raws = self._run_json_many(COMPARISON_INSTRUCTION_JSON, user_input, ComparisonOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [ComparisonOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        prompt = self._build_prompt(COMPARISON_INSTRUCTION, user_input)
        raws = self._backend.generate_many(prompt, num_return_sequences=rollouts, **kwargs)
        answers, reasonings = self._split_raws(raws) if self.reasoning else (raws, [""] * len(raws))
        return self._maybe_return_reasoned(answers, raws, reasonings, rollouts, return_reasoning)

    def find_relevant(
        self, question: str, passages: list[str], json: bool = False, **kwargs
    ) -> RetrievalResult | RetrievalOutput:
        """
        Identify which passage answers the question.
        Returns a RetrievalResult with .index (0-based) and .reasoning.
        .index is None if no passage answers the question.
        """
        rollouts = self._get_rollouts(kwargs)
        return_reasoning = self._get_return_reasoning(kwargs)
        user_input = build_retrieval_input(self._preprocess(question), [self._preprocess(p) for p in passages])
        if json:
            raws = self._run_json_many(RETRIEVAL_INSTRUCTION_JSON, user_input, RetrievalOutput, rollouts, **kwargs)
            answers, reasonings = self._json_answers_from_raws(raws)
            outputs = [RetrievalOutput.model_validate_json(answer) for answer in answers]
            return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
        prompt = self._build_prompt(RETRIEVAL_INSTRUCTION, user_input)
        raws = self._backend.generate_many(prompt, num_return_sequences=rollouts, **kwargs)
        answers, reasonings = self._split_raws(raws) if self.reasoning else (raws, [""] * len(raws))
        outputs = [parse_retrieval(answer, num_passages=len(passages)) for answer in answers]
        return self._maybe_return_reasoned(outputs, raws, reasonings, rollouts, return_reasoning)
