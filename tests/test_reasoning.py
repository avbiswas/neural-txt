import pytest
from typing import get_origin

import neuraltxt.model as model_mod
from neuraltxt import NeuralTxt, ReasonedOutput
from neuraltxt.parsing import split_thinking, strip_thinking


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        self.messages = messages
        return "\n".join(message["content"] for message in messages)


class DummyBackend:
    tokenizer = DummyTokenizer()

    def __init__(self, outputs=None):
        self.outputs = outputs or ["<think>hidden reasoning</think>Final answer"]

    def generate(self, prompt, **kwargs):
        return self.outputs[0]

    def generate_many(self, prompt, **kwargs):
        count = kwargs.get("num_return_sequences", 1)
        return self.outputs[:count]

    def generate_json(self, prompt, output_type, **kwargs):
        if get_origin(output_type) is list:
            return '["What is self-attention?"]'
        return output_type.model_json_schema() and '{"question":"What is self-attention?"}'

    def generate_reasoned_json(self, prompt, output_type, **kwargs):
        if get_origin(output_type) is list:
            return '<think>make a JSON list</think>["What is self-attention?"]'
        if output_type.__name__ == "BulletsOutput":
            return '<think>extract bullets</think>{"bullets":["First point"]}'
        if output_type.__name__ == "RetrievalOutput":
            return '<think>check both passages</think>{"passage_index":1,"reasoning":"It contains the answer."}'
        return '<think>make JSON</think>{"question":"What is self-attention?"}'

    def stream(self, prompt, **kwargs):
        yield from ()


def test_reasoning_defaults_to_hf_reasoning_model(monkeypatch):
    calls = []

    def fake_load_backend(model_path, mlx=False):
        calls.append((model_path, mlx))
        return DummyBackend()

    monkeypatch.setattr(model_mod, "load_backend", fake_load_backend)

    model = NeuralTxt(reasoning=True)

    assert model.reasoning is True
    assert calls == [("paperbd/neuraltxt-v1-135M-reasoning", False)]


def test_reasoning_defaults_to_mlx_reasoning_model(monkeypatch):
    calls = []

    def fake_load_backend(model_path, mlx=False):
        calls.append((model_path, mlx))
        return DummyBackend()

    monkeypatch.setattr(model_mod, "load_backend", fake_load_backend)

    NeuralTxt(backend="mlx", reasoning=True)

    assert calls == [("paperbd/neuraltxt-v1-135M-reasoning-mlx", True)]


def test_reasoning_uses_reasoning_system_prompt(monkeypatch):
    backend = DummyBackend()
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: backend)

    model = NeuralTxt(reasoning=True)
    model.generate_question("Transformers use self-attention.")

    system_prompt = backend.tokenizer.messages[0]["content"]
    assert "Generate your reasoning first inside <think> and </think> tags" in system_prompt
    assert "content after </think> must contain only that format" in system_prompt


def test_non_reasoning_uses_standard_system_prompt(monkeypatch):
    backend = DummyBackend()
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: backend)

    model = NeuralTxt()
    model.generate_question("Transformers use self-attention.")

    system_prompt = backend.tokenizer.messages[0]["content"]
    assert "Generate your reasoning first inside <think>" not in system_prompt


def test_strip_thinking_removes_leading_reasoning_block():
    assert strip_thinking("<think>step 1\nstep 2</think> The answer") == "The answer"


def test_split_thinking_returns_answer_and_reasoning():
    assert split_thinking("<think>step 1\nstep 2</think> The answer") == (
        "The answer",
        "step 1\nstep 2",
    )


def test_split_thinking_handles_missing_opening_tag():
    assert split_thinking("step 1\nstep 2</think> The answer") == (
        "The answer",
        "step 1\nstep 2",
    )


def test_return_reasoning_requires_reasoning_model(monkeypatch):
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: DummyBackend())

    with pytest.raises(ValueError, match="requires reasoning=True"):
        NeuralTxt(return_reasoning=True)


def test_reasoning_plain_text_output_is_stripped(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            ["<think>I should make one question.</think>What is self-attention?"]
        ),
    )

    model = NeuralTxt(reasoning=True)

    assert model.generate_question("Transformers use self-attention.") == "What is self-attention?"


def test_return_reasoning_wraps_plain_text_output(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            ["<think>I should make one question.</think>What is self-attention?"]
        ),
    )

    model = NeuralTxt(reasoning=True, return_reasoning=True)
    result = model.generate_question("Transformers use self-attention.")

    assert isinstance(result, ReasonedOutput)
    assert result.output == "What is self-attention?"
    assert result.reasoning == "I should make one question."
    assert result.raw == "<think>I should make one question.</think>What is self-attention?"


def test_return_reasoning_can_be_set_per_call(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            ["<think>I should make one question.</think>What is self-attention?"]
        ),
    )

    model = NeuralTxt(reasoning=True)
    result = model.generate_question(
        "Transformers use self-attention.",
        return_reasoning=True,
    )

    assert result.output == "What is self-attention?"
    assert result.reasoning == "I should make one question."


def test_reasoning_rollouts_are_stripped_before_parsing(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            [
                "<think>a</think>- First point",
                "<think>b</think>- Second point",
            ]
        ),
    )

    model = NeuralTxt(reasoning=True)

    assert model.extract_bullets("A passage.", rollouts=2) == [
        ["First point"],
        ["Second point"],
    ]


def test_return_reasoning_wraps_parsed_rollouts(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            [
                "<think>a</think>- First point",
                "<think>b</think>- Second point",
            ]
        ),
    )

    model = NeuralTxt(reasoning=True, return_reasoning=True)
    results = model.extract_bullets("A passage.", rollouts=2)

    assert [result.output for result in results] == [["First point"], ["Second point"]]
    assert [result.reasoning for result in results] == ["a", "b"]


def test_reasoning_retrieval_strips_thinking_before_parsing(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            ["<think>check both passages</think>**Passage:** 2\n**Why:** It contains the answer."]
        ),
    )

    model = NeuralTxt(reasoning=True)
    result = model.find_relevant("Which passage answers it?", ["No.", "Yes."])

    assert result.index == 1
    assert result.raw == "**Passage:** 2\n**Why:** It contains the answer."


def test_return_reasoning_wraps_retrieval_result(monkeypatch):
    monkeypatch.setattr(
        model_mod,
        "load_backend",
        lambda model_path, mlx=False: DummyBackend(
            ["<think>check both passages</think>**Passage:** 2\n**Why:** It contains the answer."]
        ),
    )

    model = NeuralTxt(reasoning=True, return_reasoning=True)
    result = model.find_relevant("Which passage answers it?", ["No.", "Yes."])

    assert result.output.index == 1
    assert result.output.raw == "**Passage:** 2\n**Why:** It contains the answer."
    assert result.reasoning == "check both passages"


def test_reasoning_json_mode_uses_reasoned_json(monkeypatch):
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: DummyBackend())

    model = NeuralTxt(reasoning=True)

    result = model.extract_bullets("A passage.", json=True)

    assert result.bullets == ["First point"]


def test_reasoning_json_mode_can_return_reasoning(monkeypatch):
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: DummyBackend())

    model = NeuralTxt(reasoning=True, return_reasoning=True)
    result = model.extract_bullets("A passage.", json=True)

    assert result.output.bullets == ["First point"]
    assert result.reasoning == "extract bullets"
    assert result.raw == '<think>extract bullets</think>{"bullets":["First point"]}'


def test_reasoning_json_mode_supports_list_schema(monkeypatch):
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: DummyBackend())

    model = NeuralTxt(reasoning=True, return_reasoning=True)
    result = model.generate_questions_list("A passage.", json=True)

    assert result.output.questions == ["What is self-attention?"]
    assert result.reasoning == "make a JSON list"


def test_reasoning_json_mode_supports_retrieval_schema(monkeypatch):
    monkeypatch.setattr(model_mod, "load_backend", lambda model_path, mlx=False: DummyBackend())

    model = NeuralTxt(reasoning=True, return_reasoning=True)
    result = model.find_relevant("Which passage answers it?", ["No.", "Yes."], json=True)

    assert result.output.passage_index == 1
    assert result.output.reasoning == "It contains the answer."
    assert result.reasoning == "check both passages"
