"""
Regression tests for the outlines re-templating bug.

Background: outlines' MLXLM/Transformers adapters re-apply the chat template to
any str prompt when the tokenizer has one — wrapping our already-templated
prompt as a fresh user turn. That closes the assistant turn and reopens a new
one, breaking the trained `<think>…</think>{answer}` span so constrained JSON
lists collapse to a single item. `_no_retemplate` disables that, and every
outlines model in the backend must go through it.
"""
import inspect
import re

import neuraltxt.backends as backends
from neuraltxt.backends import _no_retemplate


class _FakeAdapter:
    def __init__(self, has_chat_template=True):
        self.has_chat_template = has_chat_template


class _FakeOutlinesModel:
    def __init__(self):
        self.type_adapter = _FakeAdapter(has_chat_template=True)


def test_no_retemplate_disables_chat_template():
    model = _FakeOutlinesModel()
    returned = _no_retemplate(model)
    assert returned is model
    assert model.type_adapter.has_chat_template is False


def test_no_retemplate_is_safe_without_adapter():
    class Bare:
        pass

    bare = Bare()
    # Must not raise when there is no type_adapter / flag.
    assert _no_retemplate(bare) is bare


def test_every_outlines_model_disables_retemplate():
    """Any `outlines.from_*` call must be wrapped in `_no_retemplate(...)`.

    Guards against re-introducing the chat-template re-wrapping bug when new
    outlines call sites are added.
    """
    source = inspect.getsource(backends)
    for line in source.splitlines():
        if re.search(r"outlines\.from_(mlxlm|transformers)\(", line):
            assert "_no_retemplate(" in line, (
                f"outlines model not wrapped in _no_retemplate: {line.strip()!r}"
            )
