"""
neural-txt — a Textual terminal UI mirroring the Gradio demo.

Exploratory only; not shipped in the published package (the wheel packages
`neuraltxt/` and the sdist lists only `neuraltxt/`, README, pyproject).

Usage:
    uv run tui.py [--mlx | --hf] [--reasoning] [--temperature 0.4] [-n 1]

Layout:
    LEFT    inputs — color-coded mode bar (←/→ to cycle) + passage box,
            with a second input row appearing for answer/comparison modes.
    CENTER  controls stacked vertically (generate, format, mode, clear).
    RIGHT   streamed generation; reasoning <think>…</think> is dimmed/italic.

Keys:  ←/→ cycle modes · Tab move focus · Enter (outside the editor) or
       Ctrl+R run · Ctrl+F toggle text/json · Ctrl+C quit.
"""
from __future__ import annotations

import argparse
import resource
import time

from rich.text import Text

try:
    import textual  # noqa: F401
except ImportError:
    print("textual is not installed. Install it with:\n")
    print('  pip install "neural-txt[tui]"')
    print("  # or, from a checkout:  uv pip install -e \".[tui]\"")
    raise SystemExit(1)

from textual import work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.reactive import reactive
from textual.widget import Widget
from textual.widgets import Button, Footer, Label, Static, TextArea

import neuraltxt.tasks as t
from neuraltxt import NeuralTxt
from neuraltxt.parsing import split_thinking
from neuraltxt.tasks import REASONING_SYSTEM_PROMPT, SYSTEM_PROMPT

MAX_NEW_TOKENS = 512

# ── Modes ──────────────────────────────────────────────────────────────────────
# Each mode is color-coded; the color flows from the mode bar chip through to the
# generated output accent so a viewer can tell at a glance which mode produced what.

MODES: dict[str, dict] = {
    "bullets":        {"desc": "Extract key points as bullets",    "color": "#58a6ff"},
    "qa_pairs":       {"desc": "Generate Q&A pairs",               "color": "#bc8cff"},
    "question":       {"desc": "Generate a question from passage", "color": "#3fb950"},
    "questions_list": {"desc": "Generate a list of questions",     "color": "#56d364"},
    "fact":           {"desc": "Extract a single fact",            "color": "#f0883e"},
    "answer":         {"desc": "Answer question given passage",    "color": "#f85149"},
    "rephrase":       {"desc": "Rephrase and elaborate",           "color": "#d29922"},
    "continuation":   {"desc": "Continue passage from beginning",  "color": "#39c5cf"},
    "triplets":       {"desc": "Extract knowledge-graph triplets", "color": "#db61a2"},
    "comparison":     {"desc": "Compare two passages",             "color": "#a371f7"},
}
MODE_KEYS = list(MODES)

# Free-text tasks with no structured form in training — JSON mode unsupported.
JSON_UNSUPPORTED_MODES = {"rephrase", "continuation", "comparison"}

# Modes that take a second input row on the left.
MULTI_INPUT_MODES: dict[str, dict] = {
    "answer": {
        "label_1": "passage",
        "label_2": "question",
        "hint_1": "Paste passage…",
        "hint_2": "Ask a question about the passage…",
    },
    "comparison": {
        "label_1": "passage 1",
        "label_2": "passage 2",
        "hint_1": "Paste passage 1…",
        "hint_2": "Paste passage 2…",
    },
}

INSTRUCTION_MAP = {
    "bullets":        t.BULLETS_INSTRUCTION,
    "qa_pairs":       t.QA_PAIRS_INSTRUCTION,
    "question":       t.QUESTION_FROM_PASSAGE_INSTRUCTION,
    "questions_list": t.QUESTIONS_LIST_INSTRUCTION,
    "fact":           t.FACT_FROM_PASSAGE_INSTRUCTION,
    "answer":         t.QA_ANSWER_INSTRUCTION,
    "rephrase":       t.REPHRASE_INSTRUCTION,
    "continuation":   t.CONTINUATION_INSTRUCTION,
    "triplets":       t.TRIPLETS_INSTRUCTION,
    "comparison":     t.COMPARISON_INSTRUCTION,
}

INSTRUCTION_MAP_JSON = {
    "bullets":        t.BULLETS_INSTRUCTION_JSON,
    "qa_pairs":       t.QA_PAIRS_INSTRUCTION_JSON,
    "question":       t.QUESTION_FROM_PASSAGE_INSTRUCTION_JSON,
    "questions_list": t.QUESTIONS_LIST_INSTRUCTION_JSON,
    "fact":           t.FACT_FROM_PASSAGE_INSTRUCTION_JSON,
    "answer":         t.QA_ANSWER_INSTRUCTION_JSON,
    "triplets":       t.TRIPLETS_INSTRUCTION_JSON,
}

# Per-mode outlines output schema for streamed JSON (matches model.py's json
# paths). bullets/questions_list use a bare list[str] so the instruction's
# "list of strings" phrasing agrees with the schema; the rest use their object
# schema. rephrase/continuation/comparison have no JSON form (excluded).
def _json_stream_types() -> dict:
    from neuraltxt.types import (
        ShortText, QuestionOutput, FactOutput, AnswerOutput,
        QAPairsOutput, TripletsOutput,
    )
    return {
        "bullets":        list[ShortText],
        "questions_list": list[ShortText],
        "qa_pairs":       QAPairsOutput,
        "triplets":       TripletsOutput,
        "question":       QuestionOutput,
        "fact":           FactOutput,
        "answer":         AnswerOutput,
    }

JSON_STREAM_TYPES = _json_stream_types()

# Map a mode to a NeuralTxt public method for the non-streaming paths
# (reasoning json / multi-rollout), matching app.py.
METHOD_MAP = {
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


# ── Stats helpers ────────────────────────────────────────────────────────────

def _process_rss_gb() -> float:
    """Resident set size of this process. ru_maxrss is bytes on macOS, KiB on Linux."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    import sys
    if sys.platform == "darwin":
        return rss / 1e9
    return rss * 1024 / 1e9


def _format_stats(stats: dict, elapsed: float) -> Text:
    out = Text(no_wrap=True)
    out.append("│ ", style="#30363d")
    if stats.get("tokens") is not None:
        out.append(f"{stats['tokens']} tok", style="#8b949e")
        out.append("  ·  ", style="#30363d")
        out.append(f"{stats.get('tps', 0.0):.1f} tok/s", style="#58a6ff")
        out.append("  ·  ", style="#30363d")
    peak = stats.get("peak_memory")
    if peak:
        out.append(f"{peak:.2f} GB peak", style="#f0883e")
        out.append("  ·  ", style="#30363d")
    out.append(f"{_process_rss_gb():.2f} GB rss", style="#d29922")
    out.append("  ·  ", style="#30363d")
    out.append(f"{elapsed:.2f}s", style="#3fb950")
    return out


# ── Mode bar (left, color-coded, ←/→ to cycle) ─────────────────────────────────

class ModeCell(Static):
    """A single mode button. Color/background set by its ModeGrid parent."""


class ModeGrid(Widget):
    """Focusable grid of bordered, color-coded mode buttons. Arrows move in 2D."""

    COLS = 3

    can_focus = True
    index: reactive[int] = reactive(0)

    class Changed(Message):
        def __init__(self, key: str) -> None:
            self.key = key
            super().__init__()

    def compose(self) -> ComposeResult:
        for i, k in enumerate(MODE_KEYS):
            yield ModeCell(k, id=f"mode-{i}", classes="mode-cell")

    def on_mount(self) -> None:
        self._restyle()

    def on_click(self, event) -> None:
        # Clicking a button selects it.
        for i in range(len(MODE_KEYS)):
            if event.widget is self.query_one(f"#mode-{i}", ModeCell):
                self.index = i
                self.focus()
                break

    def action_move(self, delta: int) -> None:
        self.index = (self.index + delta) % len(MODE_KEYS)

    def action_move_row(self, delta: int) -> None:
        target = self.index + delta * self.COLS
        if 0 <= target < len(MODE_KEYS):
            self.index = target

    @property
    def key(self) -> str:
        return MODE_KEYS[self.index]

    def watch_index(self, _old: int, _new: int) -> None:
        self.post_message(self.Changed(self.key))
        self._restyle()

    def on_focus(self) -> None:
        self._restyle()

    def on_blur(self) -> None:
        self._restyle()

    def _restyle(self) -> None:
        if not self.is_mounted:
            return
        for i, k in enumerate(MODE_KEYS):
            cell = self.query_one(f"#mode-{i}", ModeCell)
            color = MODES[k]["color"]
            if i == self.index:
                cell.add_class("-selected")
                cell.styles.background = color
                cell.styles.color = "black"
            else:
                cell.remove_class("-selected")
                cell.styles.background = "#21262d"
                cell.styles.color = color


# ── Output card (right, streamed, reasoning colored) ───────────────────────────

class OutputCard(VerticalScroll):
    """Scrollable card that renders streamed text with reasoning highlighting."""

    def __init__(self, accent: str = "#3fb950", **kwargs) -> None:
        super().__init__(**kwargs)
        self.accent = accent
        self._body = Static("", id="card-body")

    def compose(self) -> ComposeResult:
        yield self._body

    def set_accent(self, color: str) -> None:
        self.accent = color

    def render_text(self, raw: str, reasoning: bool) -> Text:
        out = Text()
        if not raw:
            out.append("…", style="#484f58")
            return out

        if reasoning:
            s = raw
            if "<think>" in s:
                pre, rest = s.split("<think>", 1)
                if pre.strip():
                    out.append(pre.strip() + "\n", style="bold #ffffff")
                if "</think>" in rest:
                    think, ans = rest.split("</think>", 1)
                    self._append_reasoning(out, think)
                    if ans.strip():
                        out.append(ans.strip(), style="bold #ffffff")
                else:
                    self._append_reasoning(out, rest)
            else:
                out.append(s, style="bold #ffffff")
        else:
            out.append(raw, style="bold #ffffff")
        return out

    @staticmethod
    def _append_reasoning(out: Text, text: str) -> None:
        out.append("\n💭 reasoning\n", style="italic #6e7681")
        out.append(text.strip() + "\n\n", style="italic #8b949e")

    def update_text(self, raw: str, reasoning: bool) -> None:
        self._body.update(self.render_text(raw, reasoning))
        self.scroll_end(animate=False)


# ── App ────────────────────────────────────────────────────────────────────────

class NeuralTxtTUI(App):
    CSS = """
    Screen { background: #0d1117; }

    #title {
        height: 1;
        padding: 0 1;
        background: #0d1117;
    }

    #main { height: 1fr; }

    /* ── left: inputs ── */
    #left {
        width: 2fr;
        border-right: solid #21262d;
        padding: 0 1;
    }
    ModeGrid {
        layout: grid;
        grid-size: 3;
        grid-gutter: 1 1;
        grid-rows: 1;
        height: auto;
        padding: 1 1 0 1;
    }
    .mode-cell {
        height: 1;
        background: #21262d;
        text-align: center;
        content-align: center middle;
        text-style: bold;
    }
    .hint { color: #6e7681; height: 1; padding: 0 1; }
    .input-label {
        color: #8b949e;
        text-style: bold;
        height: 1;
        padding: 0 1;
        margin-top: 1;
    }
    #passage { height: 1fr; }
    #secondary-row { height: 0; display: none; }
    #secondary-row.visible { height: 9; display: block; }
    TextArea {
        background: #0d1117;
        border: round #30363d;
        scrollbar-background: #0d1117;
        scrollbar-color: #30363d;
    }
    TextArea:focus { border: round #58a6ff; background: #0d1117; }

    #gen-btn {
        width: 100%;
        height: 3;
        margin-top: 1;
        background: #238636;
        color: #ffffff;
        text-style: bold;
        border: none;
    }
    #gen-btn:focus { background: #2ea043; text-style: bold; }
    #gen-btn:hover { background: #2ea043; }

    /* ── right: generation ── */
    #right { width: 3fr; padding: 0 1; }
    #stats {
        height: 1;
        color: #8b949e;
        padding: 0 1;
        margin-bottom: 1;
    }
    #outputs { height: 1fr; }
    OutputCard {
        border: round #30363d;
        background: #0d1117;
        padding: 0 1;
        margin-bottom: 1;
    }
    OutputCard:focus-within { border: round #30363d; }
    #card-body { padding: 1 2; }
    """

    BINDINGS = [
        Binding("ctrl+r", "generate", "run"),
        Binding("enter", "generate", "run", show=False),
        Binding("f", "toggle_format", "text/json"),
        Binding("ctrl+l", "clear", "clear"),
        Binding("escape", "blur", "unfocus", show=True, priority=True),
        Binding("up", "nav('up')", "up", show=False),
        Binding("down", "nav('down')", "down", show=False),
        Binding("left", "nav('left')", "left", show=False),
        Binding("right", "nav('right')", "right", show=False),
        Binding("ctrl+c", "quit", "quit"),
    ]

    fmt: reactive[str] = reactive("text")

    def __init__(
        self,
        *,
        researcher: NeuralTxt,
        backend: str,
        reasoning: bool,
        temperature: float,
        num_beams: int,
    ):
        super().__init__()
        self.backend_name = backend
        self.reasoning = reasoning
        self.temperature = temperature
        self.num_beams = max(1, num_beams)
        # Loaded up front in the parent process: doing it inside a Textual
        # worker triggers "bad value(s) in fds_to_keep" because Textual
        # redirects stdout/stderr and subprocesses inherit invalid fds.
        self.researcher: NeuralTxt = researcher
        self._busy = False

    # ── layout ──────────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Static(self._title_text(), id="title")
        with Horizontal(id="main"):
            # left — inputs
            with Vertical(id="left"):
                yield ModeGrid(id="modegrid")
                yield Static("↑↓←→ select task", classes="hint", id="hint")
                yield Label("input", id="label-1", classes="input-label")
                yield TextArea(id="passage")
                with Vertical(id="secondary-row"):
                    yield Label("input 2", id="label-2", classes="input-label")
                    yield TextArea(id="secondary")
                yield Button("▶  GENERATE", id="gen-btn")
            # right — generation
            with Vertical(id="right"):
                yield Static("│ idle", id="stats")
                yield VerticalScroll(id="outputs")
        yield Footer()

    def on_mount(self) -> None:
        self._rebuild_cards()
        self.query_one("#modegrid", ModeGrid).focus()
        self._set_status("ready", "#3fb950")

    def _title_text(self) -> Text:
        out = Text()
        out.append("// neural-txt", style="bold #58a6ff")
        out.append("   ", style="")
        out.append(self.backend_name, style="#8b949e")
        if self.reasoning:
            out.append(" ·reasoning", style="#bc8cff")
        out.append("   │   ", style="#30363d")
        out.append("format ", style="#6e7681")
        out.append(self.fmt, style="bold #f0883e")
        return out

    def _refresh_title(self) -> None:
        self.query_one("#title", Static).update(self._title_text())

    # ── cards ─────────────────────────────────────────────────────────────────

    def _rebuild_cards(self) -> None:
        outputs = self.query_one("#outputs", VerticalScroll)
        outputs.remove_children()
        accent = MODES[self._mode_key()]["color"]
        for i in range(self.num_beams):
            card = OutputCard(accent=accent, id=f"card-{i}")
            outputs.mount(card)

    def _cards(self) -> list[OutputCard]:
        return list(self.query(OutputCard))

    # ── mode handling ───────────────────────────────────────────────────────

    def _mode_key(self) -> str:
        return self.query_one("#modegrid", ModeGrid).key

    def on_mode_grid_changed(self, message: ModeGrid.Changed) -> None:
        self._apply_mode(message.key)

    def action_blur(self) -> None:
        # Escape: drop focus from a text editor back onto the mode grid.
        if isinstance(self.focused, TextArea):
            self.query_one("#modegrid", ModeGrid).focus()

    def action_nav(self, direction: str) -> None:
        grid = self.query_one("#modegrid", ModeGrid)
        if direction == "left":
            grid.action_move(-1)
        elif direction == "right":
            grid.action_move(1)
        elif direction == "up":
            grid.action_move_row(-1)
        elif direction == "down":
            grid.action_move_row(1)

    def _apply_mode(self, key: str) -> None:
        accent = MODES[key]["color"]
        for card in self._cards():
            card.set_accent(accent)
        label1 = self.query_one("#label-1", Label)
        passage = self.query_one("#passage", TextArea)
        row2 = self.query_one("#secondary-row", Vertical)

        if key in MULTI_INPUT_MODES:
            mi = MULTI_INPUT_MODES[key]
            label1.update(mi["label_1"])
            self.query_one("#label-2", Label).update(mi["label_2"])
            row2.add_class("visible")
        else:
            label1.update("input")
            row2.remove_class("visible")

    # ── format ────────────────────────────────────────────────────────────────

    def action_toggle_format(self) -> None:
        self.fmt = "json" if self.fmt == "text" else "text"
        self._refresh_title()

    # ── buttons ─────────────────────────────────────────────────────────────

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "gen-btn":
            self.action_generate()

    def action_clear(self) -> None:
        self.query_one("#passage", TextArea).text = ""
        self.query_one("#secondary", TextArea).text = ""
        for card in self._cards():
            card.update_text("", self.reasoning)
        self._set_status("idle", "#8b949e")

    # ── status ────────────────────────────────────────────────────────────────

    def _set_status(self, text: str, color: str = "#8b949e") -> None:
        bar = self.query_one("#stats", Static)
        out = Text()
        out.append("│ ", style="#30363d")
        out.append(text, style=color)
        bar.update(out)

    # ── generation ──────────────────────────────────────────────────────────

    def action_generate(self) -> None:
        if self._busy:
            return

        key = self._mode_key()
        text = self.query_one("#passage", TextArea).text.strip()
        text2 = self.query_one("#secondary", TextArea).text.strip()

        if not text:
            self._set_status("no input", "#f85149")
            return
        if key in MULTI_INPUT_MODES and not text2:
            self._set_status(f"missing {MULTI_INPUT_MODES[key]['label_2']}", "#f85149")
            return
        if self.fmt == "json" and key in JSON_UNSUPPORTED_MODES:
            self._cards()[0].update_text(
                f"// json mode not supported for {key} — switch to text (f)", self.reasoning
            )
            self._set_status(f"json not supported for {key}", "#f85149")
            return

        for card in self._cards():
            card.update_text("", self.reasoning)
        self._busy = True
        self._set_status("generating…", "#58a6ff")
        self._generate_worker(key, text, text2, self.fmt)

    def _build_prompt(self, key: str, text: str, text2: str, fmt: str = "text") -> str:
        inst = INSTRUCTION_MAP_JSON[key] if fmt == "json" else INSTRUCTION_MAP[key]
        if key == "answer":
            user = f"Passage: {text}\n\nQuestion: {text2}\nWhat is the answer?"
        elif key == "comparison":
            user = f"Passage 1:\n{text}\n\nPassage 2:\n{text2}"
        else:
            user = text
        tok = self.researcher._backend.tokenizer
        system_prompt = REASONING_SYSTEM_PROMPT if self.reasoning else SYSTEM_PROMPT
        msgs = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"{inst}\n\n{user}"},
        ]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @work(thread=True, exclusive=True)
    def _generate_worker(self, key: str, text: str, text2: str, fmt: str) -> None:
        backend = self.researcher._backend
        t0 = time.perf_counter()
        try:
            # Streaming path (the live demo): single beam, plain text.
            if self.num_beams == 1 and fmt == "text":
                prompt = self._build_prompt(key, text, text2)
                stream_prompt = f"{prompt}<think>" if self.reasoning else prompt
                acc = "<think>" if self.reasoning else ""
                card = self._cards()[0]
                for chunk in backend.stream(
                    stream_prompt,
                    temperature=self.temperature,
                    max_new_tokens=MAX_NEW_TOKENS,
                ):
                    acc += chunk
                    self.call_from_thread(card.update_text, acc, self.reasoning)
                stats = getattr(backend, "_last_stats", {}) or {}
                self.call_from_thread(card.update_text, acc, self.reasoning)
            elif self.num_beams == 1 and fmt == "json" and key in JSON_STREAM_TYPES:
                # Stream schema-constrained JSON token-by-token via outlines.
                # For reasoning models, stream the <think>…</think> trace first.
                prompt = self._build_prompt(key, text, text2, fmt="json")
                card = self._cards()[0]
                output_type = JSON_STREAM_TYPES[key]
                acc = ""
                if self.reasoning:
                    stream = backend.stream_reasoned_json(
                        prompt, output_type, temperature=self.temperature,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                else:
                    stream = backend.stream_json(
                        prompt, output_type, max_new_tokens=MAX_NEW_TOKENS,
                    )
                for chunk in stream:
                    acc += chunk
                    self.call_from_thread(card.update_text, acc, self.reasoning)
                stats = getattr(backend, "_last_stats", {}) or {}
                self.call_from_thread(card.update_text, self._prettify_reasoned(acc), self.reasoning)
            else:
                # reasoning json / multi-beam: public API (non-streaming), then render.
                results = self._run_public(key, text, text2, fmt)
                for card, res in zip(self._cards(), results):
                    self.call_from_thread(card.update_text, res, self.reasoning)
                stats = getattr(backend, "_last_stats", {}) or {}
        except Exception as e:  # noqa: BLE001
            self.call_from_thread(self._set_status, f"error: {e}", "#f85149")
            self.call_from_thread(self._mark_done)
            return

        elapsed = time.perf_counter() - t0
        self.call_from_thread(self._finish, stats, elapsed)

    @staticmethod
    def _prettify_json(raw: str) -> str:
        import json as _json
        try:
            return _json.dumps(_json.loads(raw), indent=2)
        except (ValueError, TypeError):
            return raw

    def _prettify_reasoned(self, raw: str) -> str:
        # Keep the <think>…</think> trace; pretty-print only the trailing JSON.
        if "</think>" in raw:
            head, tail = raw.split("</think>", 1)
            return f"{head}</think>\n{self._prettify_json(tail.strip())}"
        return self._prettify_json(raw)

    def _run_public(self, key: str, text: str, text2: str, fmt: str) -> list[str]:
        method = getattr(self.researcher, METHOD_MAP[key])
        kwargs = {
            "json": fmt == "json",
            "rollouts": self.num_beams,
            "temperature": self.temperature,
        }
        if self.reasoning:
            kwargs["return_reasoning"] = True
        if key == "answer":
            value = method(text2, text, **kwargs)
        elif key == "comparison":
            value = method(text, text2, **kwargs)
        else:
            value = method(text, **kwargs)

        items = value if self.num_beams > 1 else [value]
        return [self._render_public(item, fmt) for item in items[: self.num_beams]]

    def _render_public(self, item, fmt: str) -> str:
        from neuraltxt.types import ReasonedOutput

        if isinstance(item, ReasonedOutput):
            body = self._stringify(item.output, fmt)
            return f"<think>{item.reasoning}</think>{body}" if self.reasoning else body
        return self._stringify(item, fmt)

    @staticmethod
    def _stringify(value, fmt: str) -> str:
        import json as _json

        if fmt == "json" and hasattr(value, "model_dump_json"):
            try:
                return _json.dumps(_json.loads(value.model_dump_json()), indent=2)
            except Exception:  # noqa: BLE001
                return value.model_dump_json()

        # text rendering for the common shapes
        if hasattr(value, "bullets"):
            return "\n".join(f"- {b}" for b in value.bullets)
        if hasattr(value, "pairs"):
            return "\n\n".join(f"Q: {p.question}\nA: {p.answer}" for p in value.pairs)
        if hasattr(value, "questions"):
            return "\n".join(value.questions)
        if hasattr(value, "triplets"):
            return "\n".join(f"({t.subject}, {t.relation}, {t.object})" for t in value.triplets)
        for attr in ("question", "fact", "answer", "text", "comparison"):
            if hasattr(value, attr):
                return getattr(value, attr)
        if isinstance(value, list):
            if value and all(hasattr(i, "question") and hasattr(i, "answer") for i in value):
                return "\n\n".join(f"Q: {i.question}\nA: {i.answer}" for i in value)
            if value and all(hasattr(i, "subject") for i in value):
                return "\n".join(f"({i.subject}, {i.relation}, {i.object})" for i in value)
            return "\n".join(f"- {i}" for i in value)
        return str(value)

    def _finish(self, stats: dict, elapsed: float) -> None:
        self.query_one("#stats", Static).update(_format_stats(stats, elapsed))
        self._busy = False

    def _mark_done(self) -> None:
        self._busy = False


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="neural-txt Textual TUI")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--mlx", action="store_true", help="Use the MLX backend (default on Apple Silicon)")
    group.add_argument("--hf", action="store_true", help="Use the HuggingFace backend")
    parser.add_argument("--reasoning", action="store_true", help="Use the reasoning model")
    parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    parser.add_argument(
        "-n", "--num-beams", "--num-generations",
        type=int, default=1, choices=[1, 2, 3, 4],
        dest="num_beams", help="Number of candidates to generate",
    )
    args = parser.parse_args()

    backend = "hf" if args.hf else "mlx"

    # Load the model in the parent process, before Textual takes over the
    # terminal — loading inside a Textual worker fails with
    # "bad value(s) in fds_to_keep".
    badge = " (reasoning)" if args.reasoning else ""
    print(f"Loading neural-txt model [{backend}{badge}]… ", end="", flush=True)
    try:
        researcher = NeuralTxt(backend=backend, reasoning=args.reasoning)
    except Exception as e:  # noqa: BLE001
        print(f"failed.\n{e}")
        raise SystemExit(1) from e
    print("ready.")

    app = NeuralTxtTUI(
        researcher=researcher,
        backend=backend,
        reasoning=args.reasoning,
        temperature=args.temperature,
        num_beams=args.num_beams,
    )
    app.run()


if __name__ == "__main__":
    main()
