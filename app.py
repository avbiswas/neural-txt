"""
NeuralTxt — terminal-style Gradio demo with parallel streaming.
Usage: uv run instruction_tuning/app.py models/mlx/my-model [--mlx]
"""
import sys, os, argparse, re, json, ast
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    import gradio
except ImportError:
    print("gradio is not installed. Install it with:\n")
    print('  uv pip install -e ".[app]"')
    sys.exit(1)

from neuraltxt import NeuralTxt
from neuraltxt.types import (
    BulletsOutput, QAPairsOutput, QuestionOutput, FactOutput,
    AnswerOutput, RephraseOutput, ContinuationOutput,
    TripletsOutput, ComparisonOutput,
)
from neuraltxt.tasks import SYSTEM_PROMPT
import neuraltxt.tasks as t
import gradio as gr

MAX_NEW_TOKENS = 512

# ── CLI ───────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser()
parser.add_argument("model_path", help="Path to model directory")
parser.add_argument("--mlx", action="store_true", help="Use MLX backend")
parser.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
parser.add_argument("-n", "--num-generations", type=int, default=2, choices=[1, 2, 3, 4], help="Number of generations", dest="n")
args, _ = parser.parse_known_args()

researcher = NeuralTxt(args.model_path, backend="mlx" if args.mlx else "hf")

# ── Theme + CSS ───────────────────────────────────────────────────────────────

THEME = gr.themes.Base(
    font=[gr.themes.GoogleFont("JetBrains Mono"), "Fira Code", "monospace"],
).set(
    # backgrounds
    body_background_fill="#0d1117",
    block_background_fill="#0d1117",
    panel_background_fill="#0d1117",
    # inputs
    input_background_fill="#161b22",
    input_border_color="#30363d",
    input_border_color_focus="#58a6ff",
    # blocks
    block_border_color="#30363d",
    block_border_width="1px",
    block_label_text_color="#58a6ff",
    block_label_text_size="11px",
    # body text
    body_text_color="#c9d1d9",
    body_text_size="12px",
    # buttons
    button_primary_background_fill="#238636",
    button_primary_background_fill_hover="#2ea043",
    button_primary_text_color="#ffffff",
    button_secondary_background_fill="#21262d",
    button_secondary_border_color="#30363d",
    button_secondary_text_color="#c9d1d9",
)

CSS = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

footer { display: none !important; }

/* Green output text */
.output-box textarea { color: #ffffff !important; font-size: 24px !important; line-height: 1.6 !important; }

/* Dropdown list items */
ul[role="listbox"] { background: #161b22 !important; }
ul[role="listbox"] li { color: #e6edf3 !important; background: #161b22 !important; }
ul[role="listbox"] li:hover { background: #21262d !important; }

/* Amber messages preview */
.messages-box textarea { color: #e3b341 !important; font-size: 11px !important; }

/* Temperature headers */
.temp-header p {
    color: #f78166 !important;
    font-size: 11px !important;
    font-weight: 600 !important;
    letter-spacing: 2px !important;
    margin: 4px 0 2px 0 !important;
}

/* Left panel textareas — fixed heights, no layout shift
   Overhead: header(60) + mode block(70) + button(52) + labels(40) + gaps/padding(50) = ~272px */
#left-panel .input-box textarea {
    height: calc((100vh - 272px) * 0.28) !important;
    overflow-y: auto !important;
    resize: none !important;
}
#left-panel .messages-box textarea {
    height: calc((100vh - 272px) * 0.62) !important;
    overflow-y: auto !important;
    resize: none !important;
}

/* Full-width generate button */
.gen-btn button { width: 100% !important; letter-spacing: 1px !important; }

/* Stats bar */
.stats-bar p {
    color: #8b949e !important;
    font-size: 14px !important;
    letter-spacing: 1px !important;
    margin: 2px 0 4px 0 !important;
}

/* Tighten spacing */
.block { padding: 6px !important; }
.gap, .gap-2 { gap: 6px !important; }

/* Left divider */
.left-col { border-right: 1px solid #21262d !important; }

"""

# Dynamic textarea height — bypasses Gradio's wrapper div chain
_PANEL_H   = "100vh - 60px"
_ROW_COUNT = 2 if args.n > 2 else 1
_BOTTOM_PAD = 80  # aligns with generate button + padding
_BOX_H     = f"calc(({_PANEL_H}) / {_ROW_COUNT} - {_BOTTOM_PAD}px)"

CSS += f"""
#right-panel .output-box textarea {{
    height: {_BOX_H} !important;
    overflow-y: auto !important;
    resize: none !important;
}}
"""

# ── Modes ─────────────────────────────────────────────────────────────────────

def _fmt_bullets(items):  return "\n".join(f"- {b}" for b in items)
def _fmt_qa(pairs):       return "\n\n".join(f"Q: {p.question}\nA: {p.answer}" for p in pairs)
def _fmt_triplets(items): return "\n".join(f"({t.subject}, {t.relation}, {t.object})" for t in items)
MULTI_INPUT_MODES = {
    "answer":     {"label_1": "passage",   "label_2": "question"},
    "comparison": {"label_1": "passage 1", "label_2": "passage 2"},
}

MODES = {
    "bullets":      {"desc": "Extract key points as bullets",     "hint": "Paste passage..."},
    "qa_pairs":     {"desc": "Generate Q&A pairs",                "hint": "Paste passage..."},
    "question":     {"desc": "Generate a question from passage",  "hint": "Paste passage..."},
    "fact":         {"desc": "Extract a single fact",             "hint": "Paste passage..."},
    "answer":       {"desc": "Answer question given passage",     "hint": "Paste passage..."},
    "rephrase":     {"desc": "Rephrase and elaborate",            "hint": "Paste passage..."},
    "continuation": {"desc": "Continue passage from beginning",   "hint": "Paste start of passage..."},
    "triplets":     {"desc": "Extract knowledge graph triplets",  "hint": "Paste passage..."},
    "comparison":   {"desc": "Compare two passages",              "hint": "Paste passage 1..."},
}

MODE_KEYS    = list(MODES.keys())
MODE_CHOICES = [f"{k}  —  {MODES[k]['desc']}" for k in MODE_KEYS]

INSTRUCTION_MAP = {
    "bullets":      t.BULLETS_INSTRUCTION,
    "qa_pairs":     t.QA_PAIRS_INSTRUCTION,
    "question":     t.QUESTION_FROM_PASSAGE_INSTRUCTION,
    "fact":         t.FACT_FROM_PASSAGE_INSTRUCTION,
    "answer":       t.QA_ANSWER_INSTRUCTION,
    "rephrase":     t.REPHRASE_INSTRUCTION,
    "continuation": t.CONTINUATION_INSTRUCTION,
    "triplets":     t.TRIPLETS_INSTRUCTION,
    "comparison":   t.COMPARISON_INSTRUCTION,
}

INSTRUCTION_MAP_JSON = {
    "bullets":      t.BULLETS_INSTRUCTION_JSON,
    "qa_pairs":     t.QA_PAIRS_INSTRUCTION_JSON,
    "question":     t.QUESTION_FROM_PASSAGE_INSTRUCTION_JSON,
    "fact":         t.FACT_FROM_PASSAGE_INSTRUCTION_JSON,
    "answer":       t.QA_ANSWER_INSTRUCTION_JSON,
    "rephrase":     t.REPHRASE_INSTRUCTION_JSON,
    "continuation": t.CONTINUATION_INSTRUCTION_JSON,
    "triplets":     t.TRIPLETS_INSTRUCTION_JSON,
    "comparison":   t.COMPARISON_INSTRUCTION_JSON,
}

def key_from_choice(c): return c.split("  —  ")[0].strip()

def on_mode_change(choice):
    key = key_from_choice(choice)
    if key in MULTI_INPUT_MODES:
        mi = MULTI_INPUT_MODES[key]
        return (
            gr.update(placeholder=MODES[key]["hint"], label=mi["label_1"]),
            gr.update(visible=True, label=mi["label_2"], value=""),
        )
    return (
        gr.update(placeholder=MODES[key]["hint"], label="input"),
        gr.update(visible=False, value=""),
    )

def build_messages_preview(choice, text, text2, fmt="text"):
    if not text.strip(): return ""
    key  = key_from_choice(choice)
    imap = INSTRUCTION_MAP_JSON if fmt == "json" else INSTRUCTION_MAP
    inst = imap.get(key, "")
    if key == "answer":
        user_content = f"{inst}\n\nPassage: {text}\n\nQuestion: {text2}\nWhat is the answer?"
    elif key == "comparison":
        user_content = f"{inst}\n\nPassage 1:\n{text}\n\nPassage 2:\n{text2}"
    else:
        user_content = f"{inst}\n\n{text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    return json.dumps(messages, indent=2)

# ── JSON pretty-print ─────────────────────────────────────────────────────────

def _maybe_prettify(text: str) -> str:
    stripped = text.strip()

    # Python list literal → join items with \n\n
    if stripped.startswith("[") and stripped.endswith("]"):
        try:
            val = ast.literal_eval(stripped)
            if isinstance(val, list):
                return "\n\n".join(str(item) for item in val)
        except (ValueError, SyntaxError):
            pass

    # JSON object/array
    try:
        return json.dumps(json.loads(stripped), indent=2)
    except (json.JSONDecodeError, ValueError):
        pass

    # Fenced ```json block
    m = re.search(r"```json\s*([\s\S]+?)```", stripped)
    if m:
        try:
            pretty = json.dumps(json.loads(m.group(1)), indent=2)
            return stripped[:m.start()] + f"```json\n{pretty}\n```" + stripped[m.end():]
        except (json.JSONDecodeError, ValueError):
            pass

    return text

# ── Parallel streaming ────────────────────────────────────────────────────────

def build_prompt(mode_key, user_text, user_text2, fmt="text"):
    imap = INSTRUCTION_MAP_JSON if fmt == "json" else INSTRUCTION_MAP
    inst = imap[mode_key]
    if mode_key == "answer":
        user_content = f"Passage: {user_text}\n\nQuestion: {user_text2}\nWhat is the answer?"
    elif mode_key == "comparison":
        user_content = f"Passage 1:\n{user_text}\n\nPassage 2:\n{user_text2}"
    else:
        user_content = user_text
    tok  = researcher._backend.tokenizer
    msgs = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": f"{inst}\n\n{user_content}"},
    ]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def _fmt_stats(stats: dict) -> str:
    if not stats:
        return ""
    return (
        f"// {stats['tokens']} tok · "
        f"{stats['tps']:.1f} tok/s · "
        f"{stats['peak_memory']:.2f} GB peak"
    )


JSON_OUTPUT_TYPES = {
    "bullets":      BulletsOutput,
    "qa_pairs":     QAPairsOutput,
    "question":     QuestionOutput,
    "fact":         FactOutput,
    "answer":       AnswerOutput,
    "rephrase":     RephraseOutput,
    "continuation": ContinuationOutput,
    "triplets":     TripletsOutput,
    "comparison":   ComparisonOutput,
}


def generate_stream(mode_choice, user_text, user_text2, fmt="text"):
    n = args.n
    if not user_text.strip():
        yield [""] + ["// no input"] * n
        return

    key    = key_from_choice(mode_choice)
    prompt = build_prompt(key, user_text, user_text2, fmt)
    temp   = args.temperature

    results = [""] * n
    stats_text = ""

    if fmt == "json":
        output_type = JSON_OUTPUT_TYPES[key]
        for idx in range(n):
            try:
                raw = researcher._backend.generate_json(
                    prompt, output_type, max_new_tokens=MAX_NEW_TOKENS
                )
                results[idx] = _maybe_prettify(raw)
                yield [stats_text] + results
            except Exception as e:
                results[idx] = f"// error: {e}"
                yield [stats_text] + results
        return

    for idx in range(n):
        try:
            for chunk in researcher._backend.stream(
                prompt, temperature=temp, max_new_tokens=MAX_NEW_TOKENS
            ):
                results[idx] += chunk
                yield [stats_text] + [_maybe_prettify(r) for r in results]
            stats_text = _fmt_stats(getattr(researcher._backend, "_last_stats", {}))
            yield [stats_text] + [_maybe_prettify(r) for r in results]
        except Exception as e:
            results[idx] = f"// error: {e}"
            yield [stats_text] + [_maybe_prettify(r) for r in results]

# ── UI ────────────────────────────────────────────────────────────────────────

with gr.Blocks(title="neural-txt") as demo:

    gr.Markdown(
        "<span style='color:#58a6ff;font-size:13px;font-weight:600'>// neural-txt</span>"
    )

    with gr.Row(equal_height=False):

        # ── Left ──────────────────────────────────────────────────────────────
        with gr.Column(scale=1, elem_classes=["left-col"], elem_id="left-panel"):

            with gr.Row():
                mode_dd = gr.Dropdown(
                    choices=MODE_CHOICES, value=MODE_CHOICES[0], label="mode", scale=4,
                )
                fmt_dd = gr.Dropdown(
                    choices=["text", "json"], value="text", label="format", scale=1,
                )

            user_box = gr.Textbox(
                label="input", lines=7,
                placeholder=MODES[MODE_KEYS[0]]["hint"],
                elem_classes=["input-box"],
            )

            user_box2 = gr.Textbox(
                label="input 2", lines=4,
                visible=False,
                elem_classes=["input-box"],
            )

            messages_box = gr.Textbox(
                label="messages", lines=11,
                interactive=False,
                elem_classes=["messages-box"],
            )

            gen_btn = gr.Button("▶  GENERATE", variant="primary", elem_classes=["gen-btn"])

        # ── Right: grid from args.n ───────────────────────────────────────────
        with gr.Column(scale=1, elem_id="right-panel"):
            stats_md = gr.Markdown("", elem_classes=["stats-bar"])
            n = args.n
            cols_per_row = 2 if n > 1 else 1
            output_cols  = []
            for row_start in range(0, n, cols_per_row):
                with gr.Row():
                    for i in range(row_start, min(row_start + cols_per_row, n)):
                        output_cols.append(
                            gr.Textbox(
                                show_label=False, lines=14,
                                interactive=False,
                                elem_classes=["output-box"],
                            )
                        )

    # ── Wiring ────────────────────────────────────────────────────────────────

    mode_dd.change(fn=on_mode_change,           inputs=mode_dd,                              outputs=[user_box, user_box2])
    mode_dd.change(fn=build_messages_preview,  inputs=[mode_dd, user_box, user_box2, fmt_dd], outputs=messages_box)
    user_box.change(fn=build_messages_preview, inputs=[mode_dd, user_box, user_box2, fmt_dd], outputs=messages_box)
    user_box2.change(fn=build_messages_preview,inputs=[mode_dd, user_box, user_box2, fmt_dd], outputs=messages_box)
    fmt_dd.change(fn=build_messages_preview,   inputs=[mode_dd, user_box, user_box2, fmt_dd], outputs=messages_box)

    gen_btn.click(
        fn=generate_stream,
        inputs=[mode_dd, user_box, user_box2, fmt_dd],
        outputs=[stats_md] + output_cols,
        show_progress=False,
    )

demo.launch(css=CSS, theme=THEME)
