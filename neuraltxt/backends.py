"""
Backend abstraction — hidden from the user.
Handles model loading and raw text generation for MLX and HuggingFace.
"""
from __future__ import annotations
from abc import ABC, abstractmethod


DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_NUM_BEAMS = 1


def _get_num_return_sequences(kwargs: dict, num_beams: int) -> int:
    num_return_sequences = kwargs.pop("num_return_sequences", None)
    if num_return_sequences is None:
        return num_beams
    try:
        num_return_sequences = int(num_return_sequences)
    except (TypeError, ValueError):
        raise ValueError(
            f"num_return_sequences must be an integer, got {num_return_sequences!r}"
        ) from None
    if num_return_sequences < 1:
        raise ValueError(
            f"num_return_sequences must be >= 1, got {num_return_sequences}"
        )
    return num_return_sequences


def _get_num_beams(kwargs: dict) -> int:
    num_beams = kwargs.pop("num_beams", DEFAULT_NUM_BEAMS)
    try:
        num_beams = int(num_beams)
    except (TypeError, ValueError):
        raise ValueError(f"num_beams must be an integer, got {num_beams!r}") from None
    if num_beams < 1:
        raise ValueError(f"num_beams must be >= 1, got {num_beams}")
    return num_beams


class Backend(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def generate_many(self, prompt: str, **kwargs) -> list[str]: ...

    @abstractmethod
    def generate_json(self, prompt: str, output_type, **kwargs) -> str: ...

    @abstractmethod
    def generate_reasoned_json(self, prompt: str, output_type, **kwargs) -> str: ...

    @abstractmethod
    def stream(self, prompt: str, **kwargs): ...  # yields str chunks


def _no_retemplate(outlines_model):
    """Stop outlines from re-applying the chat template to our prompt.

    We build fully chat-templated prompts ourselves. By default the outlines
    MLXLM/Transformers adapter wraps any str input as a NEW user message and
    re-applies the chat template — which closes the assistant turn and reopens
    a fresh one, breaking the continuous `<think>…</think>{answer}` span the
    model was trained on (constrained lists then collapse). Disabling this makes
    outlines treat the prompt as a verbatim continuation.
    """
    adapter = getattr(outlines_model, "type_adapter", None)
    if adapter is not None and hasattr(adapter, "has_chat_template"):
        adapter.has_chat_template = False
    return outlines_model


def _reasoning_prefix(text: str) -> str:
    end_tag = "</think>"
    if end_tag in text:
        return text[: text.index(end_tag) + len(end_tag)]
    text = text.strip()
    if text.startswith("<think>"):
        return f"{text}{end_tag}"
    return f"<think>{text}</think>"


class MLXBackend(Backend):
    def __init__(self, model_path: str):
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "MLX backend requires mlx-lm. Install it with:\n\n"
                '  pip install "neural-txt[mlx]"'
            ) from None
        self.model, self.tokenizer = load(model_path)
        self._outlines_model = None
        self._gen_cache: dict = {}

    def _json_generator(self, output_type):
        """Cache compiled outlines generators per schema.

        Building the schema -> FSM index is expensive (seconds for object
        schemas) and is NOT cached by outlines across calls, so we build the
        Generator once per output_type and reuse it. After the first build, the
        first streamed token arrives in tens of ms instead of ~5s.
        """
        import outlines
        if self._outlines_model is None:
            self._outlines_model = _no_retemplate(outlines.from_mlxlm(self.model, self.tokenizer))
        key = str(output_type)
        gen = self._gen_cache.get(key)
        if gen is None:
            gen = outlines.Generator(self._outlines_model, output_type)
            self._gen_cache[key] = gen
        return gen

    def warmup_json(self, output_types) -> None:
        """Pre-build generators (off the interactive path) so the first JSON
        generation per schema doesn't pay the ~5s index-compile pause live."""
        for output_type in output_types:
            self._json_generator(output_type)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_many(prompt, **kwargs)[0]

    def generate_many(self, prompt: str, **kwargs) -> list[str]:
        from mlx_lm import generate as mlx_generate
        from mlx_lm.sample_utils import make_sampler

        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        num_beams = _get_num_beams(kwargs)
        num_return_sequences = _get_num_return_sequences(kwargs, num_beams)

        return [
            mlx_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                sampler=make_sampler(temp=temperature),
                verbose=False,
                **kwargs,
            )
            for _ in range(num_return_sequences)
        ]

    def generate_json(self, prompt: str, output_type, **kwargs) -> str:
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        result = self._json_generator(output_type)(prompt, max_tokens=max_new_tokens)
        print(f"\n[RAW JSON OUTPUT]\n{result}\n[/RAW JSON OUTPUT]\n", flush=True)
        return result

    def generate_reasoned_json(self, prompt: str, output_type, **kwargs) -> str:
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        reasoning_max_new_tokens = kwargs.pop("reasoning_max_new_tokens", 256)
        _get_num_beams(kwargs)

        reasoning_text = "<think>"
        for response in stream_generate(
            self.model,
            self.tokenizer,
            f"{prompt}<think>",
            sampler=make_sampler(temp=temperature),
            max_tokens=reasoning_max_new_tokens,
        ):
            reasoning_text += response.text
            if "</think>" in reasoning_text:
                break

        prefix = _reasoning_prefix(reasoning_text)
        # The model was trained to emit the answer directly after </think>
        # (it naturally produces "</think>\n\n<output>"). Do NOT inject an extra
        # instruction here — that text is out-of-distribution and derails
        # constrained decoding (bare lists collapse to [","]).
        json_prompt = f"{prompt}{prefix}\n\n"
        result = self._json_generator(output_type)(json_prompt, max_tokens=max_new_tokens)
        raw = f"{prefix}{result}"
        print(f"\n[RAW REASONED JSON OUTPUT]\n{raw}\n[/RAW REASONED JSON OUTPUT]\n", flush=True)
        return raw

    def stream(self, prompt: str, **kwargs):
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        self._last_stats = {}
        response = None
        full_text = ""
        for response in stream_generate(
            self.model, self.tokenizer, prompt,
            sampler=make_sampler(temp=temperature),
            max_tokens=max_new_tokens,
        ):
            full_text += response.text
            yield response.text
        if response is not None:
            self._last_stats = {
                "tokens":      response.generation_tokens,
                "tps":         response.generation_tps,
                "peak_memory": response.peak_memory,
            }
            print(f"\n[RAW OUTPUT]\n{full_text}\n[/RAW OUTPUT]\n", flush=True)

    def stream_json(self, prompt: str, output_type, **kwargs):
        """Stream schema-constrained JSON token-by-token via outlines."""
        import time
        import mlx.core as mx

        kwargs.pop("temperature", None)  # constrained decoding is greedy
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        self._last_stats = {}
        generator = self._json_generator(output_type)
        mx.reset_peak_memory()
        token_count = 0
        full_text = ""
        t0 = time.perf_counter()
        for chunk in generator.stream(prompt, max_tokens=max_new_tokens):
            full_text += chunk
            try:
                token_count += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            except TypeError:
                token_count += len(self.tokenizer.encode(chunk))
            yield chunk
        elapsed = time.perf_counter() - t0
        self._last_stats = {
            "tokens":      token_count,
            "tps":         token_count / elapsed if elapsed > 0 else 0.0,
            "peak_memory": mx.get_peak_memory() / 1e9,
        }
        print(f"\n[RAW JSON OUTPUT]\n{full_text}\n[/RAW JSON OUTPUT]\n", flush=True)

    def stream_reasoned_json(self, prompt: str, output_type, **kwargs):
        """Stream the <think>…</think> reasoning, then schema-constrained JSON."""
        import time
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        reasoning_max_new_tokens = kwargs.pop("reasoning_max_new_tokens", 256)
        _get_num_beams(kwargs)
        self._last_stats = {}
        mx.reset_peak_memory()
        t0 = time.perf_counter()

        # Phase 1 — stream reasoning.
        reasoning_text = "<think>"
        yield "<think>"
        resp = None
        for resp in stream_generate(
            self.model, self.tokenizer, f"{prompt}<think>",
            sampler=make_sampler(temp=temperature),
            max_tokens=reasoning_max_new_tokens,
        ):
            reasoning_text += resp.text
            yield resp.text
            if "</think>" in reasoning_text:
                break
        if "</think>" not in reasoning_text:
            reasoning_text += "</think>"
            yield "</think>"
        reasoning_tokens = resp.generation_tokens if resp is not None else 0

        # Phase 2 — stream constrained JSON, conditioned on the reasoning.
        prefix = _reasoning_prefix(reasoning_text)
        # The model was trained to emit the answer directly after </think>
        # (it naturally produces "</think>\n\n<output>"). Do NOT inject an extra
        # instruction here — that text is out-of-distribution and derails
        # constrained decoding (bare lists collapse to [","]).
        json_prompt = f"{prompt}{prefix}\n\n"
        generator = self._json_generator(output_type)
        json_tokens = 0
        for chunk in generator.stream(json_prompt, max_tokens=max_new_tokens):
            try:
                json_tokens += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            except TypeError:
                json_tokens += len(self.tokenizer.encode(chunk))
            yield chunk

        elapsed = time.perf_counter() - t0
        token_count = reasoning_tokens + json_tokens
        self._last_stats = {
            "tokens":      token_count,
            "tps":         token_count / elapsed if elapsed > 0 else 0.0,
            "peak_memory": mx.get_peak_memory() / 1e9,
        }


class HFBackend(Backend):
    def __init__(self, model_path: str):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires torch. Install it with:\n\n"
                '  pip install "neural-txt[hf]"'
            ) from None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._torch = torch
        self._outlines_model = None
        self._gen_cache: dict = {}

    def _json_generator(self, output_type):
        """Cache compiled outlines generators per schema (see MLXBackend)."""
        import outlines
        if self._outlines_model is None:
            self._outlines_model = _no_retemplate(outlines.from_transformers(self.model, self.tokenizer))
        key = str(output_type)
        gen = self._gen_cache.get(key)
        if gen is None:
            gen = outlines.Generator(self._outlines_model, output_type)
            self._gen_cache[key] = gen
        return gen

    def warmup_json(self, output_types) -> None:
        """Pre-build generators off the interactive path."""
        for output_type in output_types:
            self._json_generator(output_type)

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_many(prompt, **kwargs)[0]

    def generate_many(self, prompt: str, **kwargs) -> list[str]:
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        num_beams = _get_num_beams(kwargs)
        num_return_sequences = _get_num_return_sequences(kwargs, num_beams)

        reasoning_prompt = f"{prompt}<think>"
        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
        if num_beams > 1:
            generation_kwargs.update(
                do_sample=False,
                num_beams=max(num_beams, num_return_sequences),
                num_return_sequences=num_return_sequences,
            )
        else:
            generation_kwargs.update(
                do_sample=temperature > 0 or num_return_sequences > 1,
                temperature=temperature if temperature > 0 else 1.0,
                num_return_sequences=num_return_sequences,
            )

        with self._torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        return self.tokenizer.batch_decode(
            output_ids[:, input_ids.shape[1]:],
            skip_special_tokens=True,
        )

    def generate_json(self, prompt: str, output_type, **kwargs) -> str:
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        result = self._json_generator(output_type)(prompt, max_new_tokens=max_new_tokens)
        print(f"\n[RAW JSON OUTPUT]\n{result}\n[/RAW JSON OUTPUT]\n", flush=True)
        return result

    def generate_reasoned_json(self, prompt: str, output_type, **kwargs) -> str:
        from transformers import StoppingCriteria, StoppingCriteriaList

        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        reasoning_max_new_tokens = kwargs.pop("reasoning_max_new_tokens", 256)
        num_beams = _get_num_beams(kwargs)

        class StopOnText(StoppingCriteria):
            def __init__(self, tokenizer, start_length: int, stop_text: str):
                self.tokenizer = tokenizer
                self.start_length = start_length
                self.stop_text = stop_text

            def __call__(self, input_ids, scores, **kwargs):
                generated = self.tokenizer.decode(
                    input_ids[0, self.start_length:],
                    skip_special_tokens=True,
                )
                return self.stop_text in generated

        reasoning_prompt = f"{prompt}<think>"
        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)

        generation_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=reasoning_max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            stopping_criteria=StoppingCriteriaList(
                [StopOnText(self.tokenizer, input_ids.shape[1], "</think>")]
            ),
            **kwargs,
        )
        if num_beams > 1:
            generation_kwargs.update(do_sample=False, num_beams=num_beams)
        else:
            generation_kwargs.update(
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else 1.0,
            )

        with self._torch.no_grad():
            output_ids = self.model.generate(**generation_kwargs)

        reasoning_text = self.tokenizer.decode(
            output_ids[0, input_ids.shape[1]:],
            skip_special_tokens=True,
        )
        reasoning_text = f"<think>{reasoning_text}"
        prefix = _reasoning_prefix(reasoning_text)
        # The model was trained to emit the answer directly after </think>
        # (it naturally produces "</think>\n\n<output>"). Do NOT inject an extra
        # instruction here — that text is out-of-distribution and derails
        # constrained decoding (bare lists collapse to [","]).
        json_prompt = f"{prompt}{prefix}\n\n"
        result = self._json_generator(output_type)(json_prompt, max_new_tokens=max_new_tokens)
        raw = f"{prefix}{result}"
        print(f"\n[RAW REASONED JSON OUTPUT]\n{raw}\n[/RAW REASONED JSON OUTPUT]\n", flush=True)
        return raw

    def stream(self, prompt: str, **kwargs):
        import time
        from transformers import TextIteratorStreamer
        from threading import Thread
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        num_beams = _get_num_beams(kwargs)
        if num_beams != 1:
            raise ValueError("stream() only supports num_beams=1")
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        thread = Thread(target=self.model.generate, kwargs=dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        ), daemon=True)
        thread.start()
        token_count = 0
        t0 = time.perf_counter()
        for chunk in streamer:
            token_count += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            yield chunk
        elapsed = time.perf_counter() - t0
        peak_memory = self._torch.cuda.max_memory_allocated() / 1e9 if self._torch.cuda.is_available() else 0.0
        self._last_stats = {
            "tokens": token_count,
            "tps": token_count / elapsed if elapsed > 0 else 0.0,
            "peak_memory": peak_memory,
        }

    def stream_json(self, prompt: str, output_type, **kwargs):
        """Stream schema-constrained JSON token-by-token via outlines."""
        import time

        kwargs.pop("temperature", None)  # constrained decoding is greedy
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        self._last_stats = {}
        generator = self._json_generator(output_type)
        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        token_count = 0
        full_text = ""
        t0 = time.perf_counter()
        for chunk in generator.stream(prompt, max_new_tokens=max_new_tokens):
            full_text += chunk
            token_count += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            yield chunk
        elapsed = time.perf_counter() - t0
        peak_memory = self._torch.cuda.max_memory_allocated() / 1e9 if self._torch.cuda.is_available() else 0.0
        self._last_stats = {
            "tokens": token_count,
            "tps": token_count / elapsed if elapsed > 0 else 0.0,
            "peak_memory": peak_memory,
        }
        print(f"\n[RAW JSON OUTPUT]\n{full_text}\n[/RAW JSON OUTPUT]\n", flush=True)

    def stream_reasoned_json(self, prompt: str, output_type, **kwargs):
        """Stream the <think>…</think> reasoning, then schema-constrained JSON."""
        import time
        from transformers import TextIteratorStreamer
        from threading import Thread

        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        reasoning_max_new_tokens = kwargs.pop("reasoning_max_new_tokens", 256)
        _get_num_beams(kwargs)
        self._last_stats = {}
        if self._torch.cuda.is_available():
            self._torch.cuda.reset_peak_memory_stats()
        t0 = time.perf_counter()
        token_count = 0

        # Phase 1 — stream reasoning.
        reasoning_prompt = f"{prompt}<think>"
        inputs = self.tokenizer(reasoning_prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = getattr(inputs, "attention_mask", None)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.model.device)
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True, skip_prompt=True)
        thread = Thread(target=self.model.generate, kwargs=dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=reasoning_max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else 1.0,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id,
            streamer=streamer,
        ), daemon=True)
        thread.start()

        reasoning_text = "<think>"
        yield "<think>"
        for chunk in streamer:
            reasoning_text += chunk
            token_count += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            yield chunk
            if "</think>" in reasoning_text:
                break
        if "</think>" not in reasoning_text:
            reasoning_text += "</think>"
            yield "</think>"

        # Phase 2 — stream constrained JSON, conditioned on the reasoning.
        prefix = _reasoning_prefix(reasoning_text)
        # The model was trained to emit the answer directly after </think>
        # (it naturally produces "</think>\n\n<output>"). Do NOT inject an extra
        # instruction here — that text is out-of-distribution and derails
        # constrained decoding (bare lists collapse to [","]).
        json_prompt = f"{prompt}{prefix}\n\n"
        generator = self._json_generator(output_type)
        for chunk in generator.stream(json_prompt, max_new_tokens=max_new_tokens):
            token_count += len(self.tokenizer.encode(chunk, add_special_tokens=False))
            yield chunk

        elapsed = time.perf_counter() - t0
        peak_memory = self._torch.cuda.max_memory_allocated() / 1e9 if self._torch.cuda.is_available() else 0.0
        self._last_stats = {
            "tokens": token_count,
            "tps": token_count / elapsed if elapsed > 0 else 0.0,
            "peak_memory": peak_memory,
        }


def load_backend(model_path: str, mlx: bool = False) -> Backend:
    if mlx:
        return MLXBackend(model_path)
    return HFBackend(model_path)
