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
    def stream(self, prompt: str, **kwargs): ...  # yields str chunks


class MLXBackend(Backend):
    def __init__(self, model_path: str):
        try:
            from mlx_lm import load
        except ImportError:
            raise ImportError(
                "MLX backend requires mlx-lm. Install it with:\n\n"
                '  uv pip install -e ".[mlx]"'
            ) from None
        self.model, self.tokenizer = load(model_path)

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
        import outlines

        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        outlines_model = outlines.from_mlxlm(self.model, self.tokenizer)
        result = outlines_model(prompt, output_type=output_type, max_tokens=max_new_tokens)
        print(f"\n[RAW JSON OUTPUT]\n{result}\n[/RAW JSON OUTPUT]\n", flush=True)
        return result

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


class HFBackend(Backend):
    def __init__(self, model_path: str):
        try:
            import torch
        except ImportError:
            raise ImportError(
                "HuggingFace backend requires torch. Install it with:\n\n"
                '  uv pip install -e ".[hf]"'
            ) from None
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._torch = torch

    def generate(self, prompt: str, **kwargs) -> str:
        return self.generate_many(prompt, **kwargs)[0]

    def generate_many(self, prompt: str, **kwargs) -> list[str]:
        temperature = kwargs.pop("temperature", DEFAULT_TEMPERATURE)
        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        num_beams = _get_num_beams(kwargs)
        num_return_sequences = _get_num_return_sequences(kwargs, num_beams)

        inputs = self.tokenizer(prompt, return_tensors="pt")
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
        import outlines

        max_new_tokens = kwargs.pop("max_new_tokens", DEFAULT_MAX_NEW_TOKENS)
        _get_num_beams(kwargs)
        outlines_model = outlines.from_transformers(self.model, self.tokenizer)
        result = outlines_model(prompt, output_type=output_type, max_new_tokens=max_new_tokens)
        print(f"\n[RAW JSON OUTPUT]\n{result}\n[/RAW JSON OUTPUT]\n", flush=True)
        return result

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


def load_backend(model_path: str, mlx: bool = False) -> Backend:
    if mlx:
        return MLXBackend(model_path)
    return HFBackend(model_path)
