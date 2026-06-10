from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from .types import RankedResponse


DEFAULT_REWARD_MODEL = "paperbd/neuraltxt-reward-tiny"
DEFAULT_MLX_REWARD_MODEL = "paperbd/neuraltxt-reward-tiny-mlx"
DEFAULT_MAX_LENGTH = 512
DEFAULT_BATCH_SIZE = 64


def _meanmax_pool(hidden, mask):
    import torch

    m = mask.unsqueeze(-1).float()
    mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)
    mx = hidden.masked_fill(m == 0, float("-inf")).max(1).values
    return torch.cat([mean, mx], dim=-1)


class NeuralTxtReward:
    """
    Reward-model interface for scoring responses against a reference answer.

    Args:
        model_path: Hugging Face repo id or local model directory.
        backend: "hf" for HuggingFace Transformers or "mlx" for Apple Silicon MLX.
        device: Optional torch device string, e.g. "cpu", "cuda", or "mps".
                Only used by the "hf" backend.
        max_length: Tokenizer truncation length.
    """

    def __init__(
        self,
        model_path: str | None = None,
        backend: str = "hf",
        device: str | None = None,
        max_length: int = DEFAULT_MAX_LENGTH,
    ):
        if backend not in ("hf", "mlx"):
            raise ValueError(f"backend must be 'hf' or 'mlx', got {backend!r}")

        if model_path is None:
            model_path = (
                DEFAULT_MLX_REWARD_MODEL if backend == "mlx" else DEFAULT_REWARD_MODEL
            )

        self.backend = backend
        self.max_length = max_length
        if backend == "mlx":
            if device is not None:
                raise ValueError("device is only supported with backend='hf'")
            from .reward_mlx import MLXRewardModel

            self._mlx_model = MLXRewardModel(model_path)
            self.tokenizer = self._mlx_model.tokenizer
            return

        try:
            import torch
            import torch.nn as nn
        except ImportError:
            raise ImportError(
                "NeuralTxtReward requires torch. Install it with:\n\n"
                '  pip install "neural-txt[hf]"'
            ) from None

        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.encoder = AutoModel.from_pretrained(model_path)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(768, 1))

        head_path = self._resolve_head_path(model_path)
        try:
            state_dict = torch.load(head_path, map_location="cpu", weights_only=True)
        except TypeError:
            state_dict = torch.load(head_path, map_location="cpu")
        self.head.load_state_dict(state_dict)

        if device is not None:
            self.encoder.to(device)
            self.head.to(device)

        self.encoder.eval()
        self.head.eval()
        self._torch = torch

    def _resolve_head_path(self, model_path: str) -> str:
        local_path = Path(model_path) / "head_weights.pt"
        if local_path.exists():
            return str(local_path)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Loading the reward head from Hugging Face requires huggingface_hub."
            ) from None
        return hf_hub_download(repo_id=model_path, filename="head_weights.pt")

    def _normalize_references(
        self, responses: Sequence[str], reference: str | Sequence[str]
    ) -> list[str]:
        if isinstance(reference, str):
            return [reference] * len(responses)
        if len(reference) != len(responses):
            raise ValueError(
                "reference must be a string or a sequence with the same length as "
                f"responses; got {len(reference)} references for "
                f"{len(responses)} responses"
            )
        return list(reference)

    def _build_inputs(
        self, responses: Sequence[str], references: Sequence[str]
    ) -> list[str]:
        return [
            f"{reference} [SEP] {response}"
            for response, reference in zip(responses, references)
        ]

    def score(self, response: str, reference: str) -> float:
        """Score one response against a reference answer on a 0-1 scale."""
        return self.batch_score([response], reference)[0]

    def batch_score(
        self,
        responses: Sequence[str],
        reference: str | Sequence[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> list[float]:
        """Score responses against one reference or paired references."""
        if not responses:
            return []
        if batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {batch_size}")

        references = self._normalize_references(responses, reference)
        scores: list[float] = []
        for start in range(0, len(responses), batch_size):
            batch_responses = responses[start : start + batch_size]
            batch_references = references[start : start + batch_size]
            if self.backend == "mlx":
                scores.extend(self._score_mlx_batch(batch_responses, batch_references))
            else:
                scores.extend(self._score_hf_batch(batch_responses, batch_references))
        return scores

    def _score_hf_batch(
        self, responses: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        enc = self.tokenizer(
            self._build_inputs(responses, references),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        device = next(self.encoder.parameters()).device
        enc = {key: value.to(device) for key, value in enc.items()}

        with self._torch.no_grad():
            out = self.encoder(**enc)
            pooled = _meanmax_pool(out.last_hidden_state, enc["attention_mask"])
            scores = self.head(pooled).squeeze(-1).clamp(0.0, 1.0)

        return [float(score) for score in scores.detach().cpu().tolist()]

    def _score_mlx_batch(
        self, responses: Sequence[str], references: Sequence[str]
    ) -> list[float]:
        enc = self.tokenizer(
            self._build_inputs(responses, references),
            return_tensors="np",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        )
        mx = self._mlx_model.mx
        input_ids = mx.array(enc["input_ids"])
        attention_mask = mx.array(enc["attention_mask"])
        token_type_ids = mx.array(enc.get("token_type_ids", enc["attention_mask"] * 0))
        scores = self._mlx_model(input_ids, attention_mask, token_type_ids)
        mx.eval(scores)
        return [float(score) for score in scores.tolist()]

    def rank(
        self, responses: Sequence[str], reference: str | Sequence[str]
    ) -> list[RankedResponse]:
        """Rank responses against a reference answer, highest score first."""
        scores = self.batch_score(responses, reference)
        ranked = [
            RankedResponse(index=index, response=response, score=score)
            for index, (response, score) in enumerate(zip(responses, scores))
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)
