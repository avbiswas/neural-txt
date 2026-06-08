from __future__ import annotations

from pathlib import Path
import struct
import zipfile


REWARD_HIDDEN_SIZE = 384
REWARD_POOLED_SIZE = REWARD_HIDDEN_SIZE * 2
REWARD_INTERMEDIATE_SIZE = 1536
REWARD_NUM_HEADS = 12
REWARD_NUM_LAYERS = 6
REWARD_LAYER_NORM_EPS = 1e-12


def _meanmax_pool(mx, hidden, mask):
    m = mx.expand_dims(mask, -1).astype(hidden.dtype)
    mean = (hidden * m).sum(axis=1) / mx.clip(m.sum(axis=1), 1e-9, None)
    masked = mx.where(m == 0, mx.full(hidden.shape, -float("inf")), hidden)
    mx_pool = masked.max(axis=1)
    return mx.concatenate([mean, mx_pool], axis=-1)


class _BertEmbeddings:
    def __init__(self, nn):
        self.word_embeddings = nn.Embedding(30522, REWARD_HIDDEN_SIZE)
        self.position_embeddings = nn.Embedding(512, REWARD_HIDDEN_SIZE)
        self.token_type_embeddings = nn.Embedding(2, REWARD_HIDDEN_SIZE)
        self.layer_norm = nn.LayerNorm(
            REWARD_HIDDEN_SIZE, eps=REWARD_LAYER_NORM_EPS
        )

    def __call__(self, mx, input_ids, token_type_ids):
        seq_len = input_ids.shape[1]
        position_ids = mx.broadcast_to(mx.arange(seq_len), input_ids.shape)
        embeddings = (
            self.word_embeddings(input_ids)
            + self.position_embeddings(position_ids)
            + self.token_type_embeddings(token_type_ids)
        )
        return self.layer_norm(embeddings)

    def load_weights(self, weights):
        self.word_embeddings.weight = weights["embeddings.word_embeddings.weight"]
        self.position_embeddings.weight = weights["embeddings.position_embeddings.weight"]
        self.token_type_embeddings.weight = weights["embeddings.token_type_embeddings.weight"]
        self.layer_norm.weight = weights["embeddings.LayerNorm.gamma"]
        self.layer_norm.bias = weights["embeddings.LayerNorm.beta"]


class _BertLayer:
    def __init__(self, nn):
        self.query = nn.Linear(REWARD_HIDDEN_SIZE, REWARD_HIDDEN_SIZE)
        self.key = nn.Linear(REWARD_HIDDEN_SIZE, REWARD_HIDDEN_SIZE)
        self.value = nn.Linear(REWARD_HIDDEN_SIZE, REWARD_HIDDEN_SIZE)
        self.attention_output = nn.Linear(REWARD_HIDDEN_SIZE, REWARD_HIDDEN_SIZE)
        self.attention_layer_norm = nn.LayerNorm(
            REWARD_HIDDEN_SIZE, eps=REWARD_LAYER_NORM_EPS
        )
        self.intermediate = nn.Linear(REWARD_HIDDEN_SIZE, REWARD_INTERMEDIATE_SIZE)
        self.output = nn.Linear(REWARD_INTERMEDIATE_SIZE, REWARD_HIDDEN_SIZE)
        self.output_layer_norm = nn.LayerNorm(
            REWARD_HIDDEN_SIZE, eps=REWARD_LAYER_NORM_EPS
        )

    def __call__(self, mx, nn, hidden_states, attention_mask):
        batch_size, seq_len, _ = hidden_states.shape
        head_dim = REWARD_HIDDEN_SIZE // REWARD_NUM_HEADS

        query = self.query(hidden_states)
        key = self.key(hidden_states)
        value = self.value(hidden_states)

        query = query.reshape(
            batch_size, seq_len, REWARD_NUM_HEADS, head_dim
        ).transpose(0, 2, 1, 3)
        key = key.reshape(
            batch_size, seq_len, REWARD_NUM_HEADS, head_dim
        ).transpose(0, 2, 1, 3)
        value = value.reshape(
            batch_size, seq_len, REWARD_NUM_HEADS, head_dim
        ).transpose(0, 2, 1, 3)

        attention_scores = (query @ key.transpose(0, 1, 3, 2)) / (head_dim ** 0.5)
        attention_scores = attention_scores + attention_mask
        attention_probs = mx.softmax(attention_scores, axis=-1)
        attention_context = attention_probs @ value
        attention_context = attention_context.transpose(0, 2, 1, 3).reshape(
            batch_size, seq_len, REWARD_HIDDEN_SIZE
        )

        attention_output = self.attention_output(attention_context)
        hidden_states = self.attention_layer_norm(attention_output + hidden_states)

        layer_output = self.output(nn.gelu(self.intermediate(hidden_states)))
        return self.output_layer_norm(layer_output + hidden_states)

    def load_weights(self, weights, index: int):
        prefix = f"encoder.layer.{index}"
        self.query.weight = weights[f"{prefix}.attention.self.query.weight"]
        self.query.bias = weights[f"{prefix}.attention.self.query.bias"]
        self.key.weight = weights[f"{prefix}.attention.self.key.weight"]
        self.key.bias = weights[f"{prefix}.attention.self.key.bias"]
        self.value.weight = weights[f"{prefix}.attention.self.value.weight"]
        self.value.bias = weights[f"{prefix}.attention.self.value.bias"]
        self.attention_output.weight = weights[f"{prefix}.attention.output.dense.weight"]
        self.attention_output.bias = weights[f"{prefix}.attention.output.dense.bias"]
        self.attention_layer_norm.weight = weights[f"{prefix}.attention.output.LayerNorm.gamma"]
        self.attention_layer_norm.bias = weights[f"{prefix}.attention.output.LayerNorm.beta"]
        self.intermediate.weight = weights[f"{prefix}.intermediate.dense.weight"]
        self.intermediate.bias = weights[f"{prefix}.intermediate.dense.bias"]
        self.output.weight = weights[f"{prefix}.output.dense.weight"]
        self.output.bias = weights[f"{prefix}.output.dense.bias"]
        self.output_layer_norm.weight = weights[f"{prefix}.output.LayerNorm.gamma"]
        self.output_layer_norm.bias = weights[f"{prefix}.output.LayerNorm.beta"]


class MLXRewardModel:
    def __init__(self, model_path: str):
        try:
            import mlx.core as mx
            import mlx.nn as nn
        except ImportError:
            raise ImportError(
                "MLX reward backend requires mlx. Install it with:\n\n"
                '  uv pip install -e ".[mlx]"'
            ) from None

        from transformers import AutoTokenizer

        self.mx = mx
        self.nn = nn
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.embeddings = _BertEmbeddings(nn)
        self.layers = [_BertLayer(nn) for _ in range(REWARD_NUM_LAYERS)]
        self.head_weight, self.head_bias = self._load_head(model_path)

        weights_path = self._resolve_file(model_path, "model.safetensors")
        weights = mx.load(weights_path)
        self.embeddings.load_weights(weights)
        for index, layer in enumerate(self.layers):
            layer.load_weights(weights, index)

    def _resolve_file(self, model_path: str, filename: str) -> str:
        local_path = Path(model_path) / filename
        if local_path.exists():
            return str(local_path)

        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError(
                "Loading the reward model from Hugging Face requires huggingface_hub."
            ) from None
        return hf_hub_download(repo_id=model_path, filename=filename)

    def _resolve_optional_file(self, model_path: str, filename: str) -> str | None:
        try:
            return self._resolve_file(model_path, filename)
        except Exception:
            return None

    def _load_head(self, model_path: str):
        head_path = self._resolve_optional_file(model_path, "reward_head.safetensors")
        if head_path is None:
            head_path = self._resolve_file(model_path, "head_weights.pt")

        if head_path.endswith(".safetensors"):
            weights = self.mx.load(head_path)
            return weights["weight"], weights["bias"]

        if not zipfile.is_zipfile(head_path):
            raise ValueError(f"Unsupported reward head format: {head_path}")

        with zipfile.ZipFile(head_path) as archive:
            prefix = archive.namelist()[0].split("/", 1)[0]
            weight_bytes = archive.read(f"{prefix}/data/0")
            bias_bytes = archive.read(f"{prefix}/data/1")

        weight = struct.unpack(f"<{REWARD_POOLED_SIZE}f", weight_bytes)
        bias = struct.unpack("<1f", bias_bytes)
        return self.mx.array([weight]), self.mx.array(bias)

    def __call__(self, input_ids, attention_mask, token_type_ids):
        mx = self.mx
        hidden_states = self.embeddings(mx, input_ids, token_type_ids)
        extended_mask = mx.expand_dims(mx.expand_dims(attention_mask, 1), 1)
        extended_mask = (1.0 - extended_mask.astype(hidden_states.dtype)) * -10000.0
        for layer in self.layers:
            hidden_states = layer(mx, self.nn, hidden_states, extended_mask)
        pooled = _meanmax_pool(mx, hidden_states, attention_mask)
        logits = pooled @ self.head_weight.T + self.head_bias
        return mx.clip(logits.squeeze(-1), 0.0, 1.0)
