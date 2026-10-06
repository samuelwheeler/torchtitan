import json
from dataclasses import dataclass
from pathlib import Path

from torchtitan.experiments.ezpz.eval.hf_tokenizer import (
    validate_tokenizer,
    write_tokenizer_assets,
)

from .model import AgptModel
from .state_dict_adapter import AgptStateDictAdapter


class AGPTDenseStateDictAdapter(AgptStateDictAdapter):
    """Export the data-matched 2B/50K model with its training tokenizer."""

    def validate_hf_assets(self):
        validate_tokenizer(self.hf_assets_path, self.model_config.vocab_size)

    def write_hf_assets(self, output_dir, export_dtype):
        model = self.model_config
        layer = model.layers[0]
        attention = layer.attention
        config = {
            "architectures": ["LlamaForCausalLM"],
            "model_type": "llama",
            "hidden_size": model.dim,
            "intermediate_size": layer.feed_forward.w13.out_features,
            "num_hidden_layers": len(model.layers),
            "num_attention_heads": attention.n_heads,
            "num_key_value_heads": attention.n_kv_heads,
            "max_position_embeddings": model.max_context_length,
            "rope_theta": attention.rope.theta,
            "rms_norm_eps": layer.attention_norm.eps,
            "vocab_size": model.vocab_size,
            "hidden_act": "silu",
            "attention_bias": False,
            "mlp_bias": False,
            "attention_dropout": 0.0,
            "bos_token_id": 1,
            "eos_token_id": 2,
            "pad_token_id": None,
            "tie_word_embeddings": model.enable_weight_tying,
            "torch_dtype": export_dtype,
        }
        write_tokenizer_assets(output_dir, self.hf_assets_path, model.vocab_size,
                               model.max_context_length)
        (Path(output_dir) / "config.json").write_text(
            json.dumps(config, indent=2) + "\n"
        )


class AGPTDense50KModel(AgptModel):
    state_dict_adapter_cls = AGPTDenseStateDictAdapter

    @dataclass(kw_only=True, slots=True)
    class Config(AgptModel.Config):
        pass
