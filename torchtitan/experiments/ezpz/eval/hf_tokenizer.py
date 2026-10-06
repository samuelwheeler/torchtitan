# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import shutil
from pathlib import Path

import sentencepiece as spm


def validate_tokenizer(hf_assets_path, vocab_size):
    assets = Path(hf_assets_path) if hf_assets_path is not None else None
    if assets is None or not (assets / "tokenizer.model").is_file():
        raise FileNotFoundError("AGPT export requires hf_assets_path/tokenizer.model")
    tokenizer = spm.SentencePieceProcessor(model_file=str(assets / "tokenizer.model"))
    if not 0 < tokenizer.get_piece_size() <= vocab_size:
        raise ValueError(
            f"Tokenizer vocabulary {tokenizer.get_piece_size()} exceeds model "
            f"vocabulary {vocab_size}"
        )
    if (
        (tokenizer.unk_id(), tokenizer.bos_id(), tokenizer.eos_id()) != (0, 1, 2)
        or [tokenizer.id_to_piece(i) for i in range(3)] != ["<unk>", "<s>", "</s>"]
        or tokenizer.pad_id() != -1
    ):
        raise ValueError("AGPT export requires Llama SentencePiece special tokens")


def write_tokenizer_assets(output_dir, hf_assets_path, vocab_size, max_length):
    validate_tokenizer(hf_assets_path, vocab_size)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(hf_assets_path) / "tokenizer.model", output / "tokenizer.model")
    config = {
        "add_bos_token": True,
        "add_eos_token": False,
        "bos_token": "<s>",
        "eos_token": "</s>",
        "unk_token": "<unk>",
        "model_max_length": max_length,
        "tokenizer_class": "LlamaTokenizer",
    }
    (output / "tokenizer_config.json").write_text(json.dumps(config, indent=2) + "\n")
    (output / "special_tokens_map.json").write_text(
        json.dumps({"bos_token": "<s>", "eos_token": "</s>", "unk_token": "<unk>"},
                   indent=2) + "\n"
    )
