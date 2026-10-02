import torch

from torchtitan.experiments.ezpz.eval.convert_to_hf import (
    _prepare_hf_state_dict,
)


def test_prepare_hf_state_dict_casts_and_packs_transpose_views():
    transposed = torch.arange(24, dtype=torch.float32).view(4, 6).T
    assert not transposed.is_contiguous()

    prepared = _prepare_hf_state_dict({"expert.weight": transposed}, torch.bfloat16)

    weight = prepared["expert.weight"]
    assert weight.dtype == torch.bfloat16
    assert weight.is_contiguous()
    torch.testing.assert_close(weight.float(), transposed)


def test_prepare_hf_state_dict_packs_without_dtype_change():
    transposed = torch.arange(12, dtype=torch.float32).view(3, 4).T

    weight = _prepare_hf_state_dict({"weight": transposed}, torch.float32)["weight"]

    assert weight.dtype == torch.float32
    assert weight.is_contiguous()
    torch.testing.assert_close(weight, transposed)
