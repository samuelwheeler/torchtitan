# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Focused tests for the optional XPU and TorchStore compatibility hooks."""

from types import SimpleNamespace

import pytest


def test_torchstore_strategy_defaults_to_automatic_transport(monkeypatch):
    from torchtitan.torchstore_compat import torchstore_transport_from_env

    monkeypatch.delenv("TORCHTITAN_TORCHSTORE_TRANSPORT", raising=False)

    assert torchstore_transport_from_env() is None


def test_torchstore_strategy_rejects_unknown_transport(monkeypatch):
    from torchtitan.torchstore_compat import torchstore_transport_from_env

    monkeypatch.setenv("TORCHTITAN_TORCHSTORE_TRANSPORT", "bogus")

    with pytest.raises(ValueError, match="got 'bogus'"):
        torchstore_transport_from_env()


def test_torchstore_strategy_selects_monarch_rdma(monkeypatch):
    from torchtitan.torchstore_compat import torchstore_transport_from_env

    transport_module = SimpleNamespace(
        TransportType=SimpleNamespace(MonarchRDMA="monarch-rdma")
    )
    monkeypatch.setitem(
        __import__("sys").modules, "torchstore.transport", transport_module
    )
    monkeypatch.setenv("TORCHTITAN_TORCHSTORE_TRANSPORT", "monarch_rdma")

    assert torchstore_transport_from_env() == "monarch-rdma"


def test_xpu_patch_excludes_unqualified_network_backends_from_auto(monkeypatch):
    import sys

    from torchtitan.experiments.ezpz.rl import xpu_overrides

    available = lambda: True
    monarch_rdma = SimpleNamespace(monarch_rdma_transport_available=available)
    xccl = SimpleNamespace(xccl_available=available)

    class GlooTransportBuffer:
        pass

    create_transport_buffer = lambda _ref: GlooTransportBuffer()
    transport = SimpleNamespace(
        monarch_rdma_transport_available=available,
        xccl_available=available,
        torchcomms_uniflow_available=available,
        torchcomms_rdma_available=available,
        get_available_transport=lambda _ref: SimpleNamespace(name="Gloo"),
        create_transport_buffer=create_transport_buffer,
        _log_transport_resolution=lambda _ref, _transport: None,
        monarch_rdma=monarch_rdma,
        xccl=xccl,
    )
    torchstore = SimpleNamespace(transport=transport)

    monkeypatch.setitem(sys.modules, "torchstore", torchstore)
    monkeypatch.setitem(sys.modules, "torchstore.transport", transport)
    monkeypatch.setitem(sys.modules, "torchstore.transport.monarch_rdma", monarch_rdma)
    monkeypatch.setitem(sys.modules, "torchstore.transport.xccl", xccl)
    monkeypatch.setattr(
        xpu_overrides.torch, "version", SimpleNamespace(xpu="2026.1"), raising=False
    )

    xpu_overrides.patch_torchstore_network_availability_for_xpu()

    assert not monarch_rdma.monarch_rdma_transport_available()
    assert not transport.monarch_rdma_transport_available()
    assert not xccl.xccl_available()
    assert not transport.xccl_available()
    assert not transport.torchcomms_uniflow_available()
    assert not transport.torchcomms_rdma_available()


def test_non_xpu_build_keeps_monarch_rdma_availability(monkeypatch):
    from torchtitan.experiments.ezpz.rl import xpu_overrides

    monkeypatch.setattr(
        xpu_overrides.torch, "version", SimpleNamespace(xpu=None), raising=False
    )
    monkeypatch.setitem(__import__("sys").modules, "torchstore", None)

    # Must return before importing or changing TorchStore on a non-XPU build.
    xpu_overrides.patch_torchstore_network_availability_for_xpu()


def test_xpu_flex_attention_uses_triton_backend():
    from torchtitan.experiments.ezpz.rl import xpu_overrides

    flex = object()
    triton_path = "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"

    class FakeBackend:
        CUSTOM = object()
        FLEX_ATTENTION = flex
        TRITON_ATTN = SimpleNamespace(get_path=lambda: triton_path)

    class FakePlatform:
        @classmethod
        def get_attn_backend_cls(cls, selected_backend, *_args, **_kwargs):
            return f"original:{selected_backend!r}"

    assert (
        xpu_overrides._select_vllm_xpu_attention_backend(
            flex, FakeBackend, lambda *_args: "original"
        )
        == triton_path
    )
