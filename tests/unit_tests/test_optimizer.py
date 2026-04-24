# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn

from torchtitan.components.optimizer import MuonOptimizersContainer


class ToyMuonModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(16, 8)
        self.hidden = nn.Linear(8, 8, bias=False)
        self.router = nn.Module()
        self.router.gate = nn.Linear(8, 4, bias=False)
        self.output = nn.Linear(8, 8, bias=False)
        self.scale = nn.Parameter(torch.ones(8))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.tok_embeddings(tokens).sum(dim=1)
        x = self.hidden(x)
        x = x + self.router.gate(x).sum(dim=-1, keepdim=True)
        x = x * self.scale
        return self.output(x).sum()


class TestMuonOptimizersContainer:
    def test_muon_partitioning(self):
        model = ToyMuonModel()
        optimizers = MuonOptimizersContainer.Config().build(model_parts=[model])
        optimizer = optimizers.optimizers[0]

        name_by_param_id = {id(param): name for name, param in model.named_parameters()}
        muon_names = {
            name_by_param_id[id(param)]
            for group in optimizer.param_groups
            if group["use_muon"]
            for param in group["params"]
        }
        adam_names = {
            name_by_param_id[id(param)]
            for group in optimizer.param_groups
            if not group["use_muon"]
            for param in group["params"]
        }

        assert muon_names == {"hidden.weight"}
        assert adam_names == {
            "tok_embeddings.weight",
            "router.gate.weight",
            "output.weight",
            "scale",
        }

    def test_muon_step_updates_both_optimizer_paths(self):
        torch.manual_seed(0)
        model = ToyMuonModel()
        optimizers = MuonOptimizersContainer.Config().build(model_parts=[model])

        tokens = torch.randint(0, 16, (4, 3))
        hidden_before = model.hidden.weight.detach().clone()
        output_before = model.output.weight.detach().clone()

        model(tokens).backward()
        optimizers.step()

        assert not torch.equal(model.hidden.weight, hidden_before)
        assert not torch.equal(model.output.weight, output_before)
