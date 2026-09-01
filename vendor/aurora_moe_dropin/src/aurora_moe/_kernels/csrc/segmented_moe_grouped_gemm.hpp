// Copyright (c) 2026
// SPDX-License-Identifier: BSD-3-Clause
//
// Source/expert physical-layout adaptation of SYCL*TLA's PVC MoE example.
// The upstream scheduler and GEMM building blocks retain their own notices.

#pragma once

#include <cute/tensor.hpp>
#include <cute/util/compat.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/gemm/gemm.h>
#include <cutlass/gemm/group_array_problem_shape.hpp>
#include <cutlass/gemm/kernel/tile_scheduler.hpp>
#include <cutlass/kernel_hardware_info.hpp>
#include <cutlass/platform/platform.h>

#include "moe_gemms.hpp"
#include "moe_tile_scheduler.hpp"

namespace aurora_moe::segmented_tla {

using namespace cute;

using ProblemShape = cutlass::gemm::GroupProblemShape<Shape<int, int, int>>;
using TileScheduler = MoE::PersistentTileSchedulerXeMoE<ProblemShape>;

template <typename T, char LayoutKind>
CUTE_DEVICE auto make_tensor(T* ptr, int rows, int columns) {
  auto shape = make_shape(rows, columns);
  if constexpr (LayoutKind == 'C') {
    return make_tensor(make_gmem_ptr<T>(ptr),
                       make_layout(shape, make_stride(_1{}, rows)));
  } else {
    return make_tensor(make_gmem_ptr<T>(ptr),
                       make_layout(shape, make_stride(columns, _1{})));
  }
}

// The physical activation/output layout is [source0/expert0,
// source0/expert1, ..., source1/expert0, ...].  It is contiguous by fragment
// but not contiguous by logical expert.  ``group % local_experts`` maps each
// physical fragment to its reused expert weight without copying either rows or
// weights.  All counts and dimensions remain runtime values.
template <class GmemTiledCopyA, class GmemTiledCopyB, class GmemTiledCopyD,
          char LayoutKindA, char LayoutKindB, char LayoutKindD, class TiledMMA,
          typename ElementA, typename ElementB, typename ElementD>
CUTE_DEVICE void SegmentedMoEGEMM(
    const ElementA* activations, const ElementB* weights, ElementD* outputs,
    TiledMMA const& mma, const int32_t* rows_per_group,
    int32_t group_count, int32_t local_experts, int32_t n, int32_t k,
    typename TileScheduler::Params scheduler_params) {
  TileScheduler scheduler{scheduler_params, const_cast<int32_t*>(rows_per_group),
                          n, k, group_count};
  auto work_tile_info = scheduler.initial_work_tile_info(Shape<_1, _1, _1>{});

  constexpr char actual_layout_of_b = LayoutKindB ^ ('R' ^ 'C');
  bool did_group_change = true;
  int32_t current_group = 0;
  int32_t previous_group = 0;
  int32_t cumulative_rows = 0;
  int32_t rows = 0;

  if (work_tile_info.is_valid()) {
    current_group = work_tile_info.L_idx;
    rows = rows_per_group[current_group];
  }

  auto a_tensor = make_tensor<ElementA, LayoutKindA>(
      const_cast<ElementA*>(activations), rows, k);
  auto b_tensor = make_tensor<ElementB, actual_layout_of_b>(
      const_cast<ElementB*>(weights), n, k);
  auto d_tensor = make_tensor<ElementD, LayoutKindD>(outputs, rows, n);

  while (work_tile_info.is_valid()) {
    const auto m_coord = work_tile_info.M_idx;
    const auto n_coord = work_tile_info.N_idx;
    const auto tile_coord = make_coord(m_coord, n_coord, _, 0);

    if (did_group_change) {
      current_group = work_tile_info.L_idx;
      rows = rows_per_group[current_group];
      // Scheduler work can move between non-adjacent groups.  This loop is
      // at most sources*experts iterations and stays entirely on device.
      for (int32_t group = previous_group; group < current_group; ++group) {
        cumulative_rows += rows_per_group[group];
      }
      previous_group = current_group;
      const int32_t expert = current_group % local_experts;

      auto* a_ptr = const_cast<ElementA*>(activations) +
          static_cast<int64_t>(cumulative_rows) * k;
      auto* b_ptr = const_cast<ElementB*>(weights) +
          static_cast<int64_t>(expert) * k * n;
      auto* d_ptr = outputs + static_cast<int64_t>(cumulative_rows) * n;
      a_tensor = make_tensor<ElementA, LayoutKindA>(a_ptr, rows, k);
      b_tensor = make_tensor<ElementB, actual_layout_of_b>(b_ptr, n, k);
      d_tensor = make_tensor<ElementD, LayoutKindD>(d_ptr, rows, n);
      did_group_change = false;
    }

    MoE::moe_gemm<GmemTiledCopyA, GmemTiledCopyB, GmemTiledCopyD>(
        a_tensor, b_tensor, d_tensor, tile_coord, mma);
    work_tile_info = scheduler.fetch_next_work(work_tile_info);
    did_group_change = current_group != work_tile_info.L_idx;
  }
}

}  // namespace aurora_moe::segmented_tla
