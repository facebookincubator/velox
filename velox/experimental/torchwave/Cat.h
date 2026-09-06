/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <ATen/ATen.h>

#include "velox/experimental/torchwave/KernelOperation.h"
#include "velox/experimental/torchwave/Registry.h"

namespace torch::wave {

/// Registers the setOutputs-based metadata for aten.cat.default and
/// aten.stack.default, replacing the standalone-only registrations in
/// Builtins.cpp. Both ops share one fused implementation: a stack is a cat that
/// gives each operand a single position along a new dimension.
void registerConcatMetadata();

/// True if 'node' is an aten.cat / aten.stack that will run fused with a result
/// of rank > 1. Such a concat allocates its output on the host and hands each
/// operand a (generally strided) view of the region it writes, so no operand's
/// extent may be computed on device inside the concat's own kernel.
bool concatNeedsHostShapes(NodeCP node, const ValueTypes& types);

/// True if this concat should have each operand filled by a kernel op of its
/// own in the previous step, rather than by a chain of copies inside the
/// concat's own kernel. Gated on WaveConfig::parallelConcatFill.
bool concatFillsInParallel(NodeCP node, const ValueTypes& types);

/// The axis a cat/stack joins its operands on and the rank of its result. 'dim'
/// is normalized to a non-negative index in the RESULT's coordinates, which for
/// a stack is one wider than the operands'.
struct ConcatSpec {
  bool isStack{false};
  int32_t dim{0};
  int8_t outRank{-1};

  int8_t elementRank() const {
    return isStack ? static_cast<int8_t>(outRank - 1) : outRank;
  }
};

/// One operand of a fused concat, as the concat's own kernel sees it: where the
/// operand's extent comes from, and whether the concat computes it, copies it,
/// or aliases it.
struct ConcatInputInfo {
  /// The operand's value: formal, in the KernelOperation's own subgraph, on the
  /// ConcatLayout the concat carries; already translated to the frame's on the
  /// copies the allocation-group pass builds. 'sizeExpr' follows it.
  nativert::ValueId valueId{-1};
  SizeExpr sizeExpr;
  OutputReserveFunc reserveShape;
  /// The operand's extent is settled by device code, so the host cannot lay it
  /// out ahead of the launch that produces it.
  bool hasShapeOnDevice{false};
  /// A producer inside the concat's own kernel sizes its output with a reserve
  /// function. The extent is the host's to compute, but only once that
  /// producer's launch is sized, which is after the allocation-group pass
  /// carves the result -- so the operand cannot be measured in time. An operand
  /// whose own descriptor carries a reserve function is caught by
  /// 'reserveShape'; this is the case where an elementwise op stands between
  /// the two and lends the operand a size expression the producer never had.
  bool hasReserveInChain{false};
  /// The kernel that produces the operand writes through its output's strides,
  /// so it can be handed a pitched band of the result. Captured here because
  /// the allocation-group pass sees only value ids, not the producing nodes.
  /// See ArgumentMeta::mayWriteStrided.
  bool mayWriteStrided{false};
  /// The operand crosses the concat's kernel boundary: an earlier kernel, or
  /// the graph itself, produces it, and the concat only copies it in.
  bool isSubgraphInput{false};
  /// The operand is a view of somebody else's buffer, so it has no storage of
  /// its own to place.
  bool isView{false};
};

/// The shape of a concat result whose operands have 'operandShapes', in join
/// order and each of the operand rank.
std::vector<Dim> concatResultShape(
    const ConcatSpec& spec,
    const std::vector<std::vector<Dim>>& operandShapes);

/// The view of 'result' occupied by a concat operand that spans 'extent'
/// positions from 'start' along the join axis. A stack operand occupies the
/// single position 'start' and drops the axis.
///
/// aliasTensor rather than narrow/select: this runs per operand per launch,
/// where the dispatch would dominate. The geometry is the caller's own -- the
/// result was sized as exactly these extents -- so there is nothing left to
/// bounds-check.
at::Tensor concatOperandView(
    const at::Tensor& result,
    const ConcatSpec& spec,
    int64_t start,
    int64_t extent);

/// Everything the host needs to lay a fused concat's result out: the operands
/// in the order the result joins them, and the geometry of the join. Carried on
/// the result's OutputDesc so the allocation-group pass can recognize the
/// concat and decide which operands can write straight into the result.
///
/// 'spec' and 'dtype' are the formal node's. Subgraph deduplication matches on
/// structure, dtype and the 'dim' attribute but not on rank, so one
/// KernelOperation serves concats that differ in both -- resolve() is what
/// gives the ones the invocation at hand actually has.
struct ConcatLayout {
  ConcatSpec spec;
  c10::ScalarType dtype{c10::ScalarType::Float};
  std::vector<ConcatInputInfo> inputs;
  nativert::ValueId outputFormalId{-1};

  /// Formal node of the concat, as the key into an invocation's node map.
  NodeCP originalNode{nullptr};
  const ValueTypes* types{nullptr};

  /// The join geometry and result dtype of the node 'nodeMap' binds
  /// 'originalNode' to, which is the formal node's own when the invocation is
  /// the formal subgraph.
  std::pair<ConcatSpec, c10::ScalarType> resolve(const NodeMap& nodeMap) const;
};

} // namespace torch::wave
