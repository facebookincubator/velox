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

#include <cstdint>

namespace torch::nativert {
class Graph;
} // namespace torch::nativert

namespace torch::wave {

/// Removes runtime data-validation assert nodes (aten._assert_async.msg and
/// aten._assert_scalar.default). Both are side-effect-only runtime no-ops whose
/// outputs are unused; the wave executor cannot run _assert_scalar as a
/// standalone (it aborts on the scalar IValue), so stripping is required, not
/// just an optimization. Mutates 'graph' in place.
void stripDataAsserts(nativert::Graph& graph);

/// Renames the inputs of scalar/symbolic ops (_operator.*, torch.sym_max/min/
/// float) to the positional names "a"/"b" that nativert's getSymInputs-based
/// kernels (SymIntOpKernel/SymBoolOpKernel/ScalarBinaryOpKernel) look up by
/// name. Some graphs (e.g. sigmoid-archive round-trips) carry such nodes whose
/// inputs are named otherwise, making inputs.at("a") throw. Idempotent for
/// already-correct nodes; excludes sym_size/sym_numel (self/dim naming).
/// Mutates 'graph' in place.
void normalizeSymOpArgNames(nativert::Graph& graph);

/// Rewrites ops that have no CUDA implementation to a CUDA-capable equivalent
/// so the graph can run on GPU. Currently rewrites fb.simple_1d_concat (its
/// CUDA registration is a throwing dummy) to aten.cat.default(dim=0). Mutates
/// 'graph'; returns the number of nodes rewritten.
int32_t rewriteGpuIncompatibleOps(nativert::Graph& graph);

/// Inserts aten._to_copy(device=cpu) before each cpuOnly-flagged tensor arg and
/// repoints just that edge, so a graph placed on GPU keeps those args on the
/// host (e.g. tensor_split indices). Mutates 'graph'; returns the number of
/// copies inserted.
int32_t insertCpuOnlyCopies(nativert::Graph& graph);

/// Retargets the sparsenn merge-and-dedup nodes
/// (fb.fused_datafm_merge_and_dedup_by_reference[_optimized]) to their fused
/// TorchWave "_tw" CUDA equivalents, but ONLY when those _tw ops are actually
/// registered in the dispatcher (registerTorchWaveMergeAndDedup was called,
/// i.e. the caller linked the torchwave_meta merge-and-dedup kernels). When
/// they are absent this is a no-op and the base op runs as a nativert
/// standalone, so the same load() path serves both the base engine and the
/// fused-ops build. Mutates 'graph'; returns the number of nodes rewritten.
int32_t rewriteMergeAndDedupToTw(nativert::Graph& graph);

} // namespace torch::wave
