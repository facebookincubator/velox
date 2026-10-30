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
#include <optional>
#include <vector>

#include "velox/experimental/torchwave/WaveGraph.h"

namespace torch::wave {

/// Sizes of a tensor that the graph fixes, one per dim, -1 for a dim known
/// only at run time.
using StaticDims = std::vector<int64_t>;

/// Returns the dims of 'value' recorded in 'types': the static dims when they
/// were inferred, else -1 for every dim of a known rank. Nullopt when the rank
/// is unknown.
std::optional<StaticDims> staticDimsOf(const ValueTypes& types, ValueCP value);

/// True when 'value' has a known rank and every dim is a constant.
bool isFullyStatic(const ValueTypes& types, ValueCP value);

/// Infers the dims of each output of 'node' from the dims of its inputs in
/// 'types' and from its attributes. Returns one entry per output, nullopt
/// where the op has no rule or the inputs carry too little. Graph inputs,
/// weights and constants are seeded from their TensorMeta by the caller.
std::vector<std::optional<StaticDims>> inferStaticDims(
    NodeCP node,
    const ValueTypes& types);

} // namespace torch::wave
