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

#include <span>

#include "velox/vector/ComplexVector.h"

namespace facebook::velox {

/// Configuration for mapConcat().
struct MapConcatConfig {
  /// Treat null map inputs as empty maps instead of propagating null to the
  /// output row.
  bool emptyForNull{false};

  /// Throw an exception when duplicate keys are found across inputs.
  bool throwOnDuplicateKeys{false};
};

/// Merge multiple map vectors into a single MapVector.  All inputs are provided
/// as DecodedVectors.  For each selected row, entries from all inputs are
/// merged.  When duplicate keys exist across inputs, the entry from the last
/// input wins.  Only rows selected in 'rows' are processed; unselected rows
/// get size 0 in the output.
MapVectorPtr mapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config);

} // namespace facebook::velox
