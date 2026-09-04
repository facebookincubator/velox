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

#include "velox/vector/FlatMapVector.h"
#include "velox/vector/MapConcat.h"

namespace facebook::velox {

/// Returns true when every input is flat map encoded, and false when none is.
/// Throws when only some are; mapConcat() and flatMapConcat() each require a
/// single encoding across all inputs.
bool allInputsAreFlatMap(std::span<DecodedVector* const> inputs);

/// Merges flat map vectors into a single FlatMapVector.  All inputs are
/// provided as DecodedVectors and must be flat map encoded; see
/// allInputsAreFlatMap().  Semantics otherwise match mapConcat(): only rows
/// selected in 'rows' are processed, and unselected rows get size 0.
///
/// Every output key must come from exactly one input, so that the output can
/// share the inputs' value vectors instead of copying them.  Inputs that share
/// a key, and encodings wrapped over a flat map input, are therefore not
/// supported and throw.
FlatMapVectorPtr flatMapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config);

} // namespace facebook::velox
