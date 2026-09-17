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

/// Returns true when every input is flat map encoded. Returns false for no
/// inputs.
bool allInputsAreFlatMap(std::span<DecodedVector* const> inputs);

/// Merges vectors into a single FlatMapVector.
/// Inputs:
///   - Expect all flatmap encoded Velox vector. Throw otherwises.
/// Duplicated key handling:
///   - Currently throw if there is any key duplication.

FlatMapVectorPtr flatMapConcat(
    memory::MemoryPool* pool,
    const TypePtr& outputType,
    std::span<DecodedVector* const> inputs,
    const SelectivityVector& rows,
    const MapConcatConfig& config);

} // namespace facebook::velox
