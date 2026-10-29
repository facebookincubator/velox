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

#include <string>
#include <vector>

#include "velox/expression/EvalCtx.h"
#include "velox/functions/sparksql/LastWin.h"
#include "velox/vector/ComplexVector.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/SelectivityVector.h"

namespace facebook::velox::functions::sparksql {

/// Registers Spark's map under the given name.
void registerMapFunction(const std::string& name);

/// Builds a map for the selected rows from per-row key and value arrays,
/// resolving repeated keys with LastWin. The result has the given size and no
/// nulls.
std::shared_ptr<MapVector> buildDeduplicatedMap(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    vector_size_t size,
    const TypePtr& outputType,
    const DecodedVector& decodedKeys,
    const DecodedVector& decodedValues,
    const DecodedVector& decodedKeyElements,
    const VectorPtr& keyElements,
    const VectorPtr& valueElements,
    bool throwOnDuplicateKey);

/// Throws on a null key or a repeated key. Used by the zero-copy path, which
/// builds no entry indices.
void checkNullAndDuplicateKeys(
    const SelectivityVector& rows,
    exec::EvalCtx& context,
    const ArrayVectorBase& keysArray,
    const DecodedVector& decodedKeyElements);

} // namespace facebook::velox::functions::sparksql
