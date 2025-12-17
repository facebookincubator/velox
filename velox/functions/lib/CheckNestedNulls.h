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

#include "velox/vector/DecodedVector.h"

namespace facebook::velox::functions {

/// Checks nested nulls in a complex type vector.
///
/// @param decoded The decoded vector to check
/// @param index The index in the decoded vector to check
/// @param baseIndex The index in the base vector to check for nested nulls
/// @param throwOnNestedNulls If true, throws exception when base vector
/// contains nulls
/// @return true if the value at specified index is null, false otherwise
bool checkNestedNulls(
    const DecodedVector& decoded,
    vector_size_t index,
    vector_size_t baseIndex,
    bool throwOnNestedNulls = false);
} // namespace facebook::velox::functions
