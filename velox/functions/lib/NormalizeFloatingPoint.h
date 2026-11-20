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

#include <cmath>
#include <limits>
#include <type_traits>

#include "velox/vector/BaseVector.h"

namespace facebook::velox::functions {

/// Returns 'value' in canonical form: -0.0 becomes 0.0 and every NaN becomes
/// the canonical quiet NaN. Other values are returned as is.
template <typename T>
FOLLY_ALWAYS_INLINE T normalizeFloatingPoint(T value) {
  static_assert(std::is_floating_point_v<T>);
  if (FOLLY_UNLIKELY(std::isnan(value))) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  // -0.0 == 0.0, so this turns -0.0 into 0.0.
  if (value == 0) {
    return 0;
  }
  return value;
}

/// Returns true if 'type' is REAL or DOUBLE, or a complex type that contains
/// REAL or DOUBLE.
bool containsFloatingPoint(const Type& type);

/// Returns 'vector' with all REAL and DOUBLE values, including the ones nested
/// in arrays, maps and rows, in the canonical form of normalizeFloatingPoint().
/// Returns 'vector' itself when no value needs to change.
VectorPtr normalizeFloatingPoint(
    const VectorPtr& vector,
    memory::MemoryPool* pool);

} // namespace facebook::velox::functions
