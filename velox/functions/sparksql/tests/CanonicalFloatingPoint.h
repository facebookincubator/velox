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

#include <bit>

#include "velox/vector/BaseVector.h"

namespace facebook::velox::functions::sparksql::test {

/// A NaN that is not the canonical NaN.
inline const double kNonCanonicalNaN =
    std::bit_cast<double>(0x7ff8000000000001ULL);

/// Checks that all REAL and DOUBLE values of 'vector', including the ones
/// nested in arrays, maps and rows, are in the canonical form of
/// normalizeFloatingPoint(). Use it with assertEqualVectors, which treats -0.0
/// and 0.0 as equal.
void expectCanonicalFloatingPoint(const BaseVector& vector);

} // namespace facebook::velox::functions::sparksql::test
