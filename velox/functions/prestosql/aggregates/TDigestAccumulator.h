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

#include "velox/common/memory/HashStringAllocator.h"
#include "velox/functions/lib/TDigest.h"

namespace facebook::velox::aggregate::prestosql {

/// Accumulator holding a TDigest sketch and an optional compression parameter.
/// Shared by TDigestAggregate and ApproxWinsorizedMeanAggregate.
struct TDigestAccumulator {
  explicit TDigestAccumulator(HashStringAllocator* allocator)
      : digest(StlAllocator<double>(allocator)) {}

  /// Compression factor controlling TDigest accuracy vs memory trade-off.
  double compression{0.0};
  /// TDigest sketch accumulating input values for quantile estimation.
  facebook::velox::functions::TDigest<StlAllocator<double>> digest;
};

} // namespace facebook::velox::aggregate::prestosql
