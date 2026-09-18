/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include "velox/common/memory/Memory.h"

namespace facebook::nimble::fuzzer {

/// Controls one stripe-stats pruning fuzzer run.
struct StripeStatsPruningFuzzerOptions {
  /// Number of randomized files to write and read back.
  int32_t numIterations{32};

  /// Seed for the run. Every random choice derives from this, so a failing
  /// iteration is reproducible from the seed alone.
  uint32_t seed{0};
};

/// Totals across a run, reported so a caller can assert the fuzzer actually
/// exercised pruning rather than silently reading every stripe.
struct StripeStatsPruningFuzzerStats {
  /// Iterations completed.
  int64_t numIterations{0};

  /// Stripes the reader skipped on stats, summed over iterations.
  int64_t numStripesSkipped{0};

  /// Iterations whose filter admitted no row, where pruning is most likely.
  int64_t numEmptyResultIterations{0};

  /// Iterations whose key column carried nulls, which forbid skipping a stripe
  /// when the filter admits null.
  int64_t numNullableKeyIterations{0};
};

/// Fuzzes stripe-stats pruning in the selective Nimble reader.
///
/// Each iteration writes a multi-stripe file whose filtered key column is
/// banded so that stripe min/max separate and pruning can fire, alongside
/// payload columns of randomized type, then reads it back under a range filter
/// and compares the result against the predicate evaluated over every written
/// row. Any row the pruned read drops is a stripe the reader skipped wrongly.
///
/// The key column type, nullability, stripe layout and filter bounds all vary
/// per iteration; banding only constrains where values land, not which rows
/// the filter admits.
class StripeStatsPruningFuzzer {
 public:
  StripeStatsPruningFuzzer(
      StripeStatsPruningFuzzerOptions options,
      velox::memory::MemoryPool& rootPool)
      : options_{options}, rootPool_{rootPool} {}

  /// Runs every iteration. Throws on the first mismatch, naming the seed and
  /// iteration so the case can be replayed.
  StripeStatsPruningFuzzerStats run();

 private:
  const StripeStatsPruningFuzzerOptions options_;
  velox::memory::MemoryPool& rootPool_;
};

} // namespace facebook::nimble::fuzzer
