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

#include <cmath>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

#include "velox/dwio/nimble/encodings/SubIntSplitSampler.h"
#include "velox/dwio/nimble/encodings/SubIntSplitSelector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitTopLevelPolicy.h"
#include "velox/dwio/nimble/encodings/selection/BitFlipProfile.h"

// Standalone SubIntSplit cost estimator for benchmarking/testing the
// bit-flip-probability top-level policies (see SubIntSplitTopLevelPolicy.h
// for scope) against real cost. Deliberately named differently from the
// `estimateSize()` convention every production encoding follows (e.g.
// DictionaryEncoding<T>::estimateSize): it exists only so tests and
// benchmarks/ml_id_compression/MlIdSelectionPolicyBenchmark.cpp can compute
// "what would SubIntSplit's real cost and the policies' gate decision have
// been" without touching production selection code.

namespace facebook::nimble::detail::subintsplit {

struct EstimatorResult {
  // True when the variance gate predicted "worth costing SubIntSplit" (or
  // the gate was bypassed via `applyGate = false`).
  bool gatePassed{false};
  // The DP's total estimated cost in bits, only meaningful when
  // `estimatedBytes` is set.
  double estimatedBits{0.0};
  // ceil(estimatedBits / 8), or nullopt when the gate predicted "skip" (no
  // DP was run) or the sample was empty.
  std::optional<uint64_t> estimatedBytes;
  // The bit-flip profile computed from `values`, always populated.
  BitFlipProfile profile;
};

// Computes SubIntSplit's real estimated size for `values`, gated by the
// variance policy (unless `applyGate` is false, e.g. to compute an
// unconstrained ground-truth cost). Always runs the DP's full, unconstrained
// grid search.
template <typename PhysicalType>
EstimatorResult estimateSubIntSplitSize(
    std::span<const PhysicalType> values,
    const TopLevelPolicyConfig& policyConfig,
    const SamplerConfig& samplerConfig = defaultSamplerConfig(),
    const SelectorConfig& selectorConfig = defaultSelectorConfig(),
    bool applyGate = true) {
  EstimatorResult result;
  result.profile = computeBitFlipProfile<PhysicalType>(values);
  result.gatePassed =
      !applyGate || bitFlipVarianceGate(result.profile, policyConfig);
  if (!result.gatePassed) {
    return result;
  }

  std::vector<uint64_t> samples;
  sampleIntoU64<PhysicalType>(values, samples, samplerConfig);
  if (samples.empty()) {
    return result;
  }

  const auto selectorResult = selectSplits(
      samples, result.profile.numBits, values.size(), selectorConfig);
  result.estimatedBits = selectorResult.totalCost;
  result.estimatedBytes =
      static_cast<uint64_t>(std::ceil(result.estimatedBits / 8.0));
  return result;
}

} // namespace facebook::nimble::detail::subintsplit
