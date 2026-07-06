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

#include <cstddef>
#include <functional>
#include <unordered_set>
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"

namespace facebook::velox::exec {

/// Plan-level features that can be enabled or disabled per executor.
/// CPU enables all; other executors (e.g. cuDF) may pass a subset.
enum class PlanFeature {
  kSpillInjection,
  kOomInjection,
  kTableScan,
  kGroupedExecution,
  kFlatInputVariants,
  kMultiDriverExecution,
};

using PlanFeatureSet = std::unordered_set<PlanFeature>;

/// Join algorithm variants that an executor may or may not implement.
/// These gate executor-level support independently of Velox plan-node validity
/// checks (e.g. MergeJoinNode::isSupported).
enum class JoinAlgorithm {
  kHash,
  kMerge,
  kNestedLoop,
};

/// Capabilities struct that parameterizes JoinFuzzer for a specific executor.
///
/// Each callback returns true if the executor supports the given combination.
/// CPU capabilities enable everything, preserving existing behavior exactly.
struct JoinFuzzerCapabilities {
  /// Whether the executor supports the given join type.
  std::function<bool(core::JoinType)> supportsJoinType;

  /// Whether the executor supports the given algorithm for the given join type.
  /// This is checked in addition to Velox plan-node isSupported() checks.
  std::function<bool(JoinAlgorithm, core::JoinType)> supportsJoinAlgorithm;

  /// Whether the executor supports filters for the given join type.
  std::function<bool(core::JoinType)> supportsFilter;

  /// Whether the executor supports the given filter expression for the given
  /// algorithm and join type. Called only when a filter has been generated.
  std::function<
      bool(JoinAlgorithm, core::JoinType, const std::string& /*filter*/)>
      supportsFilterExpression;

  /// Whether the executor supports null-aware semantics for the given join type
  /// and filter configuration.
  std::function<bool(core::JoinType, bool /*hasFilter*/)> supportsNullAware;

  /// The scalar types supported by the executor for join keys.
  /// If empty, falls back to referenceQueryRunner->supportedScalarTypes().
  std::vector<TypePtr> supportedTypes;

  /// Maximum nesting depth for randomly generated payload columns.
  /// CPU uses 2; executors with limited complex-type support may use 0.
  int payloadMaxDepth = 2;

  /// Set of plan-level features enabled for this executor.
  PlanFeatureSet planFeatures;

  /// Convenience method to check whether a plan feature is enabled.
  bool supports(PlanFeature f) const {
    return planFeatures.count(f) > 0;
  }
};

/// Returns JoinFuzzerCapabilities with everything enabled -- the CPU default.
/// This preserves the existing JoinFuzzer behavior exactly.
JoinFuzzerCapabilities cpuJoinCapabilities();

void joinFuzzer(
    size_t seed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner,
    const JoinFuzzerCapabilities& capabilities = cpuJoinCapabilities());
} // namespace facebook::velox::exec
