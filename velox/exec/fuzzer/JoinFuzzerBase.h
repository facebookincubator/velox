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

#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/exec/tests/utils/QueryAssertions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::exec {

/// Helper for formatting percentages in stats output.
std::string makePercentageString(size_t value, size_t total);

/// Abstract base class for join fuzzers that provides common functionality
/// for generating random join inputs and verifying results against a reference
/// database.
///
/// Subclasses must implement:
/// - isTargetSupported(): Returns whether the target executor supports a join
/// type
/// - getSupportedTypes(): Returns the data types to use for columns
/// - verify(): Performs the actual verification for a given join type
class JoinFuzzerBase {
 public:
  JoinFuzzerBase(
      size_t initialSeed,
      std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner,
      const std::string& poolName);

  virtual ~JoinFuzzerBase() = default;

  /// Main entry point - runs the fuzzer loop.
  void go();

 protected:
  // === Customization points (must override in subclasses) ===

  /// Returns whether the target executor (e.g., CPU Velox, cuDF) supports
  /// the given join type. Subclasses should override this to reflect their
  /// executor's capabilities.
  virtual bool isTargetSupported(core::JoinType joinType) const = 0;

  /// Returns the data types that this fuzzer should use for key and payload
  /// columns.
  virtual std::vector<TypePtr> getSupportedTypes() const = 0;

  /// Runs one test iteration for the given join type.
  virtual void verify(core::JoinType joinType) = 0;

  // === Shared utilities ===

  /// Returns the join types that this fuzzer should test. This is the
  /// intersection of join types supported by the target executor
  /// (isTargetSupported) and the reference database (isReferenceSupported).
  std::vector<core::JoinType> getSupportedJoinTypes() const;

  /// Returns whether the reference query runner (e.g., DuckDB) supports
  /// the given join type for result verification.
  static bool isReferenceSupported(core::JoinType joinType);

  static VectorFuzzer::Options getFuzzerOptions();

  void seed(size_t seed);
  void reSeed();

  int32_t randInt(int32_t min, int32_t max);

  /// Picks a random join type from getSupportedJoinTypes().
  core::JoinType pickJoinType();

  /// Returns a list of randomly generated join key types.
  std::vector<TypePtr> generateJoinKeyTypes(int32_t numKeys);

  /// Maximum nesting depth for payload column types. Override to allow
  /// nested types (arrays, maps, structs) in payload columns.
  /// Default is 0 (scalar types only).
  virtual int getPayloadMaxDepth() const {
    return 0;
  }

  /// Returns randomly generated probe input with up to 3 additional payload
  /// columns.
  std::vector<RowVectorPtr> generateProbeInput(
      const std::vector<std::string>& keyNames,
      const std::vector<TypePtr>& keyTypes);

  /// Same as generateProbeInput() but copies over 10% of the input in the probe
  /// columns to ensure some matches during joining. Also generates an empty
  /// input with a 10% chance.
  std::vector<RowVectorPtr> generateBuildInput(
      const std::vector<RowVectorPtr>& probeInput,
      const std::vector<std::string>& probeKeys,
      const std::vector<std::string>& buildKeys);

  /// Computes reference results using the reference query runner.
  std::optional<test::MaterializedRowMultiset> computeReferenceResults(
      const core::PlanNodePtr& plan,
      const std::vector<RowVectorPtr>& probeInput,
      const std::vector<RowVectorPtr>& buildInput);

  // === Shared members ===

  FuzzerGenerator rng_;
  size_t currentSeed_{0};

  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> pool_;

  VectorFuzzer vectorFuzzer_;
  std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner_;

  struct Stats {
    size_t numIterations{0};
    size_t numVerified{0};

    virtual ~Stats() = default;
    virtual std::string toString() const;
  };

  /// Creates the Stats object. Override to return a subclass with additional
  /// fields. Called once during construction.
  virtual std::unique_ptr<Stats> makeStats() {
    return std::make_unique<Stats>();
  }

  std::unique_ptr<Stats> stats_;
};

} // namespace facebook::velox::exec
