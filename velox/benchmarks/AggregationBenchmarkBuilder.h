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

#include "velox/exec/Aggregate.h"
#include "velox/exec/Task.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox {

enum PlanType { kSingle, kPartialFinal, kPartialIntermediateFinal };

struct AggregationPair {
  std::vector<std::string> groupingKeys;
  std::vector<std::string> aggregations;
};

struct AggregationBenchmarkSet {
  explicit AggregationBenchmarkSet(const TypePtr& inputType)
      : inputType_{inputType} {}

  AggregationBenchmarkSet& addAggregations(
      const std::string& name,
      const std::vector<std::string>& groupingKeys,
      const std::vector<std::string>& aggregations) {
    aggregations_.push_back({name, {groupingKeys, aggregations}});
    return *this;
  }

  AggregationBenchmarkSet& withFuzzerOptions(
      const VectorFuzzer::Options& options) {
    fuzzerOptions_ = options;
    return *this;
  }

  AggregationBenchmarkSet& withIterations(int iterations) {
    iterations_ = iterations;
    return *this;
  }

  AggregationBenchmarkSet& withPlanType(PlanType planType) {
    planType_ = planType;
    return *this;
  }

  std::vector<std::pair<std::string, AggregationPair>> aggregations_;

  // The input that will be used for for benchmarking aggregations. If not set,
  // a flat input vector is fuzzed using fuzzerOptions_.
  std::vector<RowVectorPtr> inputRowVectors_;

  // The type of the input that will be used for all the aggregations
  // benchmarked.
  TypePtr inputType_;

  // User can provide fuzzer options for the input row vector used for this
  // benchmark. Note that the fuzzer will be used to generate a flat input row
  // vector if inputRowVector_ is nullptr.
  VectorFuzzer::Options fuzzerOptions_{.vectorSize = 10000, .nullRatio = 0};

  // Number of times to run each benchmark.
  int iterations_ = 1000;

  PlanType planType_ = PlanType::kSingle;
};

class AggregationBenchmarkBuilder {
 public:
  explicit AggregationBenchmarkBuilder() {}

  // Register all the benchmarks, so that they would run when
  // folly::runBenchmarks() is called.
  void registerBenchmarks();

  AggregationBenchmarkSet& addBenchmarkSet(
      const std::string& name,
      const TypePtr& inputType) {
    VELOX_CHECK(!benchmarkSets_.count(name));
    benchmarkSets_.emplace(name, AggregationBenchmarkSet(inputType));
    return benchmarkSets_.at(name);
  }

 private:
  void ensureInputVectors();

  std::shared_ptr<memory::MemoryPool> pool_{memory::addDefaultLeafMemoryPool()};

  std::map<std::string, AggregationBenchmarkSet> benchmarkSets_;
};

} // namespace facebook::velox
