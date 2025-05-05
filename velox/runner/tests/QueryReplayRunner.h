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

#include "velox/runner/LocalRunner.h"

namespace facebook::velox::runner {

typedef std::string (*TaskPrefixExtractor)(const std::string&);

class QueryReplayRunner {
 public:
  /// Create a QueryReplayRunner with the given memory pool.
  /// @param taskPrefixExtractor A function that extracts the task prefix from a
  /// task id contained in the serialized plan fragment.
  /// @param width The number of workers for each stage except the gathering
  /// stages.
  /// @param maxDrivers The maximum number of drivers for each worker.
  explicit QueryReplayRunner(
      memory::MemoryPool* pool,
      TaskPrefixExtractor taskPrefixExtractor,
      int32_t width = kDefaultWidth,
      int32_t maxDrivers = kDefaultMaxDrivers);

  /// Runs a query with the given serialized plan fragments and returns the
  /// results. The serialized plan fragments should have the same query id as
  /// 'queryId'.
  std::vector<VectorPtr> run(
      const std::string& queryId,
      const std::vector<std::string>& serializedPlanFragments);

 private:
  std::shared_ptr<core::QueryCtx> makeQueryCtx(
      const std::string& queryId,
      std::shared_ptr<memory::MemoryPool> rootPool);

  std::shared_ptr<runner::SplitSourceFactory> makeSplitSourceFactory(
      const runner::MultiFragmentPlanPtr& plan);

  MultiFragmentPlanPtr deserializePlan(
      const std::string& queryId,
      const std::vector<std::string>& serializedPlanFragments);

  static constexpr int kWaitTimeoutUs = 5000'000;
  static const std::string kHiveConnectorId;
  static const int32_t kDefaultWidth = 2;
  static const int32_t kDefaultMaxDrivers = 4;

  inline static std::unordered_map<std::string, std::string> config_;
  inline static std::unordered_map<std::string, std::string> hiveConfig_;
  inline static std::unique_ptr<folly::CPUThreadPoolExecutor> executor_;

  memory::MemoryPool* pool_{nullptr};
  TaskPrefixExtractor taskPrefixExtractor_;
  int32_t width_;
  int32_t maxDrivers_;
};

} // namespace facebook::velox::runner
