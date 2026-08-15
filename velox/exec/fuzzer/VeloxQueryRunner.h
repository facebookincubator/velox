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

#include <folly/io/async/EventBaseThread.h>
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::exec::test {

class VeloxQueryRunner : public ReferenceQueryRunner {
 public:
  /// @param serviceUri Thrift URI of the LocalRunnerService.
  /// @param timeout Timeout in milliseconds of a request.
  VeloxQueryRunner(
      memory::MemoryPool* aggregatePool,
      std::string serviceUri,
      std::chrono::milliseconds timeout);

  RunnerType runnerType() const override {
    return RunnerType::kVeloxQueryRunner;
  }

  const std::vector<TypePtr>& supportedScalarTypes() const override;

  const std::unordered_map<std::string, DataSpec>&
  aggregationFunctionDataSpecs() const override;

  std::optional<std::string> toSql(const core::PlanNodePtr& plan) override;

  bool isConstantExprSupported(const core::TypedExprPtr& expr) override;

  bool isSupported(const exec::FunctionSignature& signature) override;

  std::pair<
      std::optional<std::multiset<std::vector<velox::variant>>>,
      ReferenceQueryErrorCode>
  execute(const core::PlanNodePtr& plan) override;

  bool supportsVeloxVectorResults() const override {
    return true;
  }

  std::vector<RowVectorPtr> execute(const std::string& sql) override;

  std::vector<RowVectorPtr> execute(
      const std::string& sql,
      const std::string& sessionProperty) override;

 private:
  // Serializes the plan node to JSON string
  std::string serializePlan(const core::PlanNodePtr& plan);

  std::string serviceUri_;
  std::chrono::milliseconds timeout_;
  folly::EventBase eventBase_;
  std::shared_ptr<memory::MemoryPool> pool_;

  // Thrift-specific members
  std::string thriftHost_;
  int thriftPort_{9091};
};

} // namespace facebook::velox::exec::test
