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

#include <memory>
#include <sstream>
#include <string>

#include <folly/init/Init.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>

#include "axiom/runner/LocalRunner.h"
#include "velox/expression/EvalCtx.h"
#include "velox/functions/facebook/prestosql/Register.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/runner/if/gen-cpp2/LocalRunnerService.h"

namespace facebook::velox::runner {

ScalarValue getScalarValue(const VectorPtr& vector, vector_size_t rowIdx);

ComplexValue getComplexValue(
    const VectorPtr& vector,
    vector_size_t rowIdx,
    exec::EvalCtx evalCtx);

Value convertValue(
    const VectorPtr& vector,
    vector_size_t rowIdx,
    const exec::EvalCtx& evalCtx);

std::vector<Value> convertVector(
    const VectorPtr& vector,
    vector_size_t size,
    const exec::EvalCtx& evalCtx);

std::vector<Batch> convertToBatches(
    const std::vector<RowVectorPtr>& rowVectors,
    const exec::EvalCtx& evalCtx);

class LocalRunnerServiceHandler
    : public apache::thrift::ServiceHandler<LocalRunnerService> {
 public:
  void execute(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override;
};

} // namespace facebook::velox::runner
