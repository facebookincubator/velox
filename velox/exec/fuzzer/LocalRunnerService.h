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

/// Thrift service implementation and library for executing Velox query plans
/// remotely.
///
/// This file provides conversion utilities and a service handler for the
/// LocalRunnerService. It enables remote execution of serialized Velox
/// expression evaluation primarily used for fuzzing where query plans need to
/// be executed on remote workers.

#pragma once

#include <folly/init/Init.h>
#include <folly/json.h>
#include <folly/portability/GFlags.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>

#include "velox/exec/fuzzer/if/gen-cpp2/LocalRunnerService.h"
#include "velox/expression/EvalCtx.h"

namespace facebook::velox::runner {

/// Extracts a scalar (primitive) value from a Velox vector at the specified
/// row. This function handles all primitive types supported by Velox, such as:
/// TINYINT, INTEGER, BIGINT, etc.
ScalarValue getScalarValue(const VectorPtr& vector, vector_size_t rowIdx);

/// Extracts a complex (nested) value from a Velox vector at the specified row.
/// This function handles all complex types supported by Velox: ARRAY, MAP and
/// ROW. The function recursively converts nested structures.
ComplexValue getComplexValue(
    const VectorPtr& vector,
    vector_size_t rowIdx,
    const exec::EvalCtx& evalCtx);

/// Converts a Velox vector value at a specific row to a Thrift Value and serves
/// as the entry point for value conversion that can be either primitive or
/// complex. Output value can either be a scalar or complex value (as mentioned,
/// using the above). This is where NULL is also defined.
Value convertValue(
    const VectorPtr& vector,
    vector_size_t rowIdx,
    const exec::EvalCtx& evalCtx);

/// Converts a Velox vector into a corresponding Thrift struct vector.
std::vector<Value> convertVector(
    const VectorPtr& vector,
    vector_size_t size,
    const exec::EvalCtx& evalCtx);

/// Converts a collection of Velox RowVectors into Thrift Batches.
std::vector<Batch> convertToBatches(
    const std::vector<RowVectorPtr>& rowVectors,
    const exec::EvalCtx& evalCtx);

/// Thrift service handler for executing Velox query plans.
/// Executes a serialized Velox query plan. This method deserializes the plan
/// from JSON, configures execution, runs the query plan to completion,
/// converts results to Thrift Batches and captures any subsequent errors or
/// output. The method returns a Thrift response containing the results.
class LocalRunnerServiceHandler
    : public apache::thrift::ServiceHandler<LocalRunnerService> {
 public:
  void execute(
      ExecutePlanResponse& response,
      std::unique_ptr<ExecutePlanRequest> request) override;
};

} // namespace facebook::velox::runner
