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

#include "velox/experimental/cudf/CudfNoDefaults.h"
#include "velox/experimental/cudf/exec/CudfEnforceSingleRow.h"
#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"

#include <cudf/table/table.hpp>

namespace facebook::velox::cudf_velox {

CudfEnforceSingleRow::CudfEnforceSingleRow(
    int32_t operatorId,
    exec::DriverCtx* driverCtx,
    const std::shared_ptr<const core::EnforceSingleRowNode>& planNode)
    : exec::Operator(
          driverCtx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "CudfEnforceSingleRow"),
      CudfOperator(operatorId, planNode->id()) {}

bool CudfEnforceSingleRow::needsInput() const {
  return true;
}

void CudfEnforceSingleRow::addInput(RowVectorPtr input) {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  // Cast to CudfVector to access GPU data
  auto cudfInput = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(
      cudfInput, "CudfEnforceSingleRow expects CudfVector input");

  auto numInput = cudfInput->size();

  VELOX_CHECK_NE(
      numInput, 0, "CudfEnforceSingleRow::addInput received empty set of rows");

  if (input_ == nullptr) {
    VELOX_USER_CHECK_EQ(
        numInput,
        1,
        "Expected single row of input. Received {} rows.",
        numInput);
    input_ = std::move(cudfInput);
  } else {
    VELOX_USER_FAIL(
        "Expected single row of input. Received {} extra rows.", numInput);
  }
}

void CudfEnforceSingleRow::noMoreInput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (!noMoreInput_ && input_ == nullptr) {
    // We have not seen any data. Return a single row of all nulls.
    auto nullRow = BaseVector::create<RowVector>(outputType_, 1, pool());
    for (auto& child : nullRow->children()) {
      child->resize(1);
      child->setNull(0, true);
    }

    // Convert to CudfVector
    auto stream = cudf::get_default_stream(cudf::allow_default_stream);
    auto cudfTable =
        with_arrow::toCudfTable(nullRow, pool(), stream, get_output_mr());
    stream.synchronize();
    input_ = std::make_shared<CudfVector>(
        pool(), outputType_, 1, std::move(cudfTable), stream);
  }

  exec::Operator::noMoreInput();
}

RowVectorPtr CudfEnforceSingleRow::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();

  if (!noMoreInput_) {
    return nullptr;
  }

  return input_;
}

bool CudfEnforceSingleRow::isFinished() {
  return noMoreInput_ && input_ == nullptr;
}

} // namespace facebook::velox::cudf_velox
