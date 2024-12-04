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
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/vector/ComplexVector.h"

#include <cudf/concatenate.hpp>
#include <cudf/table/table.hpp>
#include <cudf/utilities/default_stream.hpp>

#include <nvtx3/nvtx3.hpp>

#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_velox {

CudfFromVelox::CudfFromVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfFromVelox") {}

void CudfFromVelox::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    inputs_.push_back(std::move(input));
  }
}

void CudfFromVelox::noMoreInput() {
  exec::Operator::noMoreInput();
  NVTX3_FUNC_RANGE();

  if (inputs_.empty()) {
    outputTable_ = nullptr;
    return;
  }

  auto cudf_tables = std::vector<std::unique_ptr<cudf::table>>(inputs_.size());
  auto cudf_table_views = std::vector<cudf::table_view>(inputs_.size());
  for (int i = 0; i < inputs_.size(); i++) {
    VELOX_CHECK_NOT_NULL(inputs_[i]);
    cudf_tables[i] = with_arrow::to_cudf_table(inputs_[i], inputs_[i]->pool());
    cudf_table_views[i] = cudf_tables[i]->view();
  }
  auto tbl = cudf::concatenate(cudf_table_views);

  // Release input data
  cudf::get_default_stream().synchronize();
  cudf_table_views.clear();
  cudf_tables.clear();
  inputs_.clear();
  VELOX_CHECK_NOT_NULL(tbl);
  if (cudfDebugEnabled()) {
    std::cout << "CudfFromVelox table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "CudfFromVelox table number of rows: " << tbl->num_rows()
              << std::endl;
  }

  auto const size = tbl->num_rows();
  if (size == 0) {
    outputTable_ = nullptr;
    return;
  }
  outputTable_ =
      std::make_shared<CudfVector>(pool(), outputType_, size, std::move(tbl));
}

RowVectorPtr CudfFromVelox::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }
  finished_ = noMoreInput_;
  return outputTable_;
}

void CudfFromVelox::close() {
  cudf::get_default_stream().synchronize();
  outputTable_.reset();
  exec::Operator::close();
}

CudfToVelox::CudfToVelox(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx,
    std::string planNodeId)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          planNodeId,
          "CudfToVelox") {}

void CudfToVelox::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    auto cudf_input = std::dynamic_pointer_cast<CudfVector>(input);
    VELOX_CHECK_NOT_NULL(cudf_input);
    inputs_.push_back(std::move(cudf_input));
  }
}

void CudfToVelox::noMoreInput() {
  exec::Operator::noMoreInput();
}

RowVectorPtr CudfToVelox::getOutput() {
  if (finished_ || inputs_.empty()) {
    finished_ = noMoreInput_ && inputs_.empty();
    return nullptr;
  }

  NVTX3_FUNC_RANGE();

  std::unique_ptr<cudf::table> tbl = inputs_.front()->release();
  inputs_.pop_front();

  VELOX_CHECK_NOT_NULL(tbl);
  if (cudfDebugEnabled()) {
    std::cout << "CudfToVelox table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "CudfToVelox table number of rows: " << tbl->num_rows()
              << std::endl;
  }

  cudf::get_default_stream().synchronize();
  if (tbl->num_rows() == 0) {
    return nullptr;
  }
  RowVectorPtr output = with_arrow::to_velox_column(tbl->view(), pool(), "");
  finished_ = noMoreInput_ && inputs_.empty();
  return output;
}

void CudfToVelox::close() {
  exec::Operator::close();
  // TODO: Release stored inputs if needed
  // TODO: Release cudf memory resources
}

} // namespace facebook::velox::cudf_velox
