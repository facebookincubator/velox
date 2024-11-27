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

namespace facebook::velox::cudf_velox {

CudfConversion::CudfConversion(
    int32_t operatorId,
    RowTypePtr outputType,
    exec::DriverCtx* driverCtx)
    : exec::Operator(
          driverCtx,
          outputType,
          operatorId,
          orderByNode->id(),
          "CudfConversion") {
}

void CudfConversion::addInput(RowVectorPtr input) {
  // Accumulate inputs
  if (input->size() > 0) {
    inputs_.push_back(std::move(input));
  }
}

void CudfConversion::noMoreInput() {
  exec::Operator::noMoreInput();
  NVTX3_FUNC_RANGE();

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
    std::cout << "CudfConversion table number of columns: " << tbl->num_columns()
              << std::endl;
    std::cout << "CudfConversion table number of rows: " << tbl->num_rows()
              << std::endl;
  }

  outputTable_ = std::move(tbl);
}

RowVectorPtr CudfConversion::getOutput() {
  if (finished_ || !noMoreInput_) {
    return nullptr;
  }

  cudf::get_default_stream().synchronize();
  RowVectorPtr output =
      with_arrow::to_velox_column(outputTable_->view(), pool(), "");
  finished_ = noMoreInput_;
  outputTable_.reset();
  return output;
}

void CudfConversion::close() {
  exec::Operator::close();
  // TODO: Release stored inputs if needed
  // TODO: Release cudf memory resources
  outputTable_.reset();
}
} // namespace facebook::velox::cudf_velox
