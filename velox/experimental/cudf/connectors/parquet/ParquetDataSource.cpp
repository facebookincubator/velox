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
#include "velox/experimental/cudf/connectors/parquet/ParquetDataSource.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"

#include <cudf/io/parquet.hpp>
#include <cudf/io/types.hpp>
#include <cudf/table/table.hpp>
#include <cudf/table/table_view.hpp>

#include "velox/experimental/cudf/exec/CudfTableScan.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/exec/VeloxCudfInterop.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <memory>

namespace facebook::velox::cudf_velox::connector::parquet {

std::optional<RowVectorPtr> ParquetDataSource::next(
    uint64_t /* size */,
    velox::ContinueFuture& /* future */) {
  VELOX_CHECK(split_ != nullptr, "No split to process. Call addSplit first.");
  VELOX_CHECK_NOT_NULL(splitReader_, "No split reader present");

  if (splitReader_->emptySplit()) {
    resetSplit();
    return nullptr;
  }

  // cudf parquet reader returns has_next() = true if no chunk has yet been
  // read.
  if (splitReader_->has_next()) {
    // Read a chunk of table.
    auto [table, metadata] = splitReader_->read_chunk();
    // Check if the chunk is empty
    const auto rowsScanned = table.num_rows();
    if (rowsScanned == 0) {
      return nullptr;
    }

    // update completedRows
    completedRows_ += table.num_rows();

    // TODO: Update completedBytes_
    // completedBytes_ += what?

    // Convert to velox RowVectorPtr and return
    return std::make_optional(to_velox_column(tbl->view(), pool_));

  } else {
    return nullptr;
  }
}

void ParquetDataSource::addSplit(std::shared_ptr<ConnectorSplit> split) {
  split_ = std::dynamic_pointer_cast<ParquetConnectorSplit>(split);

  VLOG(1) << "Adding split " << split_->toString();

  // Split reader already exists
  if (splitReader_) {
    splitReader_.reset();
  }

  splitReader_ = createSplitReader();
}

std::unique_ptr<cudf::io::chunked_parquet_reader>
ParquetDataSource::createSplitReader() {
  auto const source_info = split_.getSourceInfo();
  auto options = cudf::io::parquet_reader_options::builder(source_info)
                     /*.filter()*/
                     .build();

  return std::make_unique(
      parquetConfig.chunkReadLimit(), parquetConfig.passReadLimit(), options);
}

void ParquetDataSource::resetSplit() {
  // Simply reset the split and the reader
  split_.reset();
  splitReader_.reset();
}

} // namespace facebook::velox::cudf_velox::connector::parquet
