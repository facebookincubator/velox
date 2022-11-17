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

#include "velox/dwio/parquet/reader/ParquetNestedColumnReader.h"

namespace facebook::velox::parquet {

class ListColumnReader : public ParquetRepeatedColumnReader {
 public:
  ListColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      ParquetParams& params,
      common::ScanSpec& scanSpec,
      common::ScanSpec& topLevelScanSpec);

  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input)
      override;

  void seekToRowGroup(uint32_t index) override;

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* FOLLY_NULLABLE /*incomingNulls*/) override;

  std::vector<std::shared_ptr<NestedData>> read(uint64_t offset, RowSet rows)
      override;

  void getValues(RowSet rows, VectorPtr* FOLLY_NONNULL result) override;

 private:
  std::unique_ptr<ParquetNestedColumnReader> childColumnReader_;

  BufferPtr offsets_;
  BufferPtr lengths_;
  BufferPtr nulls_;
};
} // namespace facebook::velox::parquet