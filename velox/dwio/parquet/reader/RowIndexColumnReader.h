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

#include "velox/dwio/common/SelectiveIntegerColumnReader.h"

namespace facebook::velox::parquet {

class RowIndexColumnReader : public dwio::common::SelectiveIntegerColumnReader {
 public:
  RowIndexColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveIntegerColumnReader(
            requestedType->type(),
            params,
            scanSpec,
            requestedType) {}

  bool hasBulkPath() const override {
    return true;
  }

  void seekToRowGroup(uint32_t index) override {
    scanState().clear();
    readOffset_ = 0;
  }

  uint64_t skip(uint64_t numValues) override {
    return numValues;
  }

  void getValues(RowSet rows, VectorPtr* result) override {
    auto vector = BaseVector::create<FlatVector<int64_t>>(
        CppToType<int64_t>::create(), rows.size(), &memoryPool_);

    for (auto i = 0; i < rows.size(); ++i) {
      auto curRowIndex = rowIndexOffset_ + offset_ + rows[i];
      vector->set(i, curRowIndex);
      lastReadRowIndex_ = curRowIndex;
    }
    *result = vector;
  }

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* /*incomingNulls*/) override {
    VELOX_WIDTH_DISPATCH(
        parquetSizeOfIntKind(TypeKind::BIGINT),
        prepareRead,
        offset,
        rows,
        nullptr);
    offset_ = offset;
    if (offset == 0) {
      // new row group
      rowIndexOffset_ = lastReadRowIndex_ + 1;
    }
    readOffset_ += rows.back() + 1;
  }

  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor) {
    // do nothing.
  }

 private:
  vector_size_t offset_ = 0;
  vector_size_t lastReadRowIndex_ = -1;
  // representing the row index offset of this row group
  vector_size_t rowIndexOffset_ = 0;
};
} // namespace facebook::velox::parquet
