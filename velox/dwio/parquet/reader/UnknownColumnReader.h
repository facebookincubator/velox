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

#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetData.h"
#include "velox/vector/BaseVector.h"

namespace facebook::velox::parquet {

class UnknownColumnReader : public dwio::common::SelectiveColumnReader {
 public:
  UnknownColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveColumnReader(
            requestedType,
            std::move(fileType),
            params,
            scanSpec) {}

  void seekToRowGroup(int64_t index) override {
    SelectiveColumnReader::seekToRowGroup(index);
    scanState().clear();
    readOffset_ = 0;
    formatData_->as<ParquetData>().seekToRowGroup(index);
  }

  uint64_t skip(uint64_t numValues) override {
    // Unknown columns are not read from parquet, so just skip the values.
    return numValues;
  }

  void read(
      int64_t /*offset*/,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    if (rows.empty()) {
      return;
    }

    inputRows_ = rows;
    outputRows_.clear();
    mayGetValues_ = true;
    numValues_ = 0;
    anyNulls_ = true;
    allNull_ = true;

    auto* filter = scanSpec_->filter();
    if (filter) {
      if (filter->testNull()) {
        setOutputRows(rows);
      }
    } else if (useOutputRows()) {
      setOutputRows(rows);
    }

    if (auto* hook = scanSpec_->valueHook()) {
      if (hook->acceptsNulls()) {
        for (vector_size_t i = 0; i < outputRows().size(); ++i) {
          hook->addNull(i);
        }
      }
    }

    readOffset_ += rows.back() + 1;
  }

  void getValues(const RowSet& rows, VectorPtr* result) override {
    *result =
        BaseVector::createNullConstant(requestedType_, rows.size(), pool_);
  }
};

} // namespace facebook::velox::parquet
