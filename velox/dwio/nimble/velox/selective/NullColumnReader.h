/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include "velox/dwio/nimble/velox/selective/NimbleData.h"

namespace facebook::nimble {

class NullColumnReader : public velox::dwio::common::SelectiveColumnReader {
 public:
  NullColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec)
      : SelectiveColumnReader(requestedType, fileType, params, scanSpec) {}

  void read(
      int64_t offset,
      const velox::RowSet& rows,
      const uint64_t* incomingNulls) final {
    auto numRows = rows.back() + 1;
    readNulls(offset, numRows, incomingNulls);
    inputRows_ = rows;
    if (scanSpec_->filter()) {
      if (!scanSpec_->filter()->testNull()) {
        setOutputRows({});
      } else {
        setOutputRows(rows);
      }
    }
    readOffset_ += numRows;
  }

  void getValues(const velox::RowSet& rows, velox::VectorPtr* result) final {
    if (scanSpec_->isFlatMapAsStruct()) {
      NIMBLE_CHECK(result->get() && result->get()->type()->isRow());
      *result = velox::BaseVector::createNullConstant(
          result->get()->type(), rows.size(), pool_);
    } else {
      *result = velox::BaseVector::createNullConstant(
          requestedType_, rows.size(), pool_);
    }
  }

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const final {
    byteSize = rowCount = 0;
    return true;
  }
};

} // namespace facebook::nimble
