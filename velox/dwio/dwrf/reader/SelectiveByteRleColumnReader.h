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

#include "velox/dwio/common/SelectiveByteRleColumnReader.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

class SelectiveByteRleColumnReader
    : public dwio::common::SelectiveByteRleColumnReader {
 public:
  using ValueType = int8_t;

  SelectiveByteRleColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      DwrfParams& params,
      common::ScanSpec& scanSpec,
      bool isBool);

  void getValues(const RowSet& rows, VectorPtr* result) override;

  bool hasBulkPath() const override {
    return false;
  }

  void seekToRowGroup(int64_t index) override;

  uint64_t skip(uint64_t numValues) override;

  void read(int64_t offset, const RowSet& rows, const uint64_t* incomingNulls)
      override;

  template <typename ColumnVisitor>
  void readWithVisitor(const RowSet& rows, ColumnVisitor visitor);

  std::unique_ptr<ByteRleDecoder> byteRle_;
  std::unique_ptr<BooleanRleDecoder> boolRle_;
};

template <typename ColumnVisitor>
void SelectiveByteRleColumnReader::readWithVisitor(
    const RowSet& rows,
    ColumnVisitor visitor) {
  if (boolRle_) {
    if (nullsInReadRange_) {
      boolRle_->readWithVisitor<true>(
          nullsInReadRange_->as<uint64_t>(), visitor);
    } else {
      boolRle_->readWithVisitor<false>(nullptr, visitor);
    }
  } else {
    if (nullsInReadRange_) {
      byteRle_->readWithVisitor<true>(
          nullsInReadRange_->as<uint64_t>(), visitor);
    } else {
      byteRle_->readWithVisitor<false>(nullptr, visitor);
    }
  }
}

} // namespace facebook::velox::dwrf
