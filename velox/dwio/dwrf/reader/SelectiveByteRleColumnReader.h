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

namespace facebook::velox::dwrf {

class SelectiveByteRleColumnReader
    : public dwio::common::SelectiveByteRleColumnReader {
 public:
  using ValueType = int8_t;

  SelectiveByteRleColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec,
      bool isBool)
      : dwio::common::SelectiveByteRleColumnReader(
            std::move(requestedType),
            params,
            scanSpec,
            dataType->type) {
    EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};
    auto& stripe = params.stripeStreams();
    if (isBool) {
      boolRle_ = createBooleanRleDecoder(
          stripe.getStream(encodingKey.forKind(proto::Stream_Kind_DATA), true),
          encodingKey);
    } else {
      byteRle_ = createByteRleDecoder(
          stripe.getStream(encodingKey.forKind(proto::Stream_Kind_DATA), true),
          encodingKey);
    }
  }

  void seekToRowGroup(uint32_t index) override {
    SelectiveColumnReader::seekToRowGroup(index);
    auto positionsProvider = formatData_->seekToRowGroup(index);
    if (boolRle_) {
      boolRle_->seekToRowGroup(positionsProvider);
    } else {
      byteRle_->seekToRowGroup(positionsProvider);
    }

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override {
    numValues = formatData_->skipNulls(numValues);
    if (byteRle_) {
      byteRle_->skip(numValues);
    } else {
      boolRle_->skip(numValues);
    }
    return numValues;
  }

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override {
    readCommon<SelectiveByteRleColumnReader>(offset, rows, incomingNulls);
  }

  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor);

  std::unique_ptr<ByteRleDecoder> byteRle_;
  std::unique_ptr<BooleanRleDecoder> boolRle_;
};

template <typename ColumnVisitor>
void SelectiveByteRleColumnReader::readWithVisitor(
    RowSet rows,
    ColumnVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
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
  readOffset_ += numRows;
}

} // namespace facebook::velox::dwrf
