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

#include "velox/dwio/common/SelectiveByteRleColumnReader.h"
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"

namespace facebook::nimble {

class ByteColumnReader
    : public velox::dwio::common::SelectiveByteRleColumnReader {
 public:
  ByteColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec)
      : SelectiveByteRleColumnReader(requestedType, fileType, params, scanSpec),
        decoder_(formatData().as<NimbleData>().makeScalarDecoder()) {}

  uint64_t skip(uint64_t numValues) final {
    numValues = SelectiveByteRleColumnReader::skip(numValues);
    decoder_.skip(numValues);
    return numValues;
  }

  void read(
      int64_t offset,
      const velox::RowSet& rows,
      const uint64_t* incomingNulls) final {
    readCommon<ByteColumnReader, false>(offset, rows, incomingNulls);
    readOffset_ += rows.back() + 1;
  }

  template <typename DecoderVisitor>
  void readWithVisitor(const velox::RowSet& /*rows*/, DecoderVisitor visitor) {
    decoder_.readWithVisitor(visitor);
  }

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const final {
    auto decoderRowCount = decoder_.estimateRowCount();
    if (!decoderRowCount.has_value()) {
      return false;
    }
    rowCount = *decoderRowCount;
    if (requestedType_->isBoolean()) {
      byteSize = rowCount / 8;
    } else {
      byteSize = rowCount;
    }
    return true;
  }

 private:
  bool readsNullsOnly() const final {
    return false;
  }

  ChunkedDecoder decoder_;
};

} // namespace facebook::nimble
