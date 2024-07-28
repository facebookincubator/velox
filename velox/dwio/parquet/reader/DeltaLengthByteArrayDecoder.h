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

#include "velox/dwio/parquet/reader/DeltaBpDecoder.h"

namespace facebook::velox::parquet {

// DeltaByteArrayDecoder is adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/parquet/encoding.cc#L2758-L2889
class DeltaLengthByteArrayDecoder {
 public:
  explicit DeltaLengthByteArrayDecoder(const char* start) {
    lengthDecoder_ = std::make_unique<DeltaBpDecoder>(start);
    decodeLengths();
    bufferStart_ = lengthDecoder_->bufferStart();
  }

  folly::StringPiece readString() {
    int32_t dataSize = 0;
    const int64_t length = bufferedLength_[lengthIdx_++];
    VELOX_CHECK_GE(length, 0, "negative string delta length");
    bufferStart_ += length;
    return folly::StringPiece(bufferStart_ - length, length);
  }

 private:
  void decodeLengths() {
    int64_t numLength = lengthDecoder_->validValuesCount();
    bufferedLength_.resize(numLength);
    lengthDecoder_->readValues<uint32_t>(bufferedLength_, numLength);

    lengthIdx_ = 0;
    numValidValues_ = numLength;
  }

  const char* bufferStart_;
  std::unique_ptr<DeltaBpDecoder> lengthDecoder_;
  int32_t numValidValues_{0};
  uint32_t lengthIdx_{0};
  std::vector<uint32_t> bufferedLength_;
};
} // namespace facebook::velox::parquet
