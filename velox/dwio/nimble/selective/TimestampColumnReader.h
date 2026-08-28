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
#include "velox/dwio/nimble/selective/ChunkedDecoder.h"

namespace facebook::nimble {

class TimestampColumnReader
    : public velox::dwio::common::SelectiveColumnReader {
 public:
  TimestampColumnReader(
      const velox::TypePtr& requestedType,
      const std::shared_ptr<const velox::dwio::common::TypeWithId>& fileType,
      NimbleParams& params,
      velox::common::ScanSpec& scanSpec)
      : SelectiveColumnReader(requestedType, fileType, params, scanSpec),
        microsDecoder_(formatData().as<NimbleData>().makeMicrosDecoder()),
        nanosDecoder_(formatData().as<NimbleData>().makeNanosDecoder()) {}

  uint64_t skip(uint64_t numValues) override;

  void read(
      int64_t offset,
      const velox::RowSet& rows,
      const uint64_t* incomingNulls) override;

  void getValues(const velox::RowSet& rows, velox::VectorPtr* result) override;

  bool estimateMaterializedSize(size_t& byteSize, size_t& rowCount)
      const override;

 private:
  void readHelper(
      const velox::common::Filter* filter,
      const velox::RowSet& rows);

  void processNulls(
      bool isNull,
      const velox::RowSet& rows,
      const uint64_t* rawNulls);

  void processFilter(
      const velox::common::Filter* filter,
      const velox::RowSet& rows,
      const uint64_t* rawNulls);

  static velox::Timestamp convertToVeloxTimestamp(
      int64_t micros,
      uint16_t subMicrosNanos);

  bool readsNullsOnly() const final {
    return false;
  }

  ChunkedDecoder microsDecoder_;
  ChunkedDecoder nanosDecoder_;

  // Reusable buffers for read operations.
  velox::BufferPtr microsBuffer_;
  velox::BufferPtr nanosBuffer_;
};

} // namespace facebook::nimble
