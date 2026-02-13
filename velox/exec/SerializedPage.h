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

#include "velox/common/memory/ByteStream.h"

namespace facebook::velox::exec {

/// Interface for serialized pages.
class SerializedPageBase {
 public:
  virtual ~SerializedPageBase() = default;

  /// Returns the size of the serialized data in bytes.
  virtual uint64_t size() const = 0;

  /// Returns the number of rows if available.
  virtual std::optional<int64_t> numRows() const = 0;

  /// Makes 'input' ready for deserializing 'this' with
  /// VectorStreamGroup::read().
  virtual std::unique_ptr<ByteInputStream> prepareStreamForDeserialize() = 0;

  /// Returns a clone of the IOBuf.
  virtual std::unique_ptr<folly::IOBuf> getIOBuf() const = 0;
};

/// Corresponds to Presto SerializedPage, i.e. a container for serialized
/// vectors in Presto wire format.
class PrestoSerializedPage : public SerializedPageBase {
 public:
  /// Construct from IOBuf chain.
  explicit PrestoSerializedPage(
      std::unique_ptr<folly::IOBuf> iobuf,
      std::function<void(folly::IOBuf&)> onDestructionCb = nullptr,
      std::optional<int64_t> numRows = std::nullopt);

  ~PrestoSerializedPage() override;

  uint64_t size() const override {
    return iobufBytes_;
  }

  std::optional<int64_t> numRows() const override {
    return numRows_;
  }

  std::unique_ptr<ByteInputStream> prepareStreamForDeserialize() override;

  std::unique_ptr<folly::IOBuf> getIOBuf() const override {
    return iobuf_->clone();
  }

 private:
  static int64_t chainBytes(folly::IOBuf& iobuf) {
    int64_t size = 0;
    for (auto& range : iobuf) {
      size += range.size();
    }
    return size;
  }

  // Buffers containing the serialized data. The memory is owned by 'iobuf_'.
  std::vector<ByteRange> ranges_;

  // IOBuf holding the data in 'ranges_.
  std::unique_ptr<folly::IOBuf> iobuf_;

  // Number of payload bytes in 'iobuf_'.
  const int64_t iobufBytes_;

  // Number of payload rows, if provided.
  const std::optional<int64_t> numRows_;

  // Callback that will be called on destruction of the PrestoSerializedPage,
  // primarily used to free externally allocated memory backing folly::IOBuf
  // from caller. Caller is responsible to pass in proper cleanup logic to
  // prevent any memory leak.
  std::function<void(folly::IOBuf&)> onDestructionCb_;
};

// TODO: Remove after fully migration to new SerializedPageBase and
// PrestoSerializedPage API.
using SerializedPage = PrestoSerializedPage;
} // namespace facebook::velox::exec
