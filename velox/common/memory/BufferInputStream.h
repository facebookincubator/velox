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

#include "velox/common/base/ByteInputStream.h"
#include "velox/common/base/Scratch.h"

#include <memory>

namespace facebook::velox {

/// Read-only input stream backed by a set of buffers.
class BufferInputStream : public ByteInputStream {
 public:
  explicit BufferInputStream(std::vector<ByteRange> ranges) {
    VELOX_CHECK(!ranges.empty(), "Empty BufferInputStream");
    ranges_ = std::move(ranges);
    current_ = &ranges_[0];
  }

  BufferInputStream(const BufferInputStream&) = delete;
  BufferInputStream& operator=(const BufferInputStream& other) = delete;
  BufferInputStream(BufferInputStream&& other) noexcept = delete;
  BufferInputStream& operator=(BufferInputStream&& other) noexcept = delete;

  size_t size() const override;

  bool atEnd() const override;

  std::streampos tellp() const override;

  void seekp(std::streampos pos) override;

  size_t remainingSize() const override;

  uint8_t readByte() override;

  void readBytes(uint8_t* bytes, int32_t size) override;

  std::string_view nextView(int32_t size) override;

  void skip(int32_t size) override;

  std::string toString() const override;

 private:
  // Sets 'current_' to the next range of input. The input is consecutive
  // ByteRanges in 'ranges_' for the base class but any view over external
  // buffers can be made by specialization.
  void nextRange();

  const std::vector<ByteRange>& ranges() const {
    return ranges_;
  }
};

} // namespace facebook::velox
