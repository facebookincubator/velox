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

#include <span>
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble::legacy {

/// Legacy PrefixEncoding for sorted string data with prefix compression.
/// Mirrors the master branch PrefixEncoding before stringBufferFactory was
/// added. The readWithVisitor returns decodeEntry() directly without
/// page-based string buffer management.
class PrefixEncoding final
    : public TypedEncoding<std::string_view, std::string_view> {
 public:
  using cppDataType = std::string_view;
  using physicalType = std::string_view;

  static constexpr uint32_t kDefaultRestartInterval = 16;

  PrefixEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  std::string debugString(int offset) const final;

 private:
  static uint32_t readRestartInterval(
      std::string_view data,
      uint32_t startOffset);
  static const char* restartOffsets(
      std::string_view data,
      uint32_t startOffset);
  static const char*
  dataStart(std::string_view data, uint32_t startOffset, uint32_t numRestarts);
  static uint32_t computeNumRestarts(
      uint32_t rowCount,
      uint32_t restartInterval);

  std::string_view decodeEntry();
  void seekToRestartPoint(uint32_t restartIndex);
  uint32_t restartOffset(uint32_t restartIndex) const;

  // Dummy member to maintain layout compatibility with
  // facebook::nimble::PrefixEncoding. The non-legacy dispatch in
  // EncodingUtils.h static_casts to the non-legacy type, so member offsets
  // must match.
  std::function<void*(uint32_t)> stringBufferFactory_;

  const uint32_t restartInterval_;
  const uint32_t numRestarts_;
  const char* const restartOffsets_;
  const char* const dataStart_;

  const char* currentPos_{nullptr};
  uint32_t currentRow_{0};
  Vector<char> decodedValue_;
  Vector<char> materializedValues_;
};

/// Template implementation (remain in header)
template <typename V>
void PrefixEncoding::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { this->skip(toSkip); },
      [&] { return decodeEntry(); });
}

} // namespace facebook::nimble::legacy
