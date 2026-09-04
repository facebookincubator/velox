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

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"

// Represents a slice of another encoding without performing the slice.
//
// Slicing an encoding normally means decoding enough of it to find the slice
// boundaries, slicing every child, and re-serializing. For encodings whose
// structure does not align with row offsets that is expensive: MainlyConstant
// has to count how many common values precede the slice before it can locate
// the corresponding sub-range of its otherValues child, which in the general
// case means materializing every bool up to the slice end.
//
// This encoding skips all of it. It copies the source encoding verbatim and
// records the row offset the consumer should start at, moving the work to
// decode time, where the reader is walking the rows anyway. The trade is size:
// the payload carries the whole source encoding rather than just the slice, so
// it is only worthwhile where the source is already compact relative to the
// cost of slicing it.

namespace facebook::nimble {

// Data layout is:
// Standard Encoding prefix (size varies with Options::useVarintRowCount),
//     whose row count is the SLICE length, not the inner encoding's row count.
// 4 bytes: row offset into the inner encoding at which the slice begins.
// remaining bytes: the inner encoding, verbatim.
template <typename T>
class SliceEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  SliceEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory = nullptr,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(pool, data, options),
        stringBufferFactory_{std::move(stringBufferFactory)},
        sliceOffset_{readSliceOffset(data, this->dataOffset())},
        inner_{readInner(data, this->dataOffset())} {
    reset();
  }

  void reset() final {
    // Recreated rather than reset+skip: an encoding's reset() returns it to row
    // zero, so the skip would have to be replayed on every reset anyway, and
    // constructing is what establishes the child's own read state.
    encoding_ = EncodingFactory{this->options_}.create(
        *this->pool_, inner_, stringBufferFactory_);
    if (sliceOffset_ > 0) {
      encoding_->skip(sliceOffset_);
    }
  }

  void skip(uint32_t rowCount) final {
    encoding_->skip(rowCount);
  }

  void materialize(uint32_t rowCount, void* buffer) final {
    encoding_->materialize(rowCount, buffer);
  }

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      override {
    encoding_->materializeBoolsAsBits(rowCount, buffer, begin);
  }

  template <typename DecoderVisitor>
  void readWithVisitor(
      DecoderVisitor& /*visitor*/,
      ReadWithVisitorParams& /*params*/) {
    // The selective reader path would need the visitor's row positions
    // translated by sliceOffset_. Nothing produces this encoding for that path
    // today -- it is emitted only by EncodingSliceFactory, whose output is
    // consumed through materialize() -- so fail loudly rather than silently
    // returning rows from the wrong offset.
    NIMBLE_UNSUPPORTED(
        "SliceEncoding does not support the selective reader path.");
  }

  std::string debugString(int offset) const override {
    return fmt::format(
        "{}{}<{}> rowCount={} sliceOffset={}\n{}",
        std::string(offset, ' '),
        toString(this->encodingType()),
        toString(this->dataType()),
        this->rowCount(),
        sliceOffset_,
        encoding_->debugString(offset + 2));
  }

  /// Wraps `encoded` so that a consumer sees `length` rows starting at
  /// `offset`, without slicing it. `encoded` is copied into `buffer`.
  static std::string_view wrap(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options) {
    const auto prefixSize =
        EncodingPrefix::serializedSize(length, options.useVarintRowCount);
    const auto encodingSize = prefixSize + sizeof(uint32_t) + encoded.size();
    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    EncodingPrefix::serialize(
        EncodingType::Slice,
        TypeTraits<T>::dataType,
        length,
        options.useVarintRowCount,
        pos);
    encoding::writeUint32(offset, pos);
    std::memcpy(pos, encoded.data(), encoded.size());
    pos += encoded.size();
    NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

 private:
  // Reads the 4-byte row offset that follows the encoding prefix.
  static uint32_t readSliceOffset(std::string_view data, uint32_t dataOffset) {
    const char* pos = data.data() + dataOffset;
    return encoding::readUint32(pos);
  }

  // Returns the inner encoding, stored verbatim after the row offset.
  static std::string_view readInner(
      std::string_view data,
      uint32_t dataOffset) {
    const char* pos = data.data() + dataOffset + sizeof(uint32_t);
    return {pos, static_cast<size_t>(data.data() + data.size() - pos)};
  }

  const std::function<void*(uint32_t)> stringBufferFactory_;
  const uint32_t sliceOffset_;
  const std::string_view inner_;
  // The only mutable member: reset() rebuilds the child rather than rewinding
  // it, so this is reassigned on every reset().
  std::unique_ptr<Encoding> encoding_;
};

} // namespace facebook::nimble
