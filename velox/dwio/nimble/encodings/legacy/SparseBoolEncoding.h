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
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Represents bools sparsely. Useful when only a small fraction  are set (or not
// set).
//
// In terms of space, this will be better than the trivial bitmap encoding when
// the fraction of set (or unset) bits is smaller than 1 / lg(n), where n is the
// number of rows in the stream. On normal size files thats around 5%.
//
// This encoding will likely be of most use in representing the nulls
// subencoding inside a NullableEncoding.
//
// TODO: Should we generalize the SparseEncoding idea to other data types?
// Maybe. Or maybe in our mainly-constant encoding implementation we simple use
// a SparseBoolEncoding on top of the dense encoded data.

namespace facebook::nimble::legacy {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 1 byte: whether the sparse bits are set or unset
// XX bytes: indices encoding bytes
class SparseBoolEncoding final : public TypedEncoding<bool, bool> {
 public:
  using cppDataType = bool;

  static constexpr int kPrefixSize = 1;
  static constexpr int kSparseValueOffset = EncodingPrefix::kFixedPrefixSize;
  static constexpr int kIndicesOffset = EncodingPrefix::kFixedPrefixSize + 1;

  SparseBoolEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<bool>& selection,
      std::span<const bool> values,
      Buffer& buffer);

 private:
  // If true, then indices give the position of the set bits; if false they give
  // the positions of the unset bits.
  const bool sparseValue_;
  Vector<char> indicesUncompressed_;
  detail::BufferedEncoding<uint32_t, 32> indices_;
  uint32_t nextIndex_{}; // The current index (FBA value).
  uint32_t row_{};
};

template <typename V>
void SparseBoolEncoding::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        while (nextIndex_ < row_) {
          nextIndex_ = indices_.nextValue();
        }
        if (FOLLY_UNLIKELY(nextIndex_ == row_++)) {
          nextIndex_ = indices_.nextValue();
          return sparseValue_;
        }
        return !sparseValue_;
      });
}

} // namespace facebook::nimble::legacy
