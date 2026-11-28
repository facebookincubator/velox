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

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>

#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

namespace facebook::nimble {

/// Stores a string stream whose rows mostly spell integers.
///
/// Terms:
/// - A row is one string of the stream.
/// - An integer row spells an int64 in canonical decimal form: an optional
///   '-', then digits. It has no '+' and no leading zero. "-0" is not
///   canonical. E.g. "12" and "-3" are integer rows.
/// - An exception row is any other row. E.g. "x", "012" and "+12".
/// - The is-integer stream holds one bool per row. It is true for an integer
///   row.
/// - A child is one of the three nested streams: the is-integer stream, the
///   integers and the exceptions.
///
/// Example: the rows "12", "x", "-3" store the is-integer stream
/// [true, false, true], the integers [12, -3] and the exceptions ["x"].
/// Decoding returns every row unchanged.
///
/// Encoding selection picks an encoding for each child on its own. So the
/// integers can be bit-packed. A mostly-true is-integer stream costs almost
/// nothing.
///
/// A stream stores only the children it needs. There are three shapes:
/// - Every row is an integer row: only the integers.
/// - Every row is an exception row: only the exceptions.
/// - Both kinds of rows: all three children.
/// So the is-integer stream is stored exactly when both other children are. An
/// empty stream stores no child. Any input has a shape, so a cached or
/// replayed MainlyInteger layout works on data of any shape.
///
/// Decoding turns each integer back into its decimal string. The strings live
/// in buffers from the string buffer factory.
///
/// MainlyInteger is not a default candidate. Enable it with a read factor,
/// e.g. "MainlyInteger=1.0".
///
/// Data layout:
///   EncodingPrefix bytes: standard Encoding prefix
///   varint:  size of the is-integer stream. 0 unless the stream has both
///            integer rows and exception rows.
///   X bytes: nested Bool is-integer stream
///   varint:  size of the integers. 0 when no row is an integer row.
///   Y bytes: nested integers, one uint64 per integer row. Each uint64 holds
///            the two's complement bits of the int64.
///   Z bytes: nested exceptions, one string per exception row. They run to
///            the end of the stream. No bytes when no row is an exception
///            row.
class MainlyIntegerEncoding final
    : public TypedEncoding<std::string_view, std::string_view> {
 public:
  using cppDataType = std::string_view;
  using physicalType = std::string_view;

  MainlyIntegerEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  ~MainlyIntegerEncoding() override;

  MainlyIntegerEncoding(const MainlyIntegerEncoding&) = delete;
  MainlyIntegerEncoding& operator=(const MainlyIntegerEncoding&) = delete;
  MainlyIntegerEncoding(MainlyIntegerEncoding&&) = delete;
  MainlyIntegerEncoding& operator=(MainlyIntegerEncoding&&) = delete;

  void reset() final;

  void skip(uint32_t rowCount) final;

  void materialize(uint32_t rowCount, void* buffer) final;

  /// Decodes the selected rows through the generic visitor interface. It
  /// decodes kDecodeBatchRows rows at a time.
  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> rows,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates the encoded size in bytes. Returns nullopt in three cases:
  /// - 'rows' is empty.
  /// - In a sample of the rows, fewer than half are integer rows.
  /// - No row is an integer row.
  /// Otherwise it converts every row and adds three estimates: the integers
  /// as FixedBitWidth, the exceptions as Trivial and the is-integer stream as
  /// a bitmap.
  static std::optional<uint64_t> estimateSize(
      std::span<const physicalType> rows,
      const Encoding::Options& options);

  std::string debugString(int offset) const final;

 protected:
  // Initializes the children from 'data'. rebind() has already checked its
  // prefix.
  bool rebindImpl(
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory) final;

 private:
  // How many rows readWithVisitor() decodes in one batch.
  static constexpr uint32_t kDecodeBatchRows{1'024};

  // Reads the three children from 'data', initializes each one and checks that
  // they fit together. Throws when they do not, which means the file is
  // corrupted.
  void init(
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory);

  // Decodes the next 'rowCount' rows from the children into 'output'. It does
  // not use the rows in 'decoded_'.
  void decode(uint32_t rowCount, physicalType* output);

  // Fills 'output' with 'rowCount' rows in one pass. 'isInteger' says which
  // child each row comes from. Null 'isInteger' means every row is an integer
  // row. An integer row takes the next value of 'integers', spelled in
  // decimal. An exception row takes the next value of 'exceptions'.
  void fillRows(
      uint32_t rowCount,
      const bool* isInteger,
      std::span<const uint64_t> integers,
      std::span<const physicalType> exceptions,
      physicalType* output);

  // Returns the next row for readWithVisitor(). When 'decoded_' is used up,
  // it first decodes the next batch into 'decoded_'.
  physicalType decodeNext();

  uint32_t numRemainingDecoded() const {
    return static_cast<uint32_t>(decoded_.size()) - decodedOffset_;
  }

  // How many rows the caller has not read yet. 'numReadRows_' already counts
  // the rows in 'decoded_', so the unread ones among them are added back.
  uint32_t numRemainingRows() const {
    return rowCount() - numReadRows_ + numRemainingDecoded();
  }

  // Allocates the buffers that decoded strings point into.
  std::function<void*(uint32_t)> stringBufferFactory_;
  // Null unless the stream has both integer rows and exception rows.
  std::unique_ptr<Encoding> isInteger_;
  // Null when no row is an integer row.
  std::unique_ptr<Encoding> integers_;
  // Null when no row is an exception row.
  std::unique_ptr<Encoding> exceptions_;
  // Scratch space that decode() and skip() reuse across calls.
  Vector<bool> isIntegerRows_;
  Vector<uint64_t> integerRows_;
  Vector<physicalType> exceptionRows_;
  // How many rows have been read from the children. This includes every row in
  // 'decoded_', even the ones the caller has not read yet.
  uint32_t numReadRows_{0};
  // How many values have been read from 'integers_'.
  uint32_t numConsumedIntegers_{0};
  // How many values have been read from 'exceptions_'.
  uint32_t numConsumedExceptions_{0};
  // Rows that readWithVisitor() decoded before the caller asked for them.
  Vector<physicalType> decoded_;
  // How many rows of 'decoded_' the caller has already read.
  uint32_t decodedOffset_{0};
};

//
// End of public API. Implementations follow.
//

template <typename DecoderVisitor>
void MainlyIntegerEncoding::readWithVisitor(
    DecoderVisitor& visitor,
    ReadWithVisitorParams& params) {
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] { return decodeNext(); });
}

} // namespace facebook::nimble
