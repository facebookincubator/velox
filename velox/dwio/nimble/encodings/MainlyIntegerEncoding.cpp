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
#include "velox/dwio/nimble/encodings/MainlyIntegerEncoding.h"

#include <algorithm>
#include <charconv>
#include <limits>
#include <tuple>
#include <utility>

#include <fmt/format.h>

#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"

namespace facebook::nimble {

namespace {

// The length of the longest int64 string, "-9223372036854775808".
constexpr size_t kMaxSpelledIntegerBytes{20};

// How many rows estimateSize() samples before it converts every row.
constexpr size_t kNumSampledRows{64};

// Returns the int64 value of 'row' when 'row' is an integer row. Returns
// nullopt for an exception row.
//
// An integer row is in canonical decimal form: an optional '-', then digits.
// It has no '+' and no leading zero, except "0" itself. "-0" is not
// canonical. In canonical form, spelling the value back gives the same
// string. E.g. "-42" gives -42. "042", "+42", "-0" and "4e2" are exception
// rows.
std::optional<int64_t> toInteger(std::string_view row) {
  if (row.empty() || row.size() > kMaxSpelledIntegerBytes) {
    return std::nullopt;
  }
  const char* const end = row.data() + row.size();
  int64_t parsed{0};
  const auto [last, error] = std::from_chars(row.data(), end, parsed);
  if (error != std::errc{} || last != end) {
    return std::nullopt;
  }
  // from_chars accepts "007" and "-0". Spelling the value back and comparing
  // rejects them.
  char spelled[kMaxSpelledIntegerBytes];
  const auto [spelledEnd, spellError] =
      std::to_chars(spelled, spelled + kMaxSpelledIntegerBytes, parsed);
  if (spellError != std::errc{} ||
      std::string_view{spelled, static_cast<size_t>(spelledEnd - spelled)} !=
          row) {
    return std::nullopt;
  }
  return parsed;
}

} // namespace

namespace {

// Records that 'numRows' more rows are about to be read from 'part'.
// 'numConsumed' counts the rows already read from it and is updated here.
// The is-integer stream decides how many rows come from each part. In a
// corrupted file it can mark more rows than 'part' holds, so this throws
// before the nested read would run past the end of 'part'.
void consumePartRows(
    const Encoding* part,
    uint32_t numRows,
    std::string_view partName,
    uint32_t& numConsumed) {
  const uint32_t numPartRows = part == nullptr ? 0 : part->rowCount();
  NIMBLE_CHECK_FILE_LE(
      static_cast<uint64_t>(numConsumed) + numRows,
      numPartRows,
      "MainlyInteger is-integer stream marks more rows than the part stores: {}",
      partName);
  numConsumed += numRows;
}

} // namespace

MainlyIntegerEncoding::MainlyIntegerEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<std::string_view, std::string_view>{pool, data, options},
      isIntegerRows_{getVectorBuffer<bool>()},
      integerRows_{getVectorBuffer<uint64_t>()},
      exceptionRows_{getVectorBuffer<physicalType>()},
      decoded_{getVectorBuffer<physicalType>()} {
  init(data, stringBufferFactory);
}

MainlyIntegerEncoding::~MainlyIntegerEncoding() {
  releaseVectorBuffer(isIntegerRows_);
  releaseVectorBuffer(integerRows_);
  releaseVectorBuffer(exceptionRows_);
  releaseVectorBuffer(decoded_);
}

void MainlyIntegerEncoding::init(
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory) {
  stringBufferFactory_ = stringBufferFactory;

  // Rebinds 'child' to the bytes of 'part'. Returns null when the stream
  // omits the part, i.e. when 'part' is empty.
  const auto initChild =
      [&](std::string_view part,
          std::unique_ptr<Encoding> child) -> std::unique_ptr<Encoding> {
    if (part.empty()) {
      return nullptr;
    }
    return rebindChild(part, stringBufferFactory, std::move(child));
  };

  // The parts follow the prefix:
  //   varint isIntegerSize | isInteger
  //   varint integersSize | integers
  //   exceptions
  std::string_view body = data.substr(dataOffset());
  isInteger_ =
      initChild(encoding::readLengthPrefixedBytes(body), std::move(isInteger_));
  integers_ =
      initChild(encoding::readLengthPrefixedBytes(body), std::move(integers_));

  // The exceptions run to the end of 'data'.
  exceptions_ = initChild(body, std::move(exceptions_));

  // The is-integer stream is needed only to interleave integer rows with
  // exception rows. So it is stored exactly when both other parts are.
  NIMBLE_CHECK_FILE_EQ(
      isInteger_ != nullptr,
      integers_ != nullptr && exceptions_ != nullptr,
      "MainlyInteger stores the is-integer stream exactly when it stores both "
      "integers and exceptions");
  const uint32_t numIntegerRows =
      integers_ == nullptr ? 0 : integers_->rowCount();
  const uint32_t numExceptions =
      exceptions_ == nullptr ? 0 : exceptions_->rowCount();
  // A stream with rows stores at least one of the integers and the
  // exceptions, because together they hold every row.
  NIMBLE_CHECK_FILE_EQ(
      static_cast<uint64_t>(numIntegerRows) + numExceptions,
      rowCount(),
      "MainlyInteger parts must cover every row");
  if (isInteger_ != nullptr) {
    NIMBLE_CHECK_FILE_EQ(
        isInteger_->rowCount(),
        rowCount(),
        "MainlyInteger is-integer stream must have one row per stream row");
    NIMBLE_CHECK_FILE_EQ(
        isInteger_->dataType(),
        DataType::Bool,
        "MainlyInteger is-integer stream must be a bool stream");
  }
  if (exceptions_ != nullptr) {
    NIMBLE_CHECK_FILE_EQ(
        exceptions_->dataType(),
        DataType::String,
        "MainlyInteger exceptions must be a string stream");
  }
  if (integers_ != nullptr) {
    NIMBLE_CHECK_FILE_EQ(
        integers_->dataType(),
        DataType::Uint64,
        "MainlyInteger integers must be a uint64 stream");
  }
  numReadRows_ = 0;
  numConsumedIntegers_ = 0;
  numConsumedExceptions_ = 0;
  decoded_.clear();
  decodedOffset_ = 0;
}

bool MainlyIntegerEncoding::rebindImpl(
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory) {
  init(data, stringBufferFactory);
  return true;
}

void MainlyIntegerEncoding::reset() {
  for (auto* child : {isInteger_.get(), integers_.get(), exceptions_.get()}) {
    if (child != nullptr) {
      child->reset();
    }
  }
  numReadRows_ = 0;
  numConsumedIntegers_ = 0;
  numConsumedExceptions_ = 0;
  decoded_.clear();
  decodedOffset_ = 0;
}

void MainlyIntegerEncoding::skip(uint32_t rowCount) {
  NIMBLE_CHECK_LE(
      rowCount,
      numRemainingRows(),
      "Skipping past the end of the MainlyInteger encoding");
  // Rows that readWithVisitor() already decoded come first. They are skipped
  // before any part is read.
  const auto numFromDecoded = std::min(rowCount, numRemainingDecoded());
  decodedOffset_ += numFromDecoded;
  const uint32_t numRows = rowCount - numFromDecoded;
  if (numRows == 0) {
    return;
  }
  numReadRows_ += numRows;
  // Without the is-integer stream, every row comes from the one part the
  // stream stores.
  if (isInteger_ == nullptr && integers_ != nullptr) {
    consumePartRows(integers_.get(), numRows, "integers", numConsumedIntegers_);
    integers_->skip(numRows);
    return;
  }
  if (isInteger_ == nullptr) {
    consumePartRows(
        exceptions_.get(), numRows, "exceptions", numConsumedExceptions_);
    exceptions_->skip(numRows);
    return;
  }
  isIntegerRows_.resize(numRows);
  isInteger_->materialize(numRows, isIntegerRows_.data());
  const auto numIntegerRows = static_cast<uint32_t>(
      std::count(isIntegerRows_.data(), isIntegerRows_.data() + numRows, true));
  if (numIntegerRows > 0) {
    consumePartRows(
        integers_.get(), numIntegerRows, "integers", numConsumedIntegers_);
    integers_->skip(numIntegerRows);
  }
  if (numRows > numIntegerRows) {
    consumePartRows(
        exceptions_.get(),
        numRows - numIntegerRows,
        "exceptions",
        numConsumedExceptions_);
    exceptions_->skip(numRows - numIntegerRows);
  }
}

void MainlyIntegerEncoding::materialize(uint32_t rowCount, void* buffer) {
  NIMBLE_CHECK_LE(
      rowCount,
      numRemainingRows(),
      "Materializing past the end of the MainlyInteger encoding");
  auto* output = static_cast<physicalType*>(buffer);
  // Rows that readWithVisitor() already decoded come first. They are copied
  // before any part is read.
  const auto numFromDecoded = std::min(rowCount, numRemainingDecoded());
  std::copy_n(decoded_.data() + decodedOffset_, numFromDecoded, output);
  decodedOffset_ += numFromDecoded;
  const uint32_t numRows = rowCount - numFromDecoded;
  if (numRows > 0) {
    decode(numRows, output + numFromDecoded);
    numReadRows_ += numRows;
  }
}

void MainlyIntegerEncoding::decode(uint32_t rowCount, physicalType* output) {
  // Every row is an exception row. The exceptions decode straight into
  // 'output'.
  if (isInteger_ == nullptr && integers_ == nullptr) {
    consumePartRows(
        exceptions_.get(), rowCount, "exceptions", numConsumedExceptions_);
    exceptions_->materialize(rowCount, output);
    return;
  }
  const bool* isInteger = nullptr;
  uint32_t numIntegerRows = rowCount;
  if (isInteger_ != nullptr) {
    isIntegerRows_.resize(rowCount);
    isInteger_->materialize(rowCount, isIntegerRows_.data());
    isInteger = isIntegerRows_.data();
    numIntegerRows = static_cast<uint32_t>(
        std::count(isInteger, isInteger + rowCount, true));
  }
  integerRows_.resize(numIntegerRows);
  if (numIntegerRows > 0) {
    consumePartRows(
        integers_.get(), numIntegerRows, "integers", numConsumedIntegers_);
    integers_->materialize(numIntegerRows, integerRows_.data());
  }
  const uint32_t numExceptions = rowCount - numIntegerRows;
  exceptionRows_.resize(numExceptions);
  if (numExceptions > 0) {
    consumePartRows(
        exceptions_.get(), numExceptions, "exceptions", numConsumedExceptions_);
    exceptions_->materialize(numExceptions, exceptionRows_.data());
  }
  writeRows(
      rowCount,
      isInteger,
      {integerRows_.data(), integerRows_.size()},
      {exceptionRows_.data(), exceptionRows_.size()},
      output);
}

void MainlyIntegerEncoding::writeRows(
    uint32_t rowCount,
    const bool* isInteger,
    std::span<const uint64_t> integers,
    std::span<const physicalType> exceptions,
    physicalType* output) {
  // Each integer row gets its own slot of kMaxSpelledIntegerBytes bytes.
  // Any int64 fits in one slot, so no first pass is needed to count the
  // bytes. All slots share one buffer from 'stringBufferFactory_'. So the
  // strings stay valid after this encoding is destroyed, like the decoded
  // strings of other encodings.
  const uint64_t numSlotBytes =
      uint64_t{integers.size()} * kMaxSpelledIntegerBytes;
  char* slot = nullptr;
  if (!integers.empty()) {
    NIMBLE_CHECK_LE(
        numSlotBytes,
        std::numeric_limits<uint32_t>::max(),
        "MainlyInteger decode spells too many bytes");
    slot = static_cast<char*>(
        stringBufferFactory_(static_cast<uint32_t>(numSlotBytes)));
  }
  uint32_t nextInteger{0};
  uint32_t nextException{0};
  for (uint32_t row = 0; row < rowCount; ++row) {
    if (isInteger != nullptr && !isInteger[row]) {
      output[row] = exceptions[nextException++];
      continue;
    }
    const auto [end, error] = std::to_chars(
        slot,
        slot + kMaxSpelledIntegerBytes,
        static_cast<int64_t>(integers[nextInteger++]));
    NIMBLE_DCHECK(error == std::errc{}, "int64 spelling overflows");
    output[row] = {slot, static_cast<size_t>(end - slot)};
    slot += kMaxSpelledIntegerBytes;
  }
}

MainlyIntegerEncoding::physicalType MainlyIntegerEncoding::decodeNext() {
  if (numRemainingDecoded() == 0) {
    // Checked before the subtraction below, so that it cannot wrap around.
    NIMBLE_CHECK_LT(
        numReadRows_, rowCount(), "Read past the last MainlyInteger row");
    const auto numRows = std::min(kDecodeBatchRows, rowCount() - numReadRows_);
    decoded_.resize(numRows);
    decode(numRows, decoded_.data());
    numReadRows_ += numRows;
    decodedOffset_ = 0;
  }
  return decoded_[decodedOffset_++];
}

std::string_view MainlyIntegerEncoding::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> rows,
    Buffer& buffer,
    const Encoding::Options& options) {
  NIMBLE_CHECK_LE(rows.size(), std::numeric_limits<uint32_t>::max());
  const bool useVarint = options.useVarintRowCount;
  const auto rowCount = static_cast<uint32_t>(rows.size());
  auto* pool = &buffer.getMemoryPool();
  // Splits the rows into the three parts. 'isInteger' holds one flag per
  // row. 'integers' holds each integer row as uint64. 'exceptions' holds the
  // other rows. E.g. "12", "x", "-3" give isInteger [true, false, true],
  // integers [12, -3] and exceptions ["x"].
  Vector<bool> isInteger{pool, rowCount};
  Vector<uint64_t> integers{pool};
  Vector<physicalType> exceptions{pool};
  integers.reserve(rowCount);
  for (uint32_t i = 0; i < rowCount; ++i) {
    const auto converted = toInteger(rows[i]);
    isInteger[i] = converted.has_value();
    if (converted.has_value()) {
      integers.push_back(static_cast<uint64_t>(*converted));
    } else {
      exceptions.push_back(rows[i]);
    }
  }
  // Tests change the parts here, before they are encoded, to build corrupted
  // streams. The hook gets a tuple of pointers to the three vectors. The
  // tuple order is (isInteger, integers, exceptions).
  auto childStreamPointers =
      std::make_tuple(&isInteger, &integers, &exceptions);
  velox::common::testutil::TestValue::adjust(
      "facebook::nimble::MainlyIntegerEncoding::encode", &childStreamPointers);

  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  // Each part is encoded only when the shape needs it. A replayed layout may
  // come from data of another shape. Its missing child layouts fall back to
  // normal selection, and the layouts of parts not stored here are unused.
  // Empty 'rows' store no part and decode to no rows. E.g. a Nullable stream
  // of only nulls encodes no rows.
  std::string_view serializedIsInteger;
  if (!integers.empty() && !exceptions.empty()) {
    serializedIsInteger = selection.template encodeNested<bool>(
        EncodingIdentifiers::MainlyInteger::IsInteger,
        isInteger,
        scopedBuffer.get(),
        options);
  }
  std::string_view serializedExceptions;
  if (!exceptions.empty()) {
    serializedExceptions = selection.template encodeNested<physicalType>(
        EncodingIdentifiers::MainlyInteger::Exceptions,
        exceptions,
        scopedBuffer.get(),
        options);
  }
  std::string_view serializedIntegers;
  if (!integers.empty()) {
    serializedIntegers = selection.template encodeNested<uint64_t>(
        EncodingIdentifiers::MainlyInteger::Integers,
        integers,
        scopedBuffer.get(),
        options);
  }

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(rowCount, useVarint) +
      varint::varintSize(serializedIsInteger.size()) +
      serializedIsInteger.size() +
      varint::varintSize(serializedIntegers.size()) +
      serializedIntegers.size() + serializedExceptions.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::MainlyInteger,
      TypeTraits<physicalType>::dataType,
      rowCount,
      useVarint,
      pos);
  encoding::writeVarintString(serializedIsInteger, pos);
  encoding::writeVarintString(serializedIntegers, pos);
  encoding::writeBytes(serializedExceptions, pos);
  NIMBLE_CHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

std::optional<uint64_t> MainlyIntegerEncoding::estimateSize(
    std::span<const physicalType> rows,
    const Encoding::Options& options) {
  if (rows.empty()) {
    return std::nullopt;
  }
  NIMBLE_CHECK_LE(rows.size(), std::numeric_limits<uint32_t>::max());
  // Check a sample of the rows first. If fewer than half are integer rows,
  // decline without converting every row. This keeps the candidate cheap on
  // streams that are mostly not integers.
  const size_t step = std::max<size_t>(1, rows.size() / kNumSampledRows);
  size_t numSampled{0};
  size_t numSampledIntegers{0};
  for (size_t i = 0; i < rows.size(); i += step) {
    ++numSampled;
    numSampledIntegers += toInteger(rows[i]).has_value() ? 1 : 0;
  }
  if (numSampledIntegers * 2 < numSampled) {
    return std::nullopt;
  }

  uint64_t numIntegerRows{0};
  uint64_t minInteger{std::numeric_limits<uint64_t>::max()};
  uint64_t maxInteger{0};
  uint64_t numExceptions{0};
  uint64_t exceptionBytes{0};
  uint64_t minExceptionBytes{std::numeric_limits<uint64_t>::max()};
  uint64_t maxExceptionBytes{0};
  for (const auto& row : rows) {
    const auto converted = toInteger(row);
    if (converted.has_value()) {
      // The integers part stores each value as uint64. So the range is
      // computed on that form.
      const auto physical = static_cast<uint64_t>(*converted);
      minInteger = std::min(minInteger, physical);
      maxInteger = std::max(maxInteger, physical);
      ++numIntegerRows;
      continue;
    }
    ++numExceptions;
    exceptionBytes += row.size();
    minExceptionBytes = std::min<uint64_t>(minExceptionBytes, row.size());
    maxExceptionBytes = std::max<uint64_t>(maxExceptionBytes, row.size());
  }
  if (numIntegerRows == 0) {
    return std::nullopt;
  }
  uint64_t size =
      EncodingPrefix::serializedSize(
          static_cast<uint32_t>(rows.size()), options.useVarintRowCount) +
      2 * sizeof(uint32_t) +
      FixedBitWidthEncoding<uint64_t>::estimateSize(
          numIntegerRows, minInteger, maxInteger, options);
  if (numExceptions > 0) {
    size += TrivialEncoding<std::string_view>::estimateSize(
        numExceptions,
        exceptionBytes,
        minExceptionBytes,
        maxExceptionBytes,
        options);
    // The is-integer stream is estimated as the smaller of two forms: a
    // bitmap with one bit per row, or the uint32 positions of its rarer value.
    const uint64_t bitmapBytes = (rows.size() + 7) / 8;
    const uint64_t sparseBytes =
        sizeof(uint32_t) * std::min(numExceptions, numIntegerRows);
    size += EncodingPrefix::kFixedPrefixSize + 1 +
        std::min(bitmapBytes, sparseBytes);
  }
  return size;
}

std::string MainlyIntegerEncoding::debugString(int offset) const {
  std::string result = Encoding::debugString(offset);
  const std::pair<const char*, const Encoding*> children[] = {
      {"isInteger", isInteger_.get()},
      {"integers", integers_.get()},
      {"exceptions", exceptions_.get()},
  };
  for (const auto& [name, child] : children) {
    if (child != nullptr) {
      result += fmt::format(
          "\n{}{} child:\n{}",
          std::string(offset + 2, ' '),
          name,
          child->debugString(offset + 4));
    }
  }
  return result;
}

} // namespace facebook::nimble
