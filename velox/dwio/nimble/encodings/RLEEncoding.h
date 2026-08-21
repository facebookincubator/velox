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

#include <algorithm>
#include <optional>
#include <span>
#include <vector>

#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/DictionaryEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/BufferedEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/NestedAlpSizeEstimation.h"

// Holds data in RLE format. Consecutive equal values are collapsed into runs:
//
//   input:       A A A B B C
//   run lengths: 3 2 1
//   run values:  A B C
//
// Run lengths and run values are encoded as nested streams. This makes RLE a
// good fit for sorted or clustered data where long repeated runs reduce the
// number of values that need to be encoded.
//
// Note: we might want to recursively use the encoding factory to encode the
// run values. This recursive use can lead to great compression, but also
// tends to slow things down, particularly write speed.

namespace facebook::nimble {

namespace rle {

/// Builds run lengths and applies `transform` to each run's representative
/// input value before storing it in `runValues`. This allows floating-point
/// physical values to be converted back to their logical type for nested
/// encoding.
template <typename T, typename RunValue, typename Transform>
void computeRuns(
    std::span<const T> data,
    Vector<uint32_t>* runLengths,
    Vector<RunValue>* runValues,
    Transform transform) {
  static_assert(!std::is_floating_point_v<T>);
  if (data.empty()) {
    return;
  }
  uint32_t runLength = 1;
  T last = data[0];
  for (int i = 1; i < data.size(); ++i) {
    if (data[i] == last) {
      ++runLength;
    } else {
      runLengths->push_back(runLength);
      runValues->push_back(transform(last));
      last = data[i];
      runLength = 1;
    }
  }
  runLengths->push_back(runLength);
  runValues->push_back(transform(last));
}

template <typename T>
void computeRuns(
    std::span<const T> data,
    Vector<uint32_t>* runLengths,
    Vector<T>* runValues) {
  computeRuns(data, runLengths, runValues, [](const T value) { return value; });
}

} // namespace rle

namespace internal {

// Base case covers the datatype-independent functionality. We use the CRTP
// to avoid having to use virtual functions (namely on
// RLEEncodingBase::RunValue).
// Data layout is:
//   EncodingPrefix::kFixedPrefixSize bytes: standard Encoding data
//   4 bytes: runs size
//   X bytes: runs encoding bytes
template <typename T, typename RLEEncoding>
class RLEEncodingBase
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  RLEEncodingBase(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {})
      : TypedEncoding<T, physicalType>(pool, data, options),
        materializedRunLengths_{EncodingFactory(options).create(
            pool,
            {data.data() + this->dataOffset() + 4,
             *reinterpret_cast<const uint32_t*>(
                 data.data() + this->dataOffset())},
            stringBufferFactory)} {}

  void reset() override {
    materializedRunLengths_.reset();
    derived().resetValues();
    copiesRemaining_ = 0;
  }

  void skip(uint32_t rowCount) override {
    uint32_t rowsLeft = rowCount;
    while (rowsLeft > 0) {
      if (rowsLeft <= copiesRemaining_) {
        copiesRemaining_ -= rowsLeft;
        return;
      }
      rowsLeft -= copiesRemaining_;
      advanceRun();
    }
  }

  void materialize(uint32_t rowCount, void* buffer) override {
    uint32_t rowsLeft = rowCount;
    physicalType* output = static_cast<physicalType*>(buffer);
    while (rowsLeft > 0) {
      if (copiesRemaining_ == 0) {
        advanceRun();
      }
      if (rowsLeft < copiesRemaining_) {
        velox::simd::simdFill(output, currentValue_, rowsLeft);
        copiesRemaining_ -= rowsLeft;
        return;
      }
      velox::simd::simdFill(output, currentValue_, copiesRemaining_);
      output += copiesRemaining_;
      rowsLeft -= copiesRemaining_;
      copiesRemaining_ = 0;
    }
  }

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params) {
    detail::readWithVisitorSlow(
        visitor,
        params,
        [&](auto toSkip) { skip(toSkip); },
        [&] {
          if (copiesRemaining_ == 0) {
            advanceRun();
          }
          --copiesRemaining_;
          return currentValue_;
        });
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const bool useVarint = options.useVarintRowCount;
    const uint32_t valueCount = values.size();
    auto* pool = &buffer.getMemoryPool();
    Vector<uint32_t> runLengths(pool);
    ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
    std::string_view serializedRunLengths;
    std::string_view serializedRunValues;
    if constexpr (isFloatingPointType<T>()) {
      Vector<T> logicalRunValues(pool);
      rle::computeRuns(
          values, &runLengths, &logicalRunValues, [](const physicalType value) {
            return EncodingPhysicalType<T>::asEncodingLogicalType(value);
          });
      serializedRunLengths = selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::RunLength::RunLengths,
          runLengths,
          scopedBuffer.get(),
          options);
      serializedRunValues = RLEEncoding::getSerializedLogicalRunValues(
          selection, logicalRunValues, scopedBuffer.get(), options);
    } else {
      Vector<physicalType> runValues(pool);
      rle::computeRuns(values, &runLengths, &runValues);
      serializedRunLengths = selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::RunLength::RunLengths,
          runLengths,
          scopedBuffer.get(),
          options);
      serializedRunValues = getSerializedPhysicalRunValues(
          selection, runValues, scopedBuffer.get(), options);
    }

    const uint32_t encodingSize =
        Encoding::serializePrefixSize(valueCount, useVarint) + 4 +
        serializedRunLengths.size() + serializedRunValues.size();
    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    Encoding::serializePrefix(
        EncodingType::RLE, TypeTraits<T>::dataType, valueCount, useVarint, pos);
    encoding::writeString(serializedRunLengths, pos);
    encoding::writeBytes(serializedRunValues, pos);
    NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    const auto sourceRowCount =
        EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
    NIMBLE_CHECK_LE(offset, sourceRowCount);
    NIMBLE_CHECK_LE(length, sourceRowCount - offset);
    NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

    const auto sourcePrefixSize =
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
    const char* pos = encoded.data() + sourcePrefixSize;
    const auto runLengthsSize = encoding::readUint32(pos);
    const std::string_view runLengthsData{pos, runLengthsSize};
    pos += runLengthsSize;
    const std::string_view runValuesData{
        pos, static_cast<size_t>(encoded.end() - pos)};

    auto slicedRuns =
        sliceRuns(runLengthsData, offset, length, buffer, options);
    const auto slicedRunValues = RLEEncoding::sliceRunValues(
        runValuesData,
        slicedRuns.runValueOffset,
        slicedRuns.runValueCount,
        buffer,
        options);

    const auto prefixSize =
        EncodingPrefix::serializedSize(length, options.useVarintRowCount);
    const auto encodingSize = prefixSize + sizeof(uint32_t) +
        slicedRuns.encodedLengths.size() + slicedRunValues.size();
    char* reserved = buffer.reserve(encodingSize);
    char* output = reserved;
    EncodingPrefix::serialize(
        EncodingType::RLE,
        TypeTraits<T>::dataType,
        length,
        options.useVarintRowCount,
        output);
    encoding::writeString(slicedRuns.encodedLengths, output);
    encoding::writeBytes(slicedRunValues, output);
    NIMBLE_CHECK_EQ(output - reserved, encodingSize, "Encoding size mismatch.");
    return std::string_view{reserved, encodingSize};
  }

  const char* getValuesStart() const {
    return this->data_.data() + this->dataOffset() + 4 +
        *reinterpret_cast<const uint32_t*>(
               this->data_.data() + this->dataOffset());
  }

  RLEEncoding& derived() {
    return *static_cast<RLEEncoding*>(this);
  }
  physicalType nextValue() {
    return derived().nextValue();
  }

  void advanceRunLength() {
    copiesRemaining_ = materializedRunLengths_.nextValue();
  }

  void advanceRunValue() {
    currentValue_ = nextValue();
  }

  // Advances to the next run by loading both the run length and value.
  void advanceRun() {
    advanceRunLength();
    advanceRunValue();
  }

  static std::string_view getSerializedPhysicalRunValues(
      EncodingSelection<physicalType>& selection,
      const Vector<physicalType>& runValues,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    return RLEEncoding::getSerializedPhysicalRunValues(
        selection, runValues, buffer, options);
  }

  uint32_t copiesRemaining_ = 0;
  physicalType currentValue_;
  detail::BufferedEncoding<uint32_t, 32> materializedRunLengths_;

 private:
  struct RLESliceRuns {
    // Run-length child encoding for the returned RLE slice. This view points
    // into the caller-owned output buffer.
    std::string_view encodedLengths;
    // Run-value child range matching encodedLengths.
    uint32_t runValueOffset{0};
    uint32_t runValueCount{0};
  };

  static RLESliceRuns sliceRuns(
      std::string_view encodedRunLengths,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options) {
    const auto runCount = EncodingPrefix::readRowCount(
        encodedRunLengths, options.useVarintRowCount);
    NIMBLE_CHECK_GT(
        length, 0, "RLE run slicing requires a non-empty row range.");
    NIMBLE_CHECK_GT(
        runCount, 0, "Cannot slice a non-empty range from empty RLE runs.");
    auto* pool = &buffer.getMemoryPool();
    RLESliceRuns result;

    EncodingFactory encodingFactory{options};
    auto runLengthsEncoding = encodingFactory.create(
        *pool, encodedRunLengths, [](uint32_t /* totalLength */) -> void* {
          return nullptr;
        });

    const auto end = offset + length;
    uint32_t firstRunStart{0};
    uint32_t lastRunEnd{0};
    constexpr uint32_t kRunLengthChunkSize{256};
    const auto maxRunLengthChunkSize =
        std::min({runCount, length, kRunLengthChunkSize});
    Vector<uint32_t> runLengths{pool, maxRunLengthChunkSize};
    Vector<uint32_t> slicedRunLengths{pool};
    slicedRunLengths.reserve(length);
    uint32_t row{0};
    for (uint32_t run = 0; run < runCount;) {
      const auto chunkSize =
          std::min<uint32_t>(maxRunLengthChunkSize, runCount - run);
      runLengthsEncoding->materialize(chunkSize, runLengths.data());
      for (uint32_t i = 0; i < chunkSize; ++i, ++run) {
        const auto runStart = row;
        const auto runEnd = row + runLengths[i];
        row = runEnd;
        if (runEnd <= offset) {
          continue;
        }
        if (runStart >= end) {
          break;
        }
        if (result.runValueCount == 0) {
          result.runValueOffset = run;
          firstRunStart = runStart;
        }
        lastRunEnd = runEnd;
        ++result.runValueCount;
        slicedRunLengths.push_back(runLengths[i]);
      }
      if (row >= end) {
        break;
      }
    }

    NIMBLE_CHECK_GT(
        result.runValueCount,
        0,
        "Could not find RLE runs for a non-empty row range.");
    NIMBLE_CHECK_EQ(
        slicedRunLengths.size(),
        result.runValueCount,
        "Sliced RLE run length count mismatch.");

    if (firstRunStart < offset || lastRunEnd > end) {
      slicedRunLengths[0] -= offset - firstRunStart;
      slicedRunLengths[result.runValueCount - 1] -= lastRunEnd - end;
      result.encodedLengths = encodeRunLengthsSlice(
          encodedRunLengths, slicedRunLengths, buffer, options);
    } else {
      result.encodedLengths = EncodingFactory::slice(
          encodedRunLengths,
          result.runValueOffset,
          result.runValueCount,
          buffer,
          options);
    }
    return result;
  }

  static std::string_view encodeRunLengthsSlice(
      std::string_view encodedRunLengths,
      std::span<const uint32_t> runLengths,
      Buffer& buffer,
      const Encoding::Options& options) {
    NIMBLE_CHECK_GT(
        runLengths.size(), 0, "Cannot encode empty RLE run-length slice.");
    return EncodingFactory::encodeWithCapturedLayout<uint32_t>(
        encodedRunLengths,
        runLengths,
        buffer,
        options,
        "Captured RLE run-length layout");
  }
};

} // namespace internal

// Handles the numeric and string cases. Bools are templated below.
// Data layout is:
// RLEEncodingBase bytes
// 4 * sizeof(physicalType) bytes: run values
template <typename T>
class RLEEncoding final : public internal::RLEEncodingBase<T, RLEEncoding<T>> {
  using physicalType = typename TypeTraits<T>::physicalType;

 public:
  explicit RLEEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      const std::function<void*(uint32_t)>& stringBufferFactory,
      const Encoding::Options& options = {});

  ~RLEEncoding() override {
    this->releaseBuffer(indicesBuffer_);
  }

  RLEEncoding(const RLEEncoding&) = delete;
  RLEEncoding& operator=(const RLEEncoding&) = delete;
  RLEEncoding(RLEEncoding&&) = delete;
  RLEEncoding& operator=(RLEEncoding&&) = delete;

  void reset() final;

  void skip(uint32_t rowCount) final;

  void materialize(uint32_t rowCount, void* buffer) final;

  physicalType nextValue();

  uint32_t nextIndex();

  void resetValues();

  static std::string_view getSerializedLogicalRunValues(
      EncodingSelection<physicalType>& selection,
      const Vector<T>& logicalRunValues,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    static_assert(isFloatingPointType<T>());
    auto nestedPolicy = std::unique_ptr<EncodingSelectionPolicy<T>>(
        static_cast<EncodingSelectionPolicy<T>*>(
            selection
                .template createNestedPolicy<T>(
                    selection.encodingType(),
                    EncodingIdentifiers::RunLength::RunValues)
                .release()));
    return EncodingFactory::encode<T>(
        std::move(nestedPolicy),
        std::span<const T>{logicalRunValues.data(), logicalRunValues.size()},
        buffer,
        options);
  }

  static std::string_view getSerializedPhysicalRunValues(
      EncodingSelection<physicalType>& selection,
      const Vector<physicalType>& runValues,
      Buffer& buffer,
      const Encoding::Options& options = {}) {
    static_assert(!isFloatingPointType<T>());
    return selection.template encodeNested<physicalType>(
        EncodingIdentifiers::RunLength::RunValues, runValues, buffer, options);
  }

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  bool dictionaryEnabled() const override {
    if (dictValues_ != nullptr) {
      return true;
    }
    if (valuesEncoding_ == nullptr) {
      return false;
    }
    return valuesEncoding_->dictionaryEnabled();
  }

  uint32_t dictionarySize() const override {
    if (dictValues_ != nullptr) {
      return dictValues_->dictionarySize();
    }
    NIMBLE_CHECK(
        valuesEncoding_ != nullptr,
        "dictionary metadata unavailable — flat mode was already initialized");
    return valuesEncoding_->dictionarySize();
  }

  const void* dictionaryEntry(uint32_t index) const override {
    return static_cast<const physicalType*>(dictionaryEntries()) + index;
  }

  const void* dictionaryEntries() const override {
    if (dictValues_ != nullptr) {
      return dictValues_->dictionaryEntries();
    }
    NIMBLE_CHECK(
        valuesEncoding_ != nullptr,
        "dictionary metadata unavailable — flat mode was already initialized");
    return valuesEncoding_->dictionaryEntries();
  }

  /// Materializes composed dictionary indices for rowCount dense non-null rows.
  /// For each RLE run, looks up the run value's dictionary index via the
  /// value→index map, then fills the buffer with that index repeated for
  /// the run length.
  void materializeIndices(uint32_t rowCount, uint32_t* buffer) override {
    ensureDictValues();
    uint32_t rowsLeft = rowCount;
    uint32_t numOutputRows = 0;
    while (rowsLeft > 0) {
      if (this->copiesRemaining_ == 0) {
        advanceRunLength();
        advanceRunIndex();
      }
      const auto numRows = std::min(rowsLeft, this->copiesRemaining_);
      velox::simd::simdFill(buffer + numOutputRows, currentIndex_, numRows);
      numOutputRows += numRows;
      rowsLeft -= numRows;
      this->copiesRemaining_ -= numRows;
    }
  }

  /// Reads dictionary indices through an RLE wrapper using the standard
  /// materializeIndices + dense/sparse helper pattern.
  template <typename IndicesVisitor>
  void readIndicesWithVisitor(
      IndicesVisitor& visitor,
      ReadWithVisitorParams& params) {
    ensureDictValues();
    NIMBLE_CHECK(
        !IndicesVisitor::kHasHook,
        "readIndicesWithVisitor does not support value hooks");
    const auto numReadRows =
        visitor.rowAt(visitor.numRows() - 1) - params.numScanned + 1;
    auto* rawNulls = visitor.reader().rawNullsInReadRange();
    const auto numNonNulls = rawNulls != nullptr
        ? velox::bits::countNonNulls(
              rawNulls, params.numScanned, params.numScanned + numReadRows)
        : numReadRows;

    if (IndicesVisitor::dense) {
      NIMBLE_CHECK_EQ(
          visitor.rowAt(visitor.numRows() - 1),
          visitor.rowAt(0) + visitor.numRows() - 1,
          "Dense visitor must have contiguous rows");
      detail::readDenseMaterializedIndices(
          *this, visitor, params, rawNulls, numReadRows, numNonNulls);
      return;
    }

    auto* rawIndices = ensureIndicesBuffer(numNonNulls);
    detail::readSparseMaterializedIndices(
        *this,
        visitor,
        params.numScanned,
        params.prepareResultNulls,
        rawNulls,
        numReadRows,
        numNonNulls,
        rawIndices);
  }

  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {}) {
    // Estimate the two nested streams produced by RLE:
    //
    //   run lengths: one length per consecutive run, estimated as
    //     FixedBitWidth<uint32_t> over [minRepeat, maxRepeat].
    //   run values: one value per consecutive run. Numeric values are
    //     estimated as FixedBitWidth over the original value range. String
    //     values are estimated as Dictionary over the run values.
    const uint64_t runCount = statistics.consecutiveRepeatCount();
    // Run lengths are encoded as a FixedBitWidth child.
    const uint64_t runLengthsEncodingSize =
        FixedBitWidthEncoding<uint32_t>::estimateSize(
            runCount, statistics.minRepeat(), statistics.maxRepeat(), options);

    const uint64_t runValuesEncodingSize =
        estimateRunValuesSize(runCount, statistics, options);
    const uint64_t outerEncodingSize =
        EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
    return outerEncodingSize + runValuesEncodingSize + runLengthsEncodingSize;
  }

 private:
  friend class internal::RLEEncodingBase<T, RLEEncoding<T>>;

  static std::string_view sliceRunValues(
      std::string_view encodedRunValues,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options) {
    return EncodingFactory::slice(
        encodedRunValues, offset, length, buffer, options);
  }

  static uint64_t estimateRunValuesSize(
      uint64_t runCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options) {
    if constexpr (isStringType<physicalType>()) {
      return DictionaryEncoding<std::string_view>::estimateSize(
          runCount, statistics, options);
    } else {
      uint64_t bestSize = FixedBitWidthEncoding<physicalType>::estimateSize(
          runCount, statistics, options);
      if constexpr (isFloatingPointType<T>()) {
        if (options.allowNestedAlpSelection) {
          const auto& runValues = statistics.runValues();
          if (const auto alpEncodingSize = detail::nestedAlpSize<T>(
                  std::span<const physicalType>{
                      runValues.data(), runValues.size()},
                  options)) {
            bestSize = std::min(bestSize, *alpEncodingSize);
          }
        }
      }
      return bestSize;
    }
  }

  using internal::RLEEncodingBase<T, RLEEncoding<T>>::advanceRunLength;

  void advanceRunValue();

  void ensureValues() {
    NIMBLE_CHECK_NULL(
        dictValues_,
        "flat mode unavailable — dict mode was already initialized");
    if (values_) {
      return;
    }
    NIMBLE_CHECK_NOT_NULL(valuesEncoding_);
    values_ = std::make_unique<detail::BufferedEncoding<physicalType, 128>>(
        std::move(valuesEncoding_));
    for (uint32_t i = 0; i < pendingSkips_; ++i) {
      this->currentValue_ = values_->nextValue();
    }
    pendingSkips_ = 0;
  }

  void ensureDictValues() {
    NIMBLE_CHECK_NULL(
        values_, "dict mode unavailable — flat mode was already initialized");
    if (dictValues_) {
      return;
    }
    NIMBLE_CHECK_NOT_NULL(valuesEncoding_);
    NIMBLE_CHECK(
        valuesEncoding_->dictionaryEnabled(),
        "dict mode unavailable — inner encoding is not dictionary-compatible");
    dictValues_ =
        std::make_unique<detail::BufferedDictEncoding<physicalType, 128>>(
            std::move(valuesEncoding_));
    alphabet_ =
        static_cast<const physicalType*>(dictValues_->dictionaryEntries());
    for (uint32_t i = 0; i < pendingSkips_; ++i) {
      currentIndex_ = dictValues_->nextIndex();
    }
    pendingSkips_ = 0;
  }

  uint32_t advanceRunIndex() {
    currentIndex_ = dictValues_->nextIndex();
    return currentIndex_;
  }

  template <bool kDense>
  vector_size_t findNumInRun(
      const vector_size_t* rows,
      vector_size_t rowIndex,
      vector_size_t numRows,
      vector_size_t currentRow) const;

  uint32_t* ensureIndicesBuffer(uint32_t numElements) {
    const auto bytes = numElements * sizeof(uint32_t);
    if (indicesBuffer_ == nullptr || indicesBuffer_->capacity() < bytes) {
      indicesBuffer_ = this->getBuffer(bytes);
    }
    return indicesBuffer_->asMutable<uint32_t>();
  }

  // Exactly one of valuesEncoding_, values_, or dictValues_ is non-null.
  // valuesEncoding_ holds the raw inner encoding until the first read chooses
  // flat vs dict mode, then it is moved into values_ or dictValues_.
  // When the inner encoding is dictionary-enabled, dictValues_ loads indices
  // in pages and reconstructs values from the alphabet. Otherwise values_
  // loads values directly via materialize().
  std::unique_ptr<Encoding> valuesEncoding_;
  uint32_t pendingSkips_{0};
  std::unique_ptr<detail::BufferedEncoding<physicalType, 128>> values_;
  std::unique_ptr<detail::BufferedDictEncoding<physicalType, 128>> dictValues_;
  const physicalType* alphabet_{nullptr};
  velox::BufferPtr indicesBuffer_;
  // Dict-mode current run's dictionary index. Kept as a member (mirroring
  // currentValue_ for the non-dict path) so a materialize that resumes mid-run
  // across a batch boundary reuses the in-progress run's index instead of
  // restarting from 0.
  uint32_t currentIndex_{0};
};

// For the bool case we know the values will alternate between true
// and false, so in addition to the run lengths we need only store
// whether the first value is true or false.
// RLEEncodingBase bytes
// 1 byte: whether first row is true
template <>
class RLEEncoding<bool> final
    : public internal::RLEEncodingBase<bool, RLEEncoding<bool>> {
 public:
  RLEEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  bool nextValue();
  void resetValues();
  static std::string_view getSerializedPhysicalRunValues(
      EncodingSelection<bool>& /* selection */,
      const Vector<bool>& runValues,
      Buffer& buffer,
      const Encoding::Options& /* options */ = {}) {
    char* reserved = buffer.reserve(sizeof(char));
    *reserved = runValues[0];
    return {reserved, 1};
  }

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final;

  static uint64_t estimateSize(
      uint64_t /*rowCount*/,
      const Statistics<bool>& statistics,
      const Encoding::Options& options = {}) {
    // Assumptions:
    // Run lengths are stored using bit-packing (with bit width
    // needed to store max repetition count).
    const uint64_t runCount = statistics.consecutiveRepeatCount();
    const uint64_t initialValueSize = sizeof(bool);
    // Run lengths are encoded as a FixedBitWidth child.
    const uint64_t runLengthsEncodingSize =
        FixedBitWidthEncoding<uint32_t>::estimateSize(
            runCount, statistics.minRepeat(), statistics.maxRepeat(), options);
    const uint64_t outerEncodingSize =
        EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
    return outerEncodingSize + initialValueSize + runLengthsEncodingSize;
  }

 private:
  friend class internal::RLEEncodingBase<bool, RLEEncoding<bool>>;

  static std::string_view sliceRunValues(
      std::string_view encodedRunValues,
      uint32_t offset,
      uint32_t /* length */,
      Buffer& buffer,
      const Encoding::Options& /* options */) {
    const auto initialValue =
        *reinterpret_cast<const bool*>(encodedRunValues.data());
    const auto slicedInitialValue =
        offset % 2 == 0 ? initialValue : !initialValue;
    char* reserved = buffer.reserve(sizeof(bool));
    *reserved = slicedInitialValue;
    return {reserved, sizeof(bool)};
  }

  bool initialValue_;
  bool value_;
};

//
// End of class declaration. Implementations follow.
//

template <typename T>
RLEEncoding<T>::RLEEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    const std::function<void*(uint32_t)>& stringBufferFactory,
    const Encoding::Options& options)
    : internal::RLEEncodingBase<T, RLEEncoding<T>>(
          pool,
          data,
          stringBufferFactory,
          options) {
  auto valuesView = std::string_view{
      internal::RLEEncodingBase<T, RLEEncoding<T>>::getValuesStart(),
      static_cast<size_t>(
          data.end() -
          internal::RLEEncodingBase<T, RLEEncoding<T>>::getValuesStart())};
  valuesEncoding_ =
      EncodingFactory(options).create(pool, valuesView, stringBufferFactory);
  if (!isStringType<physicalType>() || !valuesEncoding_->dictionaryEnabled()) {
    ensureValues();
  }
  this->reset();
}

// Advances to the next run value in flat (non-dict) mode by setting
// currentValue_ from the values encoding. In dict mode,
// advanceRunIndex() is used instead.
template <typename T>
void RLEEncoding<T>::advanceRunValue() {
  ensureValues();
  this->currentValue_ = values_->nextValue();
}

template <typename T>
void RLEEncoding<T>::reset() {
  // Delegate to base — copiesRemaining_ starts at 0. The first call to
  // materialize/readWithVisitor/bulkScan/materializeIndices will load
  // the first run on demand.
  internal::RLEEncodingBase<T, RLEEncoding<T>>::reset();
}

template <typename T>
void RLEEncoding<T>::skip(uint32_t rowCount) {
  uint32_t rowsLeft = rowCount;
  while (rowsLeft > 0) {
    if (rowsLeft <= this->copiesRemaining_) {
      this->copiesRemaining_ -= rowsLeft;
      return;
    }
    rowsLeft -= this->copiesRemaining_;
    advanceRunLength();
    if (valuesEncoding_ != nullptr) {
      // Mode not yet chosen — defer the value consumption until
      // ensureValues() or ensureDictValues() wraps the encoding.
      ++pendingSkips_;
    } else if (dictValues_ != nullptr) {
      advanceRunIndex();
    } else {
      advanceRunValue();
    }
  }
}

template <typename T>
void RLEEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  ensureValues();
  uint32_t rowsLeft = rowCount;
  auto* output = static_cast<physicalType*>(buffer);
  while (rowsLeft > 0) {
    if (this->copiesRemaining_ == 0) {
      this->advanceRun();
    }
    if (rowsLeft < this->copiesRemaining_) {
      velox::simd::simdFill(output, this->currentValue_, rowsLeft);
      this->copiesRemaining_ -= rowsLeft;
      return;
    }
    velox::simd::simdFill(output, this->currentValue_, this->copiesRemaining_);
    output += this->copiesRemaining_;
    rowsLeft -= this->copiesRemaining_;
    this->copiesRemaining_ = 0;
  }
}

template <typename T>
typename RLEEncoding<T>::physicalType RLEEncoding<T>::nextValue() {
  ensureValues();
  return values_->nextValue();
}

template <typename T>
uint32_t RLEEncoding<T>::nextIndex() {
  ensureDictValues();
  return dictValues_->nextIndex();
}

template <typename T>
void RLEEncoding<T>::resetValues() {
  if (dictValues_ != nullptr) {
    dictValues_->reset();
  } else if (values_ != nullptr) {
    values_->reset();
  } else if (valuesEncoding_ != nullptr) {
    valuesEncoding_->reset();
    pendingSkips_ = 0;
  }
}

template <typename T>
template <bool kDense>
vector_size_t RLEEncoding<T>::findNumInRun(
    const vector_size_t* rows,
    vector_size_t rowIndex,
    vector_size_t numRows,
    vector_size_t currentRow) const {
  if constexpr (kDense) {
    return std::min<vector_size_t>(this->copiesRemaining_, numRows - rowIndex);
  }
  if (rows[rowIndex] - currentRow >= this->copiesRemaining_) {
    // Skip this run.
    return 0;
  }
  if (rows[numRows - 1] - currentRow < this->copiesRemaining_) {
    return numRows - rowIndex;
  }
  auto* begin = rows + rowIndex;
  auto* end = begin +
      std::min<vector_size_t>(this->copiesRemaining_, numRows - rowIndex);
  auto endOfRun = currentRow + this->copiesRemaining_;
  auto* it = std::lower_bound(begin, end, endOfRun);
  NIMBLE_DCHECK(it > begin);
  return it - begin;
}

template <typename T>
template <bool kScatter, typename V>
void RLEEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  ensureValues();
  using DataType = typename V::DataType;
  using ValueType = detail::ValueType<DataType>;
  constexpr bool kScatterValues = kScatter && !V::kHasFilter && !V::kHasHook;
  auto* values = detail::mutableValues<ValueType>(
      visitor, visitor.numRows() - visitor.rowIndex());
  auto* filterHits = V::kHasFilter ? visitor.outputRows(numSelected) : nullptr;
  auto* rows = kScatter ? scatterRows : selectedRows;
  vector_size_t numValues = 0;
  vector_size_t numHits = 0;
  vector_size_t selectedRowIndex = 0;
  for (;;) {
    if (this->copiesRemaining_ == 0) {
      advanceRunLength();
      advanceRunValue();
    }
    const auto numInRun = findNumInRun<V::dense>(
        selectedRows, selectedRowIndex, numSelected, currentRow);
    if (numInRun > 0) {
      auto value = detail::castFromPhysicalType<DataType>(this->currentValue_);
      bool pass = true;
      if constexpr (V::kHasFilter) {
        pass = velox::common::applyFilter(visitor.filter(), value);
        if (pass) {
          auto* begin = rows + selectedRowIndex;
          std::copy(begin, begin + numInRun, filterHits + numHits);
          numHits += numInRun;
        }
      }
      if (!V::kFilterOnly && pass) {
        vector_size_t numRows;
        if constexpr (kScatterValues) {
          auto end = selectedRowIndex + numInRun;
          if (FOLLY_UNLIKELY(end == numSelected)) {
            numRows = visitor.numRows() - visitor.rowIndex();
          } else {
            numRows = scatterRows[end] - visitor.rowIndex();
          }
          visitor.addRowIndex(numRows);
        } else {
          numRows = numInRun;
        }
        auto* begin = values + numValues;
        velox::simd::simdFill(
            begin,
            detail::dataToValue(visitor, value),
            static_cast<uint32_t>(numRows));
        numValues += numRows;
      }
      auto endRow = selectedRows[selectedRowIndex + numInRun - 1];
      auto consumed = endRow - currentRow + 1;
      consumed = std::min<vector_size_t>(consumed, this->copiesRemaining_);
      this->copiesRemaining_ -= consumed;
      currentRow += consumed;
      selectedRowIndex += numInRun;
    }
    if (FOLLY_UNLIKELY(selectedRowIndex == numSelected)) {
      break;
    }
    currentRow += this->copiesRemaining_;
    this->copiesRemaining_ = 0;
  }
  if constexpr (kScatterValues) {
    NIMBLE_DCHECK_EQ(visitor.rowIndex(), visitor.numRows(), "");
  } else {
    visitor.setRowIndex(visitor.numRows());
  }
  if constexpr (V::kHasHook) {
    NIMBLE_DCHECK_EQ(numValues, numSelected, "");
    visitor.hook().addValues(scatterRows, values, numSelected);
  } else {
    visitor.addNumValues(V::kFilterOnly ? numHits : numValues);
  }
}

template <typename T>
template <typename V>
void RLEEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  ensureValues();
  auto* nulls = visitor.reader().rawNullsInReadRange();
  if (velox::dwio::common::useFastPath(visitor, nulls)) {
    detail::readWithVisitorFast(*this, visitor, params, nulls);
  } else {
    detail::readWithVisitorSlow(
        visitor,
        params,
        [&](auto toSkip) { this->skip(toSkip); },
        [&] {
          if (this->copiesRemaining_ == 0) {
            advanceRunLength();
            advanceRunValue();
          }
          --this->copiesRemaining_;
          return this->currentValue_;
        });
  }
}

} // namespace facebook::nimble
