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
#include <limits>
#include <span>
#include <utility>
#include <vector>
#include "folly/container/F14Map.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/ALPEncoding.h"
#include "velox/dwio/nimble/encodings/ALPRDEncoding.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/velox/selective/NimbleReaderFuzzerStats.h"

// A dictionary encoded stream is comprised of two pieces: a mapping from the
// n unique values in a stream to the integers [0, n) and the vector of indices
// that scatter those uniques back to the original ordering.

namespace facebook::nimble {

/// The layout for a dictionary encoding is:
/// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
/// 4 bytes: alphabet size
/// XX bytes: alphabet encoding bytes
/// YY bytes: indices encoding bytes
template <typename T>
class DictionaryEncoding
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  DictionaryEncoding(
      velox::memory::MemoryPool& pool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory,
      const Encoding::Options& options = {});

  ~DictionaryEncoding() override {
    this->releaseBuffer(indicesBuffer_);
  }

  DictionaryEncoding(const DictionaryEncoding&) = delete;
  DictionaryEncoding& operator=(const DictionaryEncoding&) = delete;
  DictionaryEncoding(DictionaryEncoding&&) = delete;
  DictionaryEncoding& operator=(DictionaryEncoding&&) = delete;

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  /// Reads dictionary indices only, without looking up values from the
  /// alphabet. This is used for dictionary-aware readers that return
  /// DictionaryVector. Non-legacy encodings only.
  template <typename V>
  void readIndicesWithVisitor(V& visitor, ReadWithVisitorParams& params);

  void materializeIndices(uint32_t rowCount, uint32_t* buffer) override {
    indicesEncoding_->materialize(rowCount, buffer);
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer,
      const Encoding::Options& options = {});

  static std::string_view slice(
      std::string_view encoded,
      uint32_t offset,
      uint32_t length,
      Buffer& buffer,
      const Encoding::Options& options = {});

  /// Estimates dictionary bytes, including configured ALP/ALPRD alphabet
  /// candidates for floating-point logical types.
  static uint64_t estimateSize(
      uint64_t rowCount,
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options = {},
      std::span<const std::pair<EncodingType, float>>
          alphabetEncodingReadFactors = {}) {
    // Estimate the two nested streams produced by Dictionary:
    //
    //   alphabet: unique values, estimated as Trivial or FixedBitWidth, plus
    //             configured ALP/ALPRD for floating-point logical types.
    //   indices: one dictionary index per row, estimated as FixedBitWidth.
    const auto& uniqueCounts = statistics.uniqueCounts().value();
    const uint64_t uniqueCount = uniqueCounts.size();
    const uint64_t indicesEncodingSize =
        FixedBitWidthEncoding<uint32_t>::estimateSize(
            rowCount,
            /*minValue=*/0,
            uniqueCount == 0 ? 0 : uniqueCount - 1,
            options);

    const auto alphabetEncodingSize =
        estimateAlphabetSize(statistics, options, alphabetEncodingReadFactors);
    const uint64_t outerEncodingSize =
        EncodingPrefix::kFixedPrefixSize + sizeof(uint32_t);
    return outerEncodingSize + alphabetEncodingSize + indicesEncodingSize;
  }

  std::string debugString(int offset) const final;

  bool dictionaryEnabled() const override {
    return true;
  }

  uint32_t dictionarySize() const override {
    return alphabet_.size();
  }

  const void* dictionaryEntry(uint32_t index) const override {
    return &alphabet_[index];
  }

  const void* dictionaryEntries() const override {
    return alphabet_.data();
  }

  const Encoding* indicesEncoding() const {
    return indicesEncoding_.get();
  }

 private:
  // Uses unique-value statistics so repeated input values do not skew sampling.
  static uint64_t estimateAlphabetSize(
      const Statistics<physicalType>& statistics,
      const Encoding::Options& options,
      std::span<const std::pair<EncodingType, float>>
          alphabetEncodingReadFactors) {
    const auto& uniqueCounts = statistics.uniqueCounts().value();
    const uint64_t uniqueCount = uniqueCounts.size();
    if constexpr (isStringType<physicalType>()) {
      return TrivialEncoding<std::string_view>::estimateSize(
          uniqueCount,
          uniqueCounts.uniqueStringBytes(),
          statistics.min().size(),
          statistics.max().size(),
          options);
    } else {
      auto alphabetEncodingSize = std::min(
          TrivialEncoding<physicalType>::estimateSize(uniqueCount),
          FixedBitWidthEncoding<physicalType>::estimateSize(
              uniqueCount, statistics.min(), statistics.max(), options));
      if constexpr (isFloatingPointType<T>()) {
        if (uniqueCount == 0) {
          return alphabetEncodingSize;
        }
        std::vector<physicalType> sampledValues;
        for (const auto& candidate : alphabetEncodingReadFactors) {
          const auto encodingType = candidate.first;
          if (encodingType != EncodingType::ALP &&
              encodingType != EncodingType::ALPRD) {
            continue;
          }
          if (sampledValues.empty()) {
            NIMBLE_CHECK_LE(uniqueCount, std::numeric_limits<uint32_t>::max());
            const auto sampleSize =
                ALPEncoding<T>::estimateSampleSize(uniqueCount);
            sampledValues.reserve(sampleSize);
            for (const auto& [value, count] : uniqueCounts) {
              sampledValues.push_back(value);
              if (sampledValues.size() == sampleSize) {
                break;
              }
            }
          }
          const auto estimate = encodingType == EncodingType::ALP
              ? ALPEncoding<T>::estimateSizeFromSample(
                    uniqueCount, sampledValues, options)
              : ALPRDEncodingBase::estimateSize<physicalType>(
                    sampledValues, static_cast<uint32_t>(uniqueCount), options);
          if (estimate.has_value()) {
            alphabetEncodingSize =
                std::min(alphabetEncodingSize, estimate.value());
          }
        }
      }
      // TODO: Consider PFOR after its statistics can be derived from unique
      // values without rebuilding Statistics for every dictionary estimate.
      return alphabetEncodingSize;
    }
  }

  static std::string_view encodeAlphabet(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> alphabet,
      Buffer& buffer,
      const Encoding::Options& options);

  uint32_t* ensureIndicesBuffer(uint32_t numElements) {
    const auto bytes = numElements * sizeof(uint32_t);
    if (indicesBuffer_ == nullptr || indicesBuffer_->capacity() < bytes) {
      indicesBuffer_ = this->getBuffer(bytes);
    }
    return indicesBuffer_->asMutable<uint32_t>();
  }

  // Resets the filter verdict cache to kUnknown for the current alphabet,
  // resizing if the alphabet size changed.
  void ensureFilterCache(velox::dwio::common::ScanState& scanState) {
    auto& filterCache = scanState.filterCache;
    const auto size = std::max<size_t>(1, alphabet_.size());
    filterCache.resize(size);
    std::fill(
        filterCache.begin(),
        filterCache.end(),
        velox::dwio::common::FilterResult::kUnknown);
    scanState.updateRawState();
    filterCacheInitialized_ = true;
  }

  Vector<T> alphabet_;
  std::unique_ptr<Encoding> indicesEncoding_;
  std::unique_ptr<Encoding> alphabetEncoding_;
  velox::BufferPtr indicesBuffer_;

  // True once this encoding has reset the reader's verdict cache for its
  // alphabet. SelectiveColumnReader::resetFilterCaches() invalidates the
  // verdicts when the filter changes.
  bool filterCacheInitialized_{false};
};

//
// End of public API. Implementation follows.
//

template <typename T>
DictionaryEncoding<T>::DictionaryEncoding(
    velox::memory::MemoryPool& pool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory,
    const Encoding::Options& options)
    : TypedEncoding<T, physicalType>{pool, data, options},
      alphabet_{this->pool_} {
  const auto* pos = data.data() + this->dataOffset();
  const uint32_t alphabetSize = encoding::readUint32(pos);
  alphabetEncoding_ = EncodingFactory().create(
      *this->pool_, {pos, alphabetSize}, stringBufferFactory, options);
  const uint32_t alphabetCount = alphabetEncoding_->rowCount();
  alphabet_.resize(alphabetCount);
  alphabetEncoding_->materialize(alphabetCount, alphabet_.data());

  pos += alphabetSize;
  indicesEncoding_ = EncodingFactory().create(
      *this->pool_,
      {pos, static_cast<size_t>(data.data() + data.size() - pos)},
      stringBufferFactory,
      options);
}

template <typename T>
void DictionaryEncoding<T>::reset() {
  indicesEncoding_->reset();
}

template <typename T>
void DictionaryEncoding<T>::skip(uint32_t rowCount) {
  indicesEncoding_->skip(rowCount);
}

template <typename T>
void DictionaryEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  auto* rawIndices = ensureIndicesBuffer(rowCount);
  indicesEncoding_->materialize(rowCount, rawIndices);

  T* output = static_cast<T*>(buffer);
  for (uint32_t i = 0; i < rowCount; ++i) {
    *output = alphabet_[rawIndices[i]];
    ++output;
  }
}

namespace detail {

class DictionaryIndicesHook : public velox::ValueHook {
 public:
  static constexpr bool kSkipNulls = true;

  DictionaryIndicesHook(uint32_t* indices, vector_size_t offset)
      : indices_(indices), offset_(offset) {}

  bool acceptsNulls() const final {
    return false;
  }

  void addValue(vector_size_t i, int64_t value) final {
    indices_[i - offset_] = value;
  }

  void addValues(
      const vector_size_t* rows,
      const int32_t* values,
      vector_size_t size) final {
    for (vector_size_t i = 0; i < size; ++i) {
      indices_[rows[i] - offset_] = values[i];
    }
  }

  void addNull(vector_size_t /*i*/) final {
    NIMBLE_UNREACHABLE(__PRETTY_FUNCTION__);
  }

 private:
  uint32_t* const indices_;
  const vector_size_t offset_;
};

/// Fuses index decoding with dictionary translation and filtering. Index
/// decoders call the dictionary visitor directly instead of staging indices in
/// a buffer, and index batches go through processBulk().
template <typename DictionaryVisitor>
class FusedDictionaryIndicesHook : public velox::ValueHook {
 public:
  static constexpr bool kSkipNulls = true;
  static constexpr bool kBulkFastPath = true;

  /// Constructs a hook bound to the given dictionary visitor.
  /// 'filterCache' is the visitor's per-index verdict cache. It may be null
  /// for reads without a filter.
  FusedDictionaryIndicesHook(
      DictionaryVisitor& visitor,
      const uint8_t* filterCache)
      : visitor_{visitor}, filterCache_{filterCache} {}

  bool acceptsNulls() const final {
    return false;
  }

  /// Processes a single dictionary index. Handles leading nulls from the
  /// current row to the given row, then translates the index via the visitor.
  void addValue(vector_size_t row, int64_t index) final {
    processNulls(row);
    NIMBLE_DCHECK(
        !visitor_.atEnd(), "Dictionary visitor received an extra index");
    bool atEnd{false};
    visitor_.process(index, atEnd);
    syncFilterOnlyNumValues();
  }

  /// Processes a batch of non-null indices with their requested-row positions.
  /// The visitor handles the non-null values. This hook handles the null gaps
  /// between 'rows': it writes their null bits in projection output and merges
  /// passing nulls into filtered output.
  void addValues(
      const vector_size_t* rows,
      const int32_t* indices,
      vector_size_t size) final {
    if (size == 0) {
      return;
    }
    processNulls(rows[0]);
    const auto startRowIndex = visitor_.rowIndex();
    const auto outputOffset = visitor_.reader().numValues();
    visitor_.processBulk(indices, size, rows);
    const auto numNulls = rows[size - 1] + 1 - startRowIndex - size;
    if (numNulls > 0) {
      if constexpr (DictionaryVisitor::kHasFilter) {
        if (visitor_.allowNulls()) {
          mergeBulkNulls(rows, indices, size, numNulls, outputOffset);
        }
      } else {
        setProjectionNulls(rows, size, startRowIndex, outputOffset);
      }
    }
    syncFilterOnlyNumValues();
  }

  void addNull(vector_size_t /*row*/) final {
    NIMBLE_UNREACHABLE(__PRETTY_FUNCTION__);
  }

  /// Handles nulls at the end of the read range after all indices are
  /// processed.
  void processTrailingNulls() {
    processNulls(visitor_.numRows());
  }

 private:
  // Marks the null gaps between 'rows' in projection output. The visitor has
  // already scattered the values around these gaps.
  void setProjectionNulls(
      const vector_size_t* rows,
      vector_size_t size,
      vector_size_t startRowIndex,
      vector_size_t outputOffset) {
    if constexpr (std::is_same_v<
                      typename DictionaryVisitor::Extract,
                      velox::dwio::common::ExtractToReader>) {
      if (visitor_.reader().returnReaderNulls()) {
        return;
      }
      auto* nulls = visitor_.rawNulls(0);
      velox::bits::fillBits(
          nulls,
          outputOffset,
          visitor_.reader().numValues(),
          velox::bits::kNotNull);
      for (vector_size_t position = 1; position < size; ++position) {
        const auto nullBegin = rows[position - 1] + 1;
        const auto nullEnd = rows[position];
        if (nullBegin == nullEnd) {
          continue;
        }
        velox::bits::fillBits(
            nulls,
            outputOffset + nullBegin - startRowIndex,
            outputOffset + nullEnd - startRowIndex,
            velox::bits::kNull);
      }
      visitor_.setHasNulls();
    }
  }

  // Inserts the null rows between 'rows' into the filtered output. Walks
  // backward because the inserted nulls shift passing values to the right.
  void mergeBulkNulls(
      const vector_size_t* rows,
      const int32_t* indices,
      vector_size_t size,
      vector_size_t numNulls,
      vector_size_t outputOffset) {
    using T = typename DictionaryVisitor::DataType;
    const auto numValues = visitor_.reader().numValues();
    auto* filterHits = visitor_.outputRows(numNulls) - numValues;
    auto* values = reinterpret_cast<T*>(visitor_.reader().rawValues());
    uint64_t* nulls{nullptr};
    if constexpr (!DictionaryVisitor::kFilterOnly) {
      nulls = visitor_.rawNulls(numNulls);
      velox::bits::fillBits(
          nulls, outputOffset, numValues + numNulls, velox::bits::kNotNull);
      visitor_.setHasNulls();
    }
    auto passingPosition = numValues - 1;
    auto outputPosition = numValues + numNulls - 1;

    if (numNulls * 4 >= size) {
      // Dense nulls leave few passing rows between gaps, in counts a branch
      // cannot predict. Visit every non-null row instead, and take its verdict
      // from the filter cache, which processBulk() filled for every index in
      // 'indices'. Each row copies the current passing value and moves the
      // cursors only if it passed, so no branch depends on the filter result.
      // The loop stops once every null is placed, because the passing values
      // below are already in their final slots.
      for (auto position = size - 1; outputPosition > passingPosition;
           --position) {
        NIMBLE_DCHECK_GT(position, 0);
        const bool passed = filterCache_[indices[position]] ==
            velox::dwio::common::FilterResult::kSuccess;

        // Slots above 'passingPosition' are free, so the copy is harmless
        // when the row failed the filter.
        const auto source = std::max(passingPosition, outputOffset);
        NIMBLE_DCHECK(
            !passed || filterHits[source] == visitor_.rowAt(rows[position]),
            "Filter cache disagrees with the passing rows");
        filterHits[outputPosition] = filterHits[source];
        if constexpr (!DictionaryVisitor::kFilterOnly) {
          values[outputPosition] = values[source];
        }
        passingPosition -= passed;
        outputPosition -= passed;

        const auto nullBegin = rows[position - 1] + 1;
        for (auto nullPosition = rows[position]; nullPosition-- > nullBegin;) {
          filterHits[outputPosition] = visitor_.rowAt(nullPosition);
          if constexpr (!DictionaryVisitor::kFilterOnly) {
            velox::bits::setNull(nulls, outputPosition);
          }
          --outputPosition;
        }
      }
    } else {
      // Sparse nulls leave long runs between gaps, so jump from gap to gap and
      // move the passing values in between. The branches mispredict only
      // around the gaps, which are rare.
      for (auto position = size - 1; position > 0; --position) {
        const auto nullBegin = rows[position - 1] + 1;
        const auto nullEnd = rows[position];
        if (nullBegin == nullEnd) {
          continue;
        }
        const auto firstNullRow = visitor_.rowAt(nullBegin);
        while (passingPosition >= outputOffset &&
               filterHits[passingPosition] > firstNullRow) {
          filterHits[outputPosition] = filterHits[passingPosition];
          if constexpr (!DictionaryVisitor::kFilterOnly) {
            values[outputPosition] = values[passingPosition];
          }
          --passingPosition;
          --outputPosition;
        }
        for (auto nullPosition = nullEnd; nullPosition-- > nullBegin;) {
          filterHits[outputPosition] = visitor_.rowAt(nullPosition);
          if constexpr (!DictionaryVisitor::kFilterOnly) {
            velox::bits::setNull(nulls, outputPosition);
          }
          --outputPosition;
        }
      }
    }
    NIMBLE_DCHECK_EQ(outputPosition, passingPosition);
    visitor_.addNumValues(numNulls);
  }

  // Handles nulls from the current visitor row to the given row. For
  // filter-only reads, records null rows. For projection reads, outputs
  // nulls or skips them depending on the visitor's extraction mode.
  void processNulls(vector_size_t row) {
    NIMBLE_DCHECK_LE(visitor_.rowIndex(), row);
    NIMBLE_DCHECK_LE(row, visitor_.numRows());
    const auto numNulls = row - visitor_.rowIndex();
    if (!visitor_.allowNulls()) {
      visitor_.addRowIndex(numNulls);
      return;
    }
    if constexpr (std::is_same_v<
                      typename DictionaryVisitor::Extract,
                      velox::dwio::common::ExtractToReader>) {
      if (visitor_.reader().returnReaderNulls()) {
        visitor_.addRowIndex(numNulls);
        visitor_.addNumValues(numNulls);
        return;
      }
    }
    bool atEnd{false};
    while (visitor_.rowIndex() < row) {
      visitor_.processNull(atEnd);
    }
    syncFilterOnlyNumValues();
  }

  void syncFilterOnlyNumValues() {
    if constexpr (DictionaryVisitor::kFilterOnly) {
      visitor_.reader().setNumValues(visitor_.reader().outputRows().size());
    }
  }

  // Visitor that translates and filters the forwarded indices and owns the
  // output.
  DictionaryVisitor& visitor_;

  // Verdict per dictionary index, filled by the visitor's processBulk().
  const uint8_t* filterCache_;
};

} // namespace detail

template <typename T>
template <typename V>
void DictionaryEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  // Decode indices with an AlwaysTrue visitor and forward them directly into
  // the shared logical-value dictionary visitor.
  if (params.dictionaryAwareReads) {
    if constexpr (
        isNumericType<T>() && sizeof(T) >= sizeof(int16_t) && !V::kHasHook &&
        V::FilterType::deterministic && V::kHasBulkPath) {
      using DictionaryVisitor = velox::dwio::common::DictionaryColumnVisitor<
          T,
          typename V::FilterType,
          typename V::Extract,
          V::dense>;

      auto& scanState = visitor.reader().scanState();

      // Projection-only reads never consult the verdict cache.
      if constexpr (V::kHasFilter) {
        const auto filterCacheSize = std::max<size_t>(1, alphabet_.size());
        if (!filterCacheInitialized_ ||
            scanState.filterCache.size() != filterCacheSize) {
          ensureFilterCache(scanState);
        }
      }
      velox::dwio::common::RawScanState dictionaryState{};
      dictionaryState.dictionary.values = alphabet_.data();
      NIMBLE_CHECK_LE(
          alphabet_.size(),
          static_cast<size_t>(std::numeric_limits<int32_t>::max()),
          "Dictionary size exceeds the supported index range");
      dictionaryState.dictionary.numValues =
          static_cast<int32_t>(alphabet_.size());
      dictionaryState.filterCache = scanState.filterCache.data();

      const auto numRows = visitor.numRows() - visitor.rowIndex();
      const auto outputCapacity = visitor.reader().numValues() + numRows;
      const auto requiredBytes = std::max(
          outputCapacity * sizeof(int32_t),
          outputCapacity * sizeof(T) + numRows * sizeof(int32_t));
      visitor.reader().template ensureValuesCapacity<uint8_t>(
          requiredBytes, /*preserveData=*/true);

      DictionaryVisitor dictionaryVisitor{
          visitor.filter(),
          &visitor.reader(),
          velox::RowSet(visitor.rows(), visitor.numRows()),
          visitor.extractValues(),
          dictionaryState};
      dictionaryVisitor.setRowIndex(visitor.rowIndex());
      dictionaryVisitor.setNumValuesBias(visitor.numValuesBias());

      if constexpr (std::is_same_v<
                        typename V::Extract,
                        velox::dwio::common::ExtractToReader>) {
        params.prepareResultNulls();
      }

      using Hook = detail::FusedDictionaryIndicesHook<DictionaryVisitor>;
      Hook dictionaryHook{dictionaryVisitor, dictionaryState.filterCache};
      velox::common::AlwaysTrue indicesFilter;
      DecoderVisitor<
          int32_t,
          velox::common::AlwaysTrue,
          velox::dwio::common::ExtractToHook<Hook>,
          V::dense>
          indicesVisitor{
              indicesFilter,
              &visitor.reader(),
              velox::RowSet(visitor.rows(), visitor.numRows()),
              velox::dwio::common::ExtractToHook<Hook>(&dictionaryHook)};
      indicesVisitor.setRowIndex(visitor.rowIndex());
      callReadWithVisitor(*indicesEncoding_, indicesVisitor, params);
      dictionaryHook.processTrailingNulls();
      visitor.setRowIndex(dictionaryVisitor.rowIndex());
      if constexpr (std::is_same_v<T, int16_t>) {
        fuzzer::updateSmallintDictionaryAwareReads();
      } else if constexpr (std::is_same_v<T, int32_t>) {
        fuzzer::updateIntegerDictionaryAwareReads();
      } else if constexpr (std::is_same_v<T, int64_t>) {
        fuzzer::updateBigintDictionaryAwareReads();
      } else if constexpr (std::is_same_v<T, float>) {
        fuzzer::updateRealDictionaryAwareReads();
      } else if constexpr (std::is_same_v<T, double>) {
        fuzzer::updateDoubleDictionaryAwareReads();
      }
      return;
    }
  }

  if constexpr (sizeof(T) < sizeof(uint32_t)) {
    // Although we do not use column reader values buffer in this decoder, the
    // fast path in nested decoders of indices can use it, so need to ensure it
    // can hold the indices.
    visitor.reader().template ensureValuesCapacity<uint32_t>(
        visitor.numRows(), true);
  }
  const auto startRowIndex = visitor.rowIndex();
  const auto numIndices = visitor.numRows() - startRowIndex;
  auto* rawIndices = ensureIndicesBuffer(numIndices);
  velox::common::AlwaysTrue indicesFilter;
  detail::DictionaryIndicesHook indicesHook(rawIndices, startRowIndex);
  auto indicesVisitor = DecoderVisitor<
      int32_t,
      velox::common::AlwaysTrue,
      velox::dwio::common::ExtractToHook<detail::DictionaryIndicesHook>,
      V::dense>(
      indicesFilter,
      &visitor.reader(),
      velox::RowSet(visitor.rows(), visitor.numRows()),
      velox::dwio::common::ExtractToHook<detail::DictionaryIndicesHook>(
          &indicesHook));
  indicesVisitor.setRowIndex(startRowIndex);
  callReadWithVisitor(*indicesEncoding_, indicesVisitor, params);
  detail::readWithVisitorSlow(visitor, params, nullptr, [&] {
    return alphabet_[rawIndices[visitor.rowIndex() - startRowIndex]];
  });
}

/// Reads indices only without looking up values from the alphabet.
/// This is used for dictionary-aware readers that return DictionaryVector.
/// Uses materializeIndices to avoid recursing through the visitor machinery,
/// which would incorrectly call addNumValues from the inner encoding.
template <typename T>
template <typename V>
void DictionaryEncoding<T>::readIndicesWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  NIMBLE_CHECK(this->dictionaryEnabled());
  NIMBLE_CHECK(
      !V::kHasHook, "readIndicesWithVisitor does not support value hooks");
  const auto numReadRows =
      visitor.rowAt(visitor.numRows() - 1) - params.numScanned + 1;
  auto* rawNulls = visitor.reader().rawNullsInReadRange();
  const auto numNonNulls = rawNulls != nullptr
      ? velox::bits::countNonNulls(
            rawNulls, params.numScanned, params.numScanned + numReadRows)
      : numReadRows;

  // Dense fast path: materialize directly into rawValues_, scatter for nulls.
  if (V::dense) {
    NIMBLE_CHECK_EQ(
        visitor.rowAt(visitor.numRows() - 1),
        visitor.rowAt(0) + visitor.numRows() - 1,
        "Dense visitor must have contiguous rows");
    detail::readDenseMaterializedIndices(
        *this, visitor, params, rawNulls, numReadRows, numNonNulls);
    return;
  }

  // Sparse path: materialize all rows into buffer, gather only the
  // requested positions into rawValues_.
  // TODO: Use FBW's random-access fixedBitArray_.get() to avoid
  // materializing unrequested rows. Requires templated
  // materializeIndices — see commented-out API mock-ups in this file
  // and EncodingUtils.h.
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

template <typename T>
std::string_view DictionaryEncoding<T>::encodeAlphabet(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> alphabet,
    Buffer& buffer,
    const Encoding::Options& options) {
  if constexpr (isFloatingPointType<T>()) {
    ScopedVector<T> logicalAlphabet{
        alphabet.size(), &buffer.getMemoryPool(), options.bufferPool};
    for (uint32_t i = 0; i < alphabet.size(); ++i) {
      logicalAlphabet[i] =
          EncodingPhysicalType<T>::asEncodingLogicalType(alphabet[i]);
    }

    auto nestedPolicy = std::unique_ptr<EncodingSelectionPolicy<T>>(
        static_cast<EncodingSelectionPolicy<T>*>(
            selection
                .template createNestedPolicy<T>(
                    selection.encodingType(),
                    EncodingIdentifiers::Dictionary::Alphabet)
                .release()));
    return EncodingFactory::encode<T>(
        std::move(nestedPolicy),
        std::span<const T>{logicalAlphabet.data(), logicalAlphabet.size()},
        buffer,
        options);
  } else {
    return selection.template encodeNested<physicalType>(
        EncodingIdentifiers::Dictionary::Alphabet, {alphabet}, buffer, options);
  }
}

template <typename T>
std::string_view DictionaryEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer,
    const Encoding::Options& options) {
  const bool useVarint = options.useVarintRowCount;
  const uint32_t valueCount = values.size();
  const uint32_t alphabetCount =
      selection.statistics().uniqueCounts().value().size();
  auto* pool = &buffer.getMemoryPool();

  folly::F14FastMap<physicalType, uint32_t> alphabetMapping;
  alphabetMapping.reserve(alphabetCount);
  ScopedVector<physicalType> alphabet{alphabetCount, pool, options.bufferPool};
  ScopedVector<uint32_t> indices{valueCount, pool, options.bufferPool};

  uint32_t alphabetIndex{0};
  uint32_t indiciesIndex{0};
  auto valuesIt = values.begin();

  // Based on experimentation, there are at least two consistently
  // "good" ordering approaches that result in consistently smaller storage
  // footprints:
  //
  // 1. Sort alphabet values using natural order. This improves alphabet
  // compression, but can result in worse indices compression.
  //
  // 2. Order the alphabet by the first occurrence of each value in the input.
  // Storage savings mostly come from indices being much more delta encoding
  // friendly compared to a random order.
  //
  // Other approaches, like sorting alphabet values by frequency, also
  // occasionally yield more compressible data, but they're not consistent and
  // can result in storage regression compared to the first two.
  //
  // The second approach is used in this implementation because it's essentially
  // free: we don't need to allocate O(m) memory for a sorting buffer and
  // then do O(m log m) sorting. However, the first approach has a slightly
  // better distribution curve for storage savings.

  // Step one: fill alphabet and indices until we get a full alphabet mapping.
  for (; alphabetIndex < alphabetCount; ++valuesIt, ++indiciesIndex) {
    NIMBLE_DCHECK(valuesIt != values.end(), "Values iterator out of bounds.");
    const auto& value = *valuesIt;
    const auto& it = alphabetMapping.find(value);
    if (it == alphabetMapping.end()) {
      alphabetMapping.emplace(value, alphabetIndex);
      alphabet[alphabetIndex] = value;
      indices[indiciesIndex] = alphabetIndex;
      ++alphabetIndex;
    } else {
      indices[indiciesIndex] = it->second;
    }
  }

  // Step two: fill indices for the remaining values using the already fully
  // populated alphabet mapping. In most cases, depending on the data skew, we
  // will hit this step relatively early, and it will boost performance by
  // 20-30%.
  for (; valuesIt != values.end(); ++valuesIt) {
    NIMBLE_DCHECK(
        alphabetMapping.find(*valuesIt) != alphabetMapping.end(),
        "Mapping corruption. Missing alphabet entry.");
    indices[indiciesIndex++] = alphabetMapping.find(*valuesIt)->second;
  }

  NIMBLE_CHECK_EQ(indices.size(), valueCount, "Indices size mismatch.");
  NIMBLE_CHECK_EQ(alphabet.size(), alphabetCount, "Alphabet size mismatch.");

  ScopedEncodingBuffer scopedBuffer{
      &buffer.getMemoryPool(), options.encodingBufferPool};
  std::string_view serializedAlphabet = encodeAlphabet(
      selection,
      {alphabet.data(), alphabet.size()},
      scopedBuffer.get(),
      options);
  std::string_view serializedIndices =
      selection.template encodeNested<uint32_t>(
          EncodingIdentifiers::Dictionary::Indices,
          {indices},
          scopedBuffer.get(),
          options);

  const uint32_t encodingSize =
      Encoding::serializePrefixSize(valueCount, useVarint) + 4 +
      serializedAlphabet.size() + serializedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::Dictionary,
      TypeTraits<T>::dataType,
      valueCount,
      useVarint,
      pos);
  encoding::writeUint32(serializedAlphabet.size(), pos);
  encoding::writeBytes(serializedAlphabet, pos);
  encoding::writeBytes(serializedIndices, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string_view DictionaryEncoding<T>::slice(
    std::string_view encoded,
    uint32_t offset,
    uint32_t length,
    Buffer& buffer,
    const Encoding::Options& options) {
  const auto sourceRowCount =
      EncodingPrefix::readRowCount(encoded, options.useVarintRowCount);
  NIMBLE_CHECK_LE(offset, sourceRowCount);
  NIMBLE_CHECK_LE(length, sourceRowCount - offset);
  NIMBLE_CHECK_GT(length, 0, "Cannot slice zero rows.");

  const char* pos = encoded.data() +
      EncodingPrefix::prefixSize(encoded, options.useVarintRowCount);
  const uint32_t alphabetSize = encoding::readUint32(pos);
  const std::string_view alphabet{pos, alphabetSize};
  pos += alphabetSize;
  const std::string_view indices{
      pos, static_cast<size_t>(encoded.data() + encoded.size() - pos)};

  auto* pool = &buffer.getMemoryPool();
  ScopedEncodingBuffer scopedBuffer{pool, options.encodingBufferPool};
  const auto slicedIndices = EncodingFactory::slice(
      indices, offset, length, scopedBuffer.get(), options);

  const uint32_t encodingSize =
      EncodingPrefix::serializedSize(length, options.useVarintRowCount) +
      sizeof(uint32_t) + alphabet.size() + slicedIndices.size();
  char* reserved = buffer.reserve(encodingSize);
  char* writePos = reserved;
  EncodingPrefix::serialize(
      EncodingType::Dictionary,
      TypeTraits<T>::dataType,
      length,
      options.useVarintRowCount,
      writePos);
  encoding::writeUint32(alphabet.size(), writePos);
  encoding::writeBytes(alphabet, writePos);
  encoding::writeBytes(slicedIndices, writePos);
  NIMBLE_CHECK_EQ(writePos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string DictionaryEncoding<T>::debugString(int offset) const {
  std::string log = Encoding::debugString(offset);
  log += fmt::format(
      "\n{}indices child:\n{}",
      std::string(offset, ' '),
      indicesEncoding_->debugString(offset + 2));
  log += fmt::format(
      "\n{}alphabet:\n{}entries={}",
      std::string(offset, ' '),
      std::string(offset + 2, ' '),
      alphabet_.size());
  return log;
}

} // namespace facebook::nimble
