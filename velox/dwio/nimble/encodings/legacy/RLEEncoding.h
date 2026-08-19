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

#include <array>
#include <span>
#include <type_traits>
#include "velox/common/base/SimdUtil.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Holds data in RLE format. Run lengths are bit packed, and the run values
// are stored trivially.
//
// Note: we might want to recursively use the encoding factory to encode the
// run values. This recursive use can lead to great compression, but also
// tends to slow things down, particularly write speed.

namespace facebook::nimble::legacy {

namespace rle {

template <typename T>
void computeRuns(
    std::span<const T> data,
    Vector<uint32_t>* runLengths,
    Vector<T>* runValues) {
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
      runValues->push_back(last);
      last = data[i];
      runLength = 1;
    }
  }
  runLengths->push_back(runLength);
  runValues->push_back(last);
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
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory)
      : TypedEncoding<T, physicalType>(memoryPool, data),
        materializedRunLengths_{EncodingFactory().create(
            memoryPool,
            {data.data() + EncodingPrefix::kFixedPrefixSize + 4,
             *reinterpret_cast<const uint32_t*>(
                 data.data() + EncodingPrefix::kFixedPrefixSize)},
            stringBufferFactory)} {}

  void reset() {
    materializedRunLengths_.reset();
    derived().resetValues();
    copiesRemaining_ = materializedRunLengths_.nextValue();
    currentValue_ = nextValue();
  }

  void skip(uint32_t rowCount) final {
    uint32_t rowsLeft = rowCount;
    // TODO: We should have skip blocks.
    while (rowsLeft) {
      if (rowsLeft < copiesRemaining_) {
        copiesRemaining_ -= rowsLeft;
        return;
      } else {
        rowsLeft -= copiesRemaining_;
        copiesRemaining_ = materializedRunLengths_.nextValue();
        currentValue_ = nextValue();
      }
    }
  }

  void materialize(uint32_t rowCount, void* buffer) final {
    uint32_t rowsLeft = rowCount;
    physicalType* output = static_cast<physicalType*>(buffer);
    while (rowsLeft) {
      if (rowsLeft < copiesRemaining_) {
        velox::simd::simdFill(output, currentValue_, rowsLeft);
        copiesRemaining_ -= rowsLeft;
        return;
      } else {
        velox::simd::simdFill(output, currentValue_, copiesRemaining_);
        output += copiesRemaining_;
        rowsLeft -= copiesRemaining_;
        copiesRemaining_ = materializedRunLengths_.nextValue();
        currentValue_ = nextValue();
      }
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
            copiesRemaining_ = materializedRunLengths_.nextValue();
            currentValue_ = nextValue();
          }
          --copiesRemaining_;
          return currentValue_;
        });
  }

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer) {
    const uint32_t valueCount = values.size();
    Vector<uint32_t> runLengths(&buffer.getMemoryPool());
    Vector<physicalType> runValues(&buffer.getMemoryPool());
    rle::computeRuns(values, &runLengths, &runValues);

    Buffer tempBuffer{buffer.getMemoryPool()};
    std::string_view serializedRunLengths =
        selection.template encodeNested<uint32_t>(
            EncodingIdentifiers::RunLength::RunLengths, runLengths, tempBuffer);

    std::string_view serializedRunValues =
        getSerializedRunValues(selection, runValues, tempBuffer);

    const uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize + 4 +
        serializedRunLengths.size() + serializedRunValues.size();
    char* reserved = buffer.reserve(encodingSize);
    char* pos = reserved;
    Encoding::serializePrefix(
        EncodingType::RLE, TypeTraits<T>::dataType, valueCount, false, pos);
    encoding::writeString(serializedRunLengths, pos);
    encoding::writeBytes(serializedRunValues, pos);
    NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
    return {reserved, encodingSize};
  }

  const char* getValuesStart() const {
    return this->data_.data() + EncodingPrefix::kFixedPrefixSize + 4 +
        *reinterpret_cast<const uint32_t*>(
               this->data_.data() + EncodingPrefix::kFixedPrefixSize);
  }

  RLEEncoding& derived() {
    return *static_cast<RLEEncoding*>(this);
  }
  physicalType nextValue() {
    return derived().nextValue();
  }
  static std::string_view getSerializedRunValues(
      EncodingSelection<physicalType>& selection,
      const Vector<physicalType>& runValues,
      Buffer& buffer) {
    return RLEEncoding::getSerializedRunValues(selection, runValues, buffer);
  }

  uint32_t copiesRemaining_ = 0;
  physicalType currentValue_;
  detail::BufferedEncoding<uint32_t, 32> materializedRunLengths_;
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
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  physicalType nextValue();
  void resetValues();
  static std::string_view getSerializedRunValues(
      EncodingSelection<physicalType>& selection,
      const Vector<physicalType>& runValues,
      Buffer& buffer) {
    return selection.template encodeNested<physicalType>(
        EncodingIdentifiers::RunLength::RunValues, runValues, buffer);
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

 private:
  template <bool kDense>
  vector_size_t findNumInRun(
      const vector_size_t* rows,
      vector_size_t rowIndex,
      vector_size_t numRows,
      vector_size_t currentRow) const;

  detail::BufferedEncoding<physicalType, 32> values_;
};

// For the bool case we know the values will alternative between true
// and false, so in addition to the run lengths we need only store
// whether the first value is true or false.
// RLEEncodingBase bytes
// 1 byte: whether first row is true
template <>
class RLEEncoding<bool> final
    : public internal::RLEEncodingBase<bool, RLEEncoding<bool>> {
 public:
  RLEEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  bool nextValue();
  void resetValues();
  static std::string_view getSerializedRunValues(
      EncodingSelection<bool>& /* selection */,
      const Vector<bool>& runValues,
      Buffer& buffer) {
    char* reserved = buffer.reserve(sizeof(char));
    *reserved = runValues[0];
    return {reserved, 1};
  }

  void materializeBoolsAsBits(uint32_t rowCount, uint64_t* buffer, int begin)
      final;

 private:
  bool initialValue_;
  bool value_{};
};

//
// End of public API. Implementations follow.
//

template <typename T>
RLEEncoding<T>::RLEEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : internal::RLEEncodingBase<T, RLEEncoding<T>>(
          memoryPool,
          data,
          stringBufferFactory),
      values_{EncodingFactory().create(
          memoryPool,
          {internal::RLEEncodingBase<T, RLEEncoding<T>>::getValuesStart(),
           static_cast<size_t>(
               data.end() -
               internal::RLEEncodingBase<T, RLEEncoding<T>>::getValuesStart())},
          stringBufferFactory)} {
  internal::RLEEncodingBase<T, RLEEncoding<T>>::reset();
}

template <typename T>
typename RLEEncoding<T>::physicalType RLEEncoding<T>::nextValue() {
  return values_.nextValue();
}

template <typename T>
void RLEEncoding<T>::resetValues() {
  values_.reset();
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
      this->copiesRemaining_ = this->materializedRunLengths_.nextValue();
      this->currentValue_ = nextValue();
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
  auto* nulls = visitor.reader().rawNullsInReadRange();
  if (velox::dwio::common::useFastPath(visitor, nulls)) {
    detail::readWithVisitorFast(*this, visitor, params, nulls);
  } else {
    internal::RLEEncodingBase<T, RLEEncoding<T>>::readWithVisitor(
        visitor, params);
  }
}

} // namespace facebook::nimble::legacy
