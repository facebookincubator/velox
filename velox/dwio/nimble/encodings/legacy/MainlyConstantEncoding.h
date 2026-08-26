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
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrimitives.h"
#include "velox/dwio/nimble/encodings/common/EncodingType.h"
#include "velox/dwio/nimble/encodings/legacy/Encoding.h"
#include "velox/dwio/nimble/encodings/legacy/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/selection/EncodingIdentifier.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"

// Encodes data that is 'mainly' a single value by using a bool child vectors
// to mark the rows that are that value, and another child encoding to encode
// the other values. We don't actually require that the single value be any
// given fraction of the data, but generally the encoding is effective when that
// constant fraction is large (say 50%+). All data types except bool are
// supported.
//
// E.g. if the data is
// 1 1 2 1 1 1 3 1 1 9
//
// The single value would be 1
//
// The bool vector would be
// T T F T T T F T T F
//
// The other-values child encoding would be
// 2 3 9
//
// This construction is quite similar to that of NullableEncoding, but instead
// of marking nulls specially we mark the constant value.

namespace facebook::nimble::legacy {

// Data layout is:
// EncodingPrefix::kFixedPrefixSize bytes: standard Encoding prefix
// 4 bytes: num isCommon encoding bytes (X)
// X bytes: isCommon encoding bytes
// 4 bytes: num otherValues encoding bytes (Y)
// Y bytes: otherValues encoding bytes
// Z bytes: the constant value via encoding primitive.
template <typename T>
class MainlyConstantEncoding final
    : public TypedEncoding<T, typename TypeTraits<T>::physicalType> {
 public:
  using cppDataType = T;
  using physicalType = typename TypeTraits<T>::physicalType;

  MainlyConstantEncoding(
      velox::memory::MemoryPool& memoryPool,
      std::string_view data,
      std::function<void*(uint32_t)> stringBufferFactory);

  void reset() final;
  void skip(uint32_t rowCount) final;
  void materialize(uint32_t rowCount, void* buffer) final;

  template <typename DecoderVisitor>
  void readWithVisitor(DecoderVisitor& visitor, ReadWithVisitorParams& params);

  template <bool kScatter, typename Visitor>
  void bulkScan(
      Visitor& visitor,
      vector_size_t currentRow,
      const vector_size_t* selectedRows,
      vector_size_t numSelected,
      const vector_size_t* scatterRows);

  static std::string_view encode(
      EncodingSelection<physicalType>& selection,
      std::span<const physicalType> values,
      Buffer& buffer);

  std::string debugString(int offset) const final;

 private:
  std::unique_ptr<Encoding> isCommon_;
  std::unique_ptr<Encoding> otherValues_;
  physicalType commonValue_;
  // Temporary bufs.
  Vector<bool> isCommonBuffer_;
  Vector<physicalType> otherValuesBuffer_;
};

//
// End of public API. Implementation follows.
//

template <typename T>
MainlyConstantEncoding<T>::MainlyConstantEncoding(
    velox::memory::MemoryPool& memoryPool,
    std::string_view data,
    std::function<void*(uint32_t)> stringBufferFactory)
    : TypedEncoding<T, physicalType>(memoryPool, data),
      isCommonBuffer_(&memoryPool),
      otherValuesBuffer_(&memoryPool) {
  const EncodingFactory factory;
  const char* pos = data.data() + EncodingPrefix::kFixedPrefixSize;
  const uint32_t isCommonBytes = encoding::readUint32(pos);
  isCommon_ = factory.create(
      *this->pool_,
      {pos, isCommonBytes},
      stringBufferFactory,
      Encoding::Options{});
  pos += isCommonBytes;
  const uint32_t otherValuesBytes = encoding::readUint32(pos);
  otherValues_ = factory.create(
      *this->pool_,
      {pos, otherValuesBytes},
      stringBufferFactory,
      Encoding::Options{});
  pos += otherValuesBytes;
  commonValue_ = encoding::read<physicalType>(pos);
  NIMBLE_CHECK(pos == data.end(), "Unexpected mainly constant encoding end");
}

template <typename T>
void MainlyConstantEncoding<T>::reset() {
  isCommon_->reset();
  otherValues_->reset();
}

template <typename T>
void MainlyConstantEncoding<T>::skip(uint32_t rowCount) {
  // Hrm this isn't ideal. We should return to this later -- a new
  // encoding func? Encoding::Accumulate to add up next N rows?
  isCommonBuffer_.resize(rowCount);
  isCommon_->materialize(rowCount, isCommonBuffer_.data());
  const uint32_t commonCount =
      std::accumulate(isCommonBuffer_.begin(), isCommonBuffer_.end(), 0U);
  const uint32_t nonCommonCount = rowCount - commonCount;
  if (nonCommonCount == 0) {
    return;
  }

  otherValues_->skip(nonCommonCount);
}

template <typename T>
void MainlyConstantEncoding<T>::materialize(uint32_t rowCount, void* buffer) {
  // This too isn't ideal. We will want an Encoding::Indices method or
  // something our SparseBool can use, giving back just the set indices
  // rather than a materialization.
  isCommonBuffer_.resize(rowCount);
  isCommon_->materialize(rowCount, isCommonBuffer_.data());
  const uint32_t commonCount =
      std::accumulate(isCommonBuffer_.begin(), isCommonBuffer_.end(), 0U);
  const uint32_t nonCommonCount = rowCount - commonCount;

  if (nonCommonCount == 0) {
    physicalType* output = static_cast<physicalType*>(buffer);
    std::fill(output, output + rowCount, commonValue_);
    return;
  }

  otherValuesBuffer_.reserve(nonCommonCount);
  otherValues_->materialize(nonCommonCount, otherValuesBuffer_.data());
  physicalType* output = static_cast<physicalType*>(buffer);
  const physicalType* nextOtherValue = otherValuesBuffer_.begin();
  // This is a generic scatter -- should we have a common scatter func?
  for (uint32_t i = 0; i < rowCount; ++i) {
    if (isCommonBuffer_[i]) {
      *output++ = commonValue_;
    } else {
      *output++ = *nextOtherValue++;
    }
  }
  NIMBLE_DCHECK_EQ(
      nextOtherValue - otherValuesBuffer_.begin(),
      nonCommonCount,
      "Encoding size mismatch.");
}

template <typename T>
template <bool kScatter, typename V>
void MainlyConstantEncoding<T>::bulkScan(
    V& visitor,
    vector_size_t currentRow,
    const vector_size_t* selectedRows,
    vector_size_t numSelected,
    const vector_size_t* scatterRows) {
  using DataType = typename V::DataType;
  using ValueType = detail::ValueType<DataType>;
  constexpr bool kScatterValues = kScatter && !V::kHasFilter && !V::kHasHook;
  ValueType* values;
  const auto commonData = detail::castFromPhysicalType<DataType>(commonValue_);
  const bool commonPassed =
      velox::common::applyFilter(visitor.filter(), commonData);
  if constexpr (!V::kFilterOnly) {
    auto numRows = visitor.numRows() - visitor.rowIndex();
    values = detail::mutableValues<ValueType>(visitor, numRows);
    if (commonPassed) {
      auto commonValue = detail::dataToValue(visitor, commonData);
      std::fill(values, values + numRows, commonValue);
    }
  }
  const auto numIsCommon = selectedRows[numSelected - 1] + 1 - currentRow;
  isCommonBuffer_.resize(velox::bits::nwords(numIsCommon) * sizeof(uint64_t));
  auto* isCommon = reinterpret_cast<uint64_t*>(isCommonBuffer_.data());
  // TODO: Wrap otherValues_ in BufferedEncoding.  This way when isCommon_ is
  // SparseBoolEncoding or RLE, we can materialize it on demand and do not need
  // to allocate memory for the indices.
  isCommon_->materializeBoolsAsBits(numIsCommon, isCommon, 0);
  auto numOtherValues =
      numIsCommon - velox::bits::countBits(isCommon, 0, numIsCommon);
  otherValuesBuffer_.resize(numOtherValues);
  otherValues_->materialize(numOtherValues, otherValuesBuffer_.data());
  numOtherValues = 0;
  auto* filterHits = V::kHasFilter ? visitor.outputRows(numSelected) : nullptr;
  auto* rows = kScatter ? scatterRows : selectedRows;
  vector_size_t numValues = 0;
  vector_size_t numHits = 0;
  vector_size_t selectedRowIndex = 0;
  velox::bits::forEachUnsetBit(isCommon, 0, numIsCommon, [&](vector_size_t i) {
    i += currentRow;
    auto commonBegin = selectedRowIndex;
    if constexpr (V::dense) {
      selectedRowIndex += i - selectedRows[selectedRowIndex];
    } else {
      while (selectedRows[selectedRowIndex] < i) {
        ++selectedRowIndex;
      }
    }
    const auto numCommon = selectedRowIndex - commonBegin;
    if (V::kHasFilter && commonPassed && numCommon > 0) {
      auto* begin = rows + commonBegin;
      std::copy(begin, begin + numCommon, filterHits + numHits);
      numHits += numCommon;
    }
    if (selectedRows[selectedRowIndex] > i) {
      if constexpr (!V::kFilterOnly) {
        vector_size_t numRows;
        if constexpr (kScatterValues) {
          numRows = scatterRows[selectedRowIndex] - visitor.rowIndex();
          visitor.addRowIndex(numRows);
        } else {
          numRows = commonPassed * numCommon;
        }
        numValues += numRows;
      }
      ++numOtherValues;
      return;
    }
    auto otherData = detail::castFromPhysicalType<DataType>(
        otherValuesBuffer_[numOtherValues++]);
    bool otherPassed;
    if constexpr (V::kHasFilter) {
      otherPassed = velox::common::applyFilter(visitor.filter(), otherData);
      if (otherPassed) {
        filterHits[numHits++] = rows[selectedRowIndex];
      }
    } else {
      otherPassed = true;
    }
    if constexpr (!V::kFilterOnly) {
      auto* begin = values + numValues;
      vector_size_t numRows;
      if constexpr (kScatterValues) {
        begin[scatterRows[selectedRowIndex] - visitor.rowIndex()] =
            detail::dataToValue(visitor, otherData);
        auto end = selectedRowIndex + 1;
        if (FOLLY_UNLIKELY(end == numSelected)) {
          numRows = visitor.numRows() - visitor.rowIndex();
        } else {
          numRows = scatterRows[end] - visitor.rowIndex();
        }
        visitor.addRowIndex(numRows);
      } else {
        numRows = commonPassed * numCommon;
        if (otherPassed) {
          begin[numRows++] = detail::dataToValue(visitor, otherData);
        }
      }
      numValues += numRows;
    }
    ++selectedRowIndex;
  });
  auto numCommon = numSelected - selectedRowIndex;
  if (commonPassed && numCommon > 0) {
    if constexpr (V::kHasFilter) {
      auto* begin = rows + selectedRowIndex;
      std::copy(begin, begin + numCommon, filterHits + numHits);
      numHits += numCommon;
    }
    if constexpr (!V::kFilterOnly) {
      if constexpr (kScatterValues) {
        numValues += visitor.numRows() - visitor.rowIndex();
      } else {
        numValues += numCommon;
      }
    }
  }
  visitor.setRowIndex(visitor.numRows());
  if constexpr (V::kHasHook) {
    NIMBLE_DCHECK_EQ(numValues, numSelected);
    visitor.hook().addValues(scatterRows, values, numSelected);
  } else {
    visitor.addNumValues(V::kFilterOnly ? numHits : numValues);
  }
}

template <typename T>
template <typename V>
void MainlyConstantEncoding<T>::readWithVisitor(
    V& visitor,
    ReadWithVisitorParams& params) {
  auto* nulls = visitor.reader().rawNullsInReadRange();
  if (velox::dwio::common::useFastPath(visitor, nulls)) {
    detail::readWithVisitorFast(*this, visitor, params, nulls);
    return;
  }
  detail::readWithVisitorSlow(
      visitor,
      params,
      [&](auto toSkip) { skip(toSkip); },
      [&] {
        bool isCommon;
        isCommon_->materialize(1, &isCommon);
        if (isCommon) {
          return commonValue_;
        }
        physicalType otherValue;
        otherValues_->materialize(1, &otherValue);
        return otherValue;
      });
}

namespace internal {} // namespace internal

template <typename T>
std::string_view MainlyConstantEncoding<T>::encode(
    EncodingSelection<physicalType>& selection,
    std::span<const physicalType> values,
    Buffer& buffer) {
  if (values.empty()) {
    NIMBLE_INCOMPATIBLE_ENCODING("MainlyConstantEncoding cannot be empty.");
  }

  // Tie-break on the value so the common value is deterministic;
  // absl::flat_hash_map iteration order is randomized per run, which would
  // otherwise make the encoded output vary for identical input.
  const auto commonElement = std::max_element(
      selection.statistics().uniqueCounts().value().cbegin(),
      selection.statistics().uniqueCounts().value().cend(),
      [](const auto& a, const auto& b) {
        if (a.second != b.second) {
          return a.second < b.second;
        }
        return a.first < b.first;
      });

  const uint32_t entryCount = values.size();
  const uint32_t uncommonCount = entryCount - commonElement->second;

  Vector<bool> isCommon{&buffer.getMemoryPool(), values.size(), true};
  Vector<physicalType> otherValues(&buffer.getMemoryPool());
  otherValues.reserve(uncommonCount);

  physicalType commonValue = commonElement->first;
  for (auto i = 0; i < values.size(); ++i) {
    physicalType currentValue = values[i];
    if (currentValue != commonValue) {
      isCommon[i] = false;
      otherValues.push_back(std::move(currentValue));
    }
  }

  Buffer tempBuffer{buffer.getMemoryPool()};
  std::string_view serializedIsCommon = selection.template encodeNested<bool>(
      EncodingIdentifiers::MainlyConstant::IsCommon, isCommon, tempBuffer);
  std::string_view serializedOtherValues =
      selection.template encodeNested<physicalType>(
          EncodingIdentifiers::MainlyConstant::OtherValues,
          otherValues,
          tempBuffer);

  uint32_t encodingSize = EncodingPrefix::kFixedPrefixSize + 8 +
      serializedIsCommon.size() + serializedOtherValues.size();
  if constexpr (isNumericType<physicalType>()) {
    encodingSize += sizeof(physicalType);
  } else {
    encodingSize += 4 + commonValue.size();
  }
  char* reserved = buffer.reserve(encodingSize);
  char* pos = reserved;
  Encoding::serializePrefix(
      EncodingType::MainlyConstant,
      TypeTraits<T>::dataType,
      entryCount,
      false,
      pos);
  // TODO: Reorder these so that metadata is at the beginning.
  encoding::writeString(serializedIsCommon, pos);
  encoding::writeString(serializedOtherValues, pos);
  encoding::write<physicalType>(commonValue, pos);
  NIMBLE_DCHECK_EQ(pos - reserved, encodingSize, "Encoding size mismatch.");
  return {reserved, encodingSize};
}

template <typename T>
std::string MainlyConstantEncoding<T>::debugString(int offset) const {
  std::string log = fmt::format(
      "{}{}<{}> rowCount={} commonValue={}",
      std::string(offset, ' '),
      toString(Encoding::encodingType()),
      toString(Encoding::dataType()),
      Encoding::rowCount(),
      commonValue_);
  log += fmt::format(
      "\n{}isCommon child:\n{}",
      std::string(offset + 2, ' '),
      isCommon_->debugString(offset + 4));
  log += fmt::format(
      "\n{}otherValues child:\n{}",
      std::string(offset + 2, ' '),
      otherValues_->debugString(offset + 4));
  return log;
}

} // namespace facebook::nimble::legacy
