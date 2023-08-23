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

#include "velox/dwio/dwrf/reader/SelectiveDecimalColumnReader.h"

namespace facebook::velox::dwrf {

template <typename DataT>
SelectiveDecimalColumnReader<DataT>::SelectiveDecimalColumnReader(
    const std::shared_ptr<const TypeWithId>& nodeType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : SelectiveColumnReader(nodeType->type(), params, scanSpec, nodeType) {
  EncodingKey encodingKey{fileType_->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  if constexpr (std::is_same_v<DataT, std::int64_t>) {
    scale_ = requestedType_->asShortDecimal().scale();
  } else {
    scale_ = requestedType_->asLongDecimal().scale();
  }
  version_ = convertRleVersion(stripe.getEncoding(encodingKey).kind());
  auto data = encodingKey.forKind(proto::Stream_Kind_DATA);
  valueDecoder_ = createDirectDecoder</*isSigned*/ true>(
      stripe.getStream(data, params.streamLabels().label(), true),
      stripe.getUseVInts(data),
      sizeof(DataT));

  // [NOTICE] DWRF's NANO_DATA has the same enum value as ORC's SECONDARY
  auto secondary = encodingKey.forKind(proto::Stream_Kind_NANO_DATA);
  scaleDecoder_ = createRleDecoder</*isSigned*/ true>(
      stripe.getStream(secondary, params.streamLabels().label(), true),
      version_,
      memoryPool_,
      stripe.getUseVInts(secondary),
      LONG_BYTE_SIZE);
}

template <typename DataT>
uint64_t SelectiveDecimalColumnReader<DataT>::skip(uint64_t numValues) {
  numValues = SelectiveColumnReader::skip(numValues);
  valueDecoder_->skip(numValues);
  scaleDecoder_->skip(numValues);
  return numValues;
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::seekToRowGroup(uint32_t index) {
  SelectiveColumnReader::seekToRowGroup(index);
  auto positionsProvider = formatData_->seekToRowGroup(index);
  valueDecoder_->seekToRowGroup(positionsProvider);
  scaleDecoder_->seekToRowGroup(positionsProvider);
  VELOX_CHECK(!positionsProvider.hasNext());
}

template <typename DataT>
template <bool isDense, typename TFilter, typename ExtractValues>
void SelectiveDecimalColumnReader<DataT>::readHelper(
    RowSet rows,
    velox::common::Filter* decodeFilter,
    ExtractValues extractValues,
    const velox::common::Filter& valuesFilter) {
  vector_size_t numRows = rows.back() + 1;
  DirectRleColumnVisitor<int64_t, TFilter, ExtractValues, isDense> visitor(
      *reinterpret_cast<TFilter*>(decodeFilter), this, rows, extractValues);
  // decode scale stream
  if (version_ == velox::dwrf::RleVersion_1) {
    decodeWithVisitor<RleDecoderV1<true>>(scaleDecoder_.get(), visitor);
  } else {
    decodeWithVisitor<RleDecoderV2<true>>(scaleDecoder_.get(), visitor);
  }

  // copy scales into scaleBuffer_
  ensureCapacity<int64_t>(scaleBuffer_, numValues_, &memoryPool_);
  size_t size = numValues_ * sizeof(int64_t);
  scaleBuffer_->setSize(size);
  memcpy(scaleBuffer_->asMutable<char>(), rawValues_, size);

  // reset numValues_ and outputRows_ before reading values
  numValues_ = 0;
  outputRows_.clear();
  valueSize_ = sizeof(DataT);
  ensureValuesCapacity<DataT>(numRows);

  // decode value stream
  facebook::velox::dwio::common::
      ColumnVisitor<DataT, TFilter, ExtractValues, isDense>
          valueVisitor(
              *reinterpret_cast<TFilter*>(decodeFilter),
              this,
              rows,
              extractValues);
  decodeWithVisitor<DirectDecoder<true>>(valueDecoder_.get(), valueVisitor);
  auto scales = scaleBuffer_->as<int64_t>();
  auto values = values_->asMutable<DataT>();

  // if nullsPtr is NULL fillDecimals can go fast path
  uint64_t* nullsPtr = nullptr;
  if (decodeFilter->testNull()) {
    VELOX_CHECK(
        outputRows_.empty(),
        "outputRows_ should be empty if decodeFilter accept Nulls.");
    // if decodeFilter acceptNulls(e.g. AlwaysTrue), we should process all rows
    for (auto row : rows) {
      outputRows_.push_back(row);
    }
    // in this case nullsPtr is needed to judge whether the target value is NULL
    if (nullsInReadRange_) {
      nullsPtr = nullsInReadRange_->asMutable<uint64_t>();
    }
  }
  DecimalUtil::fillDecimals<DataT>(
      nullsPtr, values, values, scales, scale_, outputRows_, valuesFilter);

  numValues_ = outputRows_.size();
  rawValues_ = values_->asMutable<char>();
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  velox::common::Filter* filter =
      scanSpec_->filter() ? scanSpec_->filter() : &alwaysTrue();
  if (filter->testNull() && version_ == velox::dwrf::RleVersion_2) {
    /// RLEv2 does't support fastpath yet, temporarily disable bulkPathEnable_
    /// to ensure that resultNulls_ will be allocated during prepareNulls()
    /// or addNull() will be failed if the target Filter accept NULLs
    bulkPathEnable_ = false;
  }
  prepareRead<int64_t>(offset, rows, incomingNulls);
  // enable fastpath after prepare has been finished
  bulkPathEnable_ = true;
  bool isDense = rows.back() == rows.size() - 1;
  if (scanSpec_->keepValues()) {
    if (scanSpec_->valueHook()) {
      if (isDense) {
        return processValueHook<true>(rows, scanSpec_->valueHook());
      } else {
        return processValueHook<false>(rows, scanSpec_->valueHook());
      }
    }
    if (isDense) {
      processFilter<true>(filter, ExtractToReader(this), rows);
    } else {
      processFilter<false>(filter, ExtractToReader(this), rows);
    }
  } else {
    if (isDense) {
      processFilter<true>(filter, DropValues(), rows);
    } else {
      processFilter<false>(filter, DropValues(), rows);
    }
  }
  readOffset_ += rows.back() + 1;
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::getValues(
    RowSet rows,
    VectorPtr* result) {
  getIntValues(rows, requestedType_, result);
}

template <typename DataT>
template <bool isDense, typename ExtractValues>
void SelectiveDecimalColumnReader<DataT>::processFilter(
    velox::common::Filter* filter,
    ExtractValues extractValues,
    RowSet rows) {
  velox::common::IsNotNull isNotNull;
  switch (filter->kind()) {
    case velox::common::FilterKind::kAlwaysTrue:
      readHelper<isDense, velox::common::AlwaysTrue, ExtractValues>(
          rows, filter, extractValues, alwaysTrue());
      break;
    case velox::common::FilterKind::kIsNull:
      // scale's type is int64_t
      filterNulls<int64_t>(
          rows, true, !std::is_same_v<ExtractValues, DropValues>);
      break;
    case velox::common::FilterKind::kIsNotNull:
      if (std::is_same_v<ExtractValues, DropValues>) {
        filterNulls<int64_t>(rows, false, false);
        break;
      }
    case velox::common::FilterKind::kBigintRange:
    case velox::common::FilterKind::kBigintMultiRange:
    case velox::common::FilterKind::kBigintValuesUsingHashTable:
    case velox::common::FilterKind::kBigintValuesUsingBitmask:
    case velox::common::FilterKind::kNegatedBigintRange:
    case velox::common::FilterKind::kNegatedBigintValuesUsingHashTable:
    case velox::common::FilterKind::kNegatedBigintValuesUsingBitmask:
    case velox::common::FilterKind::kHugeintRange:
    case velox::common::FilterKind::kHugeintValuesUsingHashTable:
      // Filters don't accept Nulls, so we can use IsNotNull during decoding.
      readHelper<isDense, velox::common::IsNotNull>(
          rows, &isNotNull, ExtractToReader(this), *filter);
      break;
    default:
      VELOX_FAIL("Unsupported Filter Type {}.", filter->toString());
      break;
  }
}

template <typename DataT>
template <bool isDense>
void SelectiveDecimalColumnReader<DataT>::processValueHook(
    RowSet rows,
    ValueHook* hook) {
  if (hook->kSkipNulls) {
    // use IsNotNull to Filter out some data during decoding phase
    velox::common::IsNotNull isNotNull;
    readHelper<isDense, velox::common::IsNotNull>(
        rows, &isNotNull, ExtractToReader(this), alwaysTrue());
  } else {
    readHelper<isDense, velox::common::AlwaysTrue>(
        rows, &alwaysTrue(), ExtractToReader(this), alwaysTrue());
  }
  hook->addValues(outputRows_.data(), rawValues_, numValues_, sizeof(DataT));
}

template class SelectiveDecimalColumnReader<int64_t>;
template class SelectiveDecimalColumnReader<int128_t>;

} // namespace facebook::velox::dwrf
