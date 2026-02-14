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

#include "velox/dwio/dwrf/reader/SelectiveDecimalColumnReader.h"

namespace facebook::velox::dwrf {

template <typename DataT>
SelectiveDecimalColumnReader<DataT>::SelectiveDecimalColumnReader(
    const std::shared_ptr<const TypeWithId>& fileType,
    DwrfParams& params,
    common::ScanSpec& scanSpec)
    : SelectiveColumnReader(fileType->type(), fileType, params, scanSpec) {
  EncodingKey encodingKey{fileType_->id(), params.flatMapContext().sequence};
  auto& stripe = params.stripeStreams();
  if constexpr (std::is_same_v<DataT, std::int64_t>) {
    scale_ = requestedType_->asShortDecimal().scale();
  } else {
    scale_ = requestedType_->asLongDecimal().scale();
  }
  version_ = convertRleVersion(stripe, encodingKey);
  auto data = StripeStreamsUtil::getStreamForKind(
      stripe,
      encodingKey,
      proto::Stream_Kind_DATA,
      proto::orc::Stream_Kind_DATA);
  valueDecoder_ = createDirectDecoder</*isSigned*/ true>(
      stripe.getStream(data, params.streamLabels().label(), true),
      stripe.getUseVInts(data),
      sizeof(DataT));

  // [NOTICE] DWRF's NANO_DATA has the same enum value as ORC's SECONDARY
  auto secondary = StripeStreamsUtil::getStreamForKind(
      stripe,
      encodingKey,
      proto::Stream_Kind_NANO_DATA,
      proto::orc::Stream_Kind_SECONDARY);
  scaleDecoder_ = createRleDecoder</*isSigned*/ true>(
      stripe.getStream(secondary, params.streamLabels().label(), true),
      version_,
      *memoryPool_,
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
void SelectiveDecimalColumnReader<DataT>::seekToRowGroup(int64_t index) {
  SelectiveColumnReader::seekToRowGroup(index);
  auto positionsProvider = formatData_->seekToRowGroup(index);
  valueDecoder_->seekToRowGroup(positionsProvider);
  scaleDecoder_->seekToRowGroup(positionsProvider);
  VELOX_CHECK(!positionsProvider.hasNext());
}

template <typename DataT>
template <bool kDense>
void SelectiveDecimalColumnReader<DataT>::readHelper(
    const common::Filter* filter,
    RowSet rows) {
  ExtractToReader extractValues(this);
  common::AlwaysTrue alwaysTrue;
  DirectRleColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      decltype(extractValues),
      kDense>
      visitor(alwaysTrue, this, rows, extractValues);

  // decode scale stream
  if (version_ == velox::dwrf::RleVersion_1) {
    decodeWithVisitor<RleDecoderV1<true>>(scaleDecoder_.get(), visitor);
  } else {
    decodeWithVisitor<RleDecoderV2<true>>(scaleDecoder_.get(), visitor);
  }

  // copy scales into scaleBuffer_
  ensureCapacity<int64_t>(scaleBuffer_, numValues_, memoryPool_);
  scaleBuffer_->setSize(numValues_ * sizeof(int64_t));
  memcpy(
      scaleBuffer_->asMutable<char>(),
      rawValues_,
      numValues_ * sizeof(int64_t));

  // reset numValues_ before reading values
  numValues_ = 0;
  valueSize_ = sizeof(DataT);
  vector_size_t numRows = rows.back() + 1;
  ensureValuesCapacity<DataT>(numRows);

  // decode value stream
  facebook::velox::dwio::common::
      ColumnVisitor<DataT, common::AlwaysTrue, decltype(extractValues), kDense>
          valueVisitor(alwaysTrue, this, rows, extractValues);
  decodeWithVisitor<DirectDecoder<true>>(valueDecoder_.get(), valueVisitor);
  readOffset_ += numRows;

  // Fill decimals before applying filter.
  fillDecimals();

  // 'nullsInReadRange_' is the nulls for the entire read range, and if the row
  // set is not dense, result nulls should be allocated, which represents the
  // nulls for the selected rows before filtering.
  const auto rawNulls = nullsInReadRange_
      ? (kDense ? nullsInReadRange_->as<uint64_t>() : rawResultNulls_)
      : nullptr;
  // Process filter.
  process(filter, rows, rawNulls);
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::processNulls(
    bool isNull,
    const RowSet& rows,
    const uint64_t* rawNulls) {
  if (!rawNulls) {
    return;
  }
  returnReaderNulls_ = false;
  anyNulls_ = !isNull;
  allNull_ = isNull;

  auto rawDecimal = values_->asMutable<DataT>();
  auto rawScale = scaleBuffer_->asMutable<int64_t>();

  vector_size_t idx = 0;
  if (isNull) {
    for (vector_size_t i = 0; i < numValues_; i++) {
      if (bits::isBitNull(rawNulls, i)) {
        bits::setNull(rawResultNulls_, idx);
        addOutputRow(rows[i]);
        idx++;
      }
    }
  } else {
    for (vector_size_t i = 0; i < numValues_; i++) {
      if (!bits::isBitNull(rawNulls, i)) {
        bits::setNull(rawResultNulls_, idx, false);
        rawDecimal[idx] = rawDecimal[i];
        rawScale[idx] = rawScale[i];
        addOutputRow(rows[i]);
        idx++;
      }
    }
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::processFilter(
    const common::Filter* filter,
    const RowSet& rows,
    const uint64_t* rawNulls) {
  VELOX_CHECK_NOT_NULL(filter, "Filter must not be null.");
  returnReaderNulls_ = false;
  anyNulls_ = false;
  allNull_ = true;

  vector_size_t idx = 0;
  auto rawDecimal = values_->asMutable<DataT>();
  for (vector_size_t i = 0; i < numValues_; i++) {
    if (rawNulls && bits::isBitNull(rawNulls, i)) {
      if (filter->testNull()) {
        bits::setNull(rawResultNulls_, idx);
        addOutputRow(rows[i]);
        anyNulls_ = true;
        idx++;
      }
    } else {
      bool tested;
      if constexpr (std::is_same_v<DataT, int64_t>) {
        tested = filter->testInt64(rawDecimal[i]);
      } else {
        tested = filter->testInt128(rawDecimal[i]);
      }

      if (tested) {
        if (rawNulls) {
          bits::setNull(rawResultNulls_, idx, false);
        }
        rawDecimal[idx] = rawDecimal[i];
        addOutputRow(rows[i]);
        allNull_ = false;
        idx++;
      }
    }
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::process(
    const common::Filter* filter,
    const RowSet& rows,
    const uint64_t* rawNulls) {
  if (!filter) {
    // No filter and "hasDeletion" is false so input rows will be
    // reused.
    return;
  }

  switch (filter->kind()) {
    case common::FilterKind::kIsNull:
      processNulls(true, rows, rawNulls);
      break;
    case common::FilterKind::kIsNotNull: {
      if (rawNulls) {
        processNulls(false, rows, rawNulls);
      } else {
        for (vector_size_t i = 0; i < numValues_; i++) {
          addOutputRow(rows[i]);
        }
      }
      break;
    }
    default:
      processFilter(filter, rows, rawNulls);
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::read(
    int64_t offset,
    const RowSet& rows,
    const uint64_t* incomingNulls) {
  VELOX_CHECK(!scanSpec_->valueHook());
  prepareRead<int64_t>(offset, rows, incomingNulls);
  if (!scanSpec_->keepValues() && scanSpec_->filter() &&
      (!resultNulls_ || !resultNulls_->unique() ||
       resultNulls_->capacity() * 8 < rows.size())) {
    // Make sure a dedicated resultNulls_ is allocated with enough capacity as
    // RleDecoder always assumes it is available and 'prepareRead' skips
    // allocation when the column is not projected.
    resultNulls_ = AlignedBuffer::allocate<bool>(rows.size(), memoryPool_);
    rawResultNulls_ = resultNulls_->asMutable<uint64_t>();
  }
  rawValues_ = values_->asMutable<char>();
  bool isDense = rows.back() == rows.size() - 1;
  if (isDense) {
    readHelper<true>(scanSpec_->filter(), rows);
  } else {
    readHelper<false>(scanSpec_->filter(), rows);
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::getValues(
    const RowSet& rows,
    VectorPtr* result) {
  getIntValues(rows, requestedType_, result);
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::fillDecimals() {
  auto nullsPtr =
      resultNulls() ? resultNulls()->template as<uint64_t>() : nullptr;
  auto scales = scaleBuffer_->as<int64_t>();
  auto values = values_->asMutable<DataT>();
  DecimalUtil::fillDecimals<DataT>(
      values, nullsPtr, values, scales, numValues_, scale_);
}

template class SelectiveDecimalColumnReader<int64_t>;
template class SelectiveDecimalColumnReader<int128_t>;

} // namespace facebook::velox::dwrf
