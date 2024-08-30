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
template <bool kDense>
void SelectiveDecimalColumnReader<DataT>::readScales(RowSet rows) {
  ExtractToReader extractScales(this);
  common::AlwaysTrue filter;
  DirectRleColumnVisitor<
      int64_t,
      common::AlwaysTrue,
      decltype(extractScales),
      kDense>
      visitor(filter, this, rows, extractScales);

  // decode scale stream
  if (version_ == velox::dwrf::RleVersion_1) {
    decodeWithVisitor<RleDecoderV1<true>>(scaleDecoder_.get(), visitor);
  } else {
    decodeWithVisitor<RleDecoderV2<true>>(scaleDecoder_.get(), visitor);
  }

  // copy scales into scaleBuffer_
  ensureCapacity<int64_t>(scaleBuffer_, numValues_, &memoryPool_);
  scaleBuffer_->setSize(numValues_ * sizeof(int64_t));
  memcpy(
      scaleBuffer_->asMutable<char>(),
      rawValues_,
      numValues_ * sizeof(int64_t));
}

template <typename DataT>
template <bool kDense, typename ExtractValues>
void SelectiveDecimalColumnReader<DataT>::readHelper(RowSet rows, ExtractValues extractValues) {
  vector_size_t numRows = rows.back() + 1;

  // reset numValues_ before reading values
  numValues_ = 0;
  valueSize_ = sizeof(DataT);
  ensureValuesCapacity<DataT>(numRows);

  // decode value stream
  common::AlwaysTrue filter;
  facebook::velox::dwio::common::
      ColumnVisitor<DataT, common::AlwaysTrue, ExtractValues, kDense>
          valueVisitor(filter, this, rows, extractValues);
  decodeWithVisitor<DirectDecoder<true>>(valueDecoder_.get(), valueVisitor);
  readOffset_ += numRows;
}

template <typename DataT>
template <bool kDense>
void SelectiveDecimalColumnReader<DataT>::processValueHook(RowSet rows, ValueHook* hook) {
  const auto *scales = scaleBuffer_->as<int64_t>();
  switch (hook->kind()) {
    case aggregate::AggregationHook::kSumBigintToBigint:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::SumHook<int64_t, int64_t, false>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kSumBigintToBigintOverflow:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::SumHook<int64_t, int64_t, true>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kBigintMax:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::MinMaxHook<int64_t, false>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kBigintMin:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::MinMaxHook<int64_t, true>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kSumHugeintToHugeint:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::SumHook<int128_t, int128_t, false>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kSumHugeintToHugeintOverflow:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::SumHook<int128_t, int128_t, true>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kHugeintMax:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::MinMaxHook<int128_t, false>>(hook, scales, scale_));
      break;
    case aggregate::AggregationHook::kHugeintMin:
      readHelper<kDense>(
          rows,
          ExtractToDecimalHook<aggregate::MinMaxHook<int128_t, true>>(hook, scales, scale_));
      break;
    default:
      readHelper<kDense>(rows, ExtractToGenericDecimalHook(hook, scales, scale_));
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* incomingNulls) {
  VELOX_CHECK(!scanSpec_->filter());
  VELOX_CHECK(!scanSpec_->valueHook());
  prepareRead<int64_t>(offset, rows, incomingNulls);
  bool isDense = rows.back() == rows.size() - 1;
  if (isDense) {
    readScales<true>(rows);
  } else {
    readScales<false>(rows);
  }
  if (scanSpec_->valueHook()) {
    if (isDense) {
      processValueHook<true>(rows, scanSpec_->valueHook());
    } else {
      processValueHook<false>(rows, scanSpec_->valueHook());
    }
  } else {
    if (isDense) {
      readHelper<true>(rows, ExtractToReader(this));
    } else {
      readHelper<false>(rows, ExtractToReader(this));
    }
  }
}

template <typename DataT>
void SelectiveDecimalColumnReader<DataT>::getValues(
    RowSet rows,
    VectorPtr* result) {
  auto nullsPtr =
      resultNulls() ? resultNulls()->template as<uint64_t>() : nullptr;
  auto scales = scaleBuffer_->as<int64_t>();
  auto values = values_->asMutable<DataT>();

  DecimalUtil::fillDecimals<DataT>(
      values, nullsPtr, values, scales, numValues_, scale_);

  rawValues_ = values_->asMutable<char>();
  getIntValues(rows, requestedType_, result);
}

template class SelectiveDecimalColumnReader<int64_t>;
template class SelectiveDecimalColumnReader<int128_t>;

} // namespace facebook::velox::dwrf
