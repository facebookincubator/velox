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

#pragma once

#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

class SelectiveByteRleColumnReader
    : public dwio::common::SelectiveColumnReader {
 public:
  using ValueType = int8_t;

  SelectiveByteRleColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec,
      bool isBool)
      : SelectiveColumnReader(
            std::move(requestedType),
            params,
            scanSpec,
            dataType->type) {
    EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};
    auto& stripe = params.stripeStreams();
    if (isBool) {
      boolRle_ = createBooleanRleDecoder(
          stripe.getStream(encodingKey.forKind(proto::Stream_Kind_DATA), true),
          encodingKey);
    } else {
      byteRle_ = createByteRleDecoder(
          stripe.getStream(encodingKey.forKind(proto::Stream_Kind_DATA), true),
          encodingKey);
    }
  }

  void seekToRowGroup(uint32_t index) override {
    SelectiveColumnReader::seekToRowGroup(index);
    auto positionsProvider = formatData_->seekToRowGroup(index);
    if (boolRle_) {
      boolRle_->seekToRowGroup(positionsProvider);
    } else {
      byteRle_->seekToRowGroup(positionsProvider);
    }

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* nulls) override;

  void getValues(RowSet rows, VectorPtr* result) override {
    switch (nodeType_->type->kind()) {
      case TypeKind::BOOLEAN:
        getFlatValues<int8_t, bool>(rows, result);
        break;
      case TypeKind::TINYINT:
        getFlatValues<int8_t, int8_t>(rows, result);
        break;
      case TypeKind::SMALLINT:
        getFlatValues<int8_t, int16_t>(rows, result);
        break;
      case TypeKind::INTEGER:
        getFlatValues<int8_t, int32_t>(rows, result);
        break;
      case TypeKind::BIGINT:
        getFlatValues<int8_t, int64_t>(rows, result);
        break;
      default:
        VELOX_FAIL(
            "Result type not supported in ByteRLE encoding: {}",
            nodeType_->type->toString());
    }
  }

  bool useBulkPath() const override {
    return false;
  }

 private:
  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor);

  template <bool isDense, typename ExtractValues>
  void processFilter(
      common::Filter* filter,
      ExtractValues extractValues,
      RowSet rows);

  template <bool isDence>
  void processValueHook(RowSet rows, ValueHook* hook);

  template <typename TFilter, bool isDense, typename ExtractValues>
  void
  readHelper(common::Filter* filter, RowSet rows, ExtractValues extractValues);

  std::unique_ptr<ByteRleDecoder> byteRle_;
  std::unique_ptr<BooleanRleDecoder> boolRle_;
};

template <typename ColumnVisitor>
void SelectiveByteRleColumnReader::readWithVisitor(
    RowSet rows,
    ColumnVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;
  if (boolRle_) {
    if (nullsInReadRange_) {
      boolRle_->readWithVisitor<true>(
          nullsInReadRange_->as<uint64_t>(), visitor);
    } else {
      boolRle_->readWithVisitor<false>(nullptr, visitor);
    }
  } else {
    if (nullsInReadRange_) {
      byteRle_->readWithVisitor<true>(
          nullsInReadRange_->as<uint64_t>(), visitor);
    } else {
      byteRle_->readWithVisitor<false>(nullptr, visitor);
    }
  }
  readOffset_ += numRows;
}

template <typename TFilter, bool isDense, typename ExtractValues>
void SelectiveByteRleColumnReader::readHelper(
    common::Filter* filter,
    RowSet rows,
    ExtractValues extractValues) {
  readWithVisitor(
      rows,
      dwio::common::ColumnVisitor<int8_t, TFilter, ExtractValues, isDense>(
          *reinterpret_cast<TFilter*>(filter), this, rows, extractValues));
}

template <bool isDense, typename ExtractValues>
void SelectiveByteRleColumnReader::processFilter(
    common::Filter* filter,
    ExtractValues extractValues,
    RowSet rows) {
  using common::FilterKind;
  switch (filter ? filter->kind() : FilterKind::kAlwaysTrue) {
    case FilterKind::kAlwaysTrue:
      readHelper<common::AlwaysTrue, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kIsNull:
      filterNulls<int8_t>(
          rows,
          true,
          !std::is_same_v<decltype(extractValues), dwio::common::DropValues>);
      break;
    case FilterKind::kIsNotNull:
      if (std::is_same_v<decltype(extractValues), dwio::common::DropValues>) {
        filterNulls<int8_t>(rows, false, false);
      } else {
        readHelper<common::IsNotNull, isDense>(filter, rows, extractValues);
      }
      break;
    case FilterKind::kBigintRange:
      readHelper<common::BigintRange, isDense>(filter, rows, extractValues);
      break;
    case FilterKind::kNegatedBigintRange:
      readHelper<common::NegatedBigintRange, isDense>(
          filter, rows, extractValues);
      break;
    case FilterKind::kBigintValuesUsingBitmask:
      readHelper<common::BigintValuesUsingBitmask, isDense>(
          filter, rows, extractValues);
      break;
    case FilterKind::kNegatedBigintValuesUsingBitmask:
      readHelper<common::NegatedBigintValuesUsingBitmask, isDense>(
          filter, rows, extractValues);
      break;
    default:
      readHelper<common::Filter, isDense>(filter, rows, extractValues);
      break;
  }
}

template <bool isDense>
void SelectiveByteRleColumnReader::processValueHook(
    RowSet rows,
    ValueHook* hook) {
  using namespace facebook::velox::aggregate;
  switch (hook->kind()) {
    case aggregate::AggregationHook::kSumBigintToBigint:
      readHelper<common::AlwaysTrue, isDense>(
          &dwio::common::alwaysTrue(),
          rows,
          dwio::common::ExtractToHook<SumHook<int64_t, int64_t>>(hook));
      break;
    default:
      readHelper<common::AlwaysTrue, isDense>(
          &dwio::common::alwaysTrue(),
          rows,
          dwio::common::ExtractToGenericHook(hook));
  }
}

} // namespace facebook::velox::dwrf
