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

#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/ColumnVisitors.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

class SelectiveShortDecimalColumnReader
    : public dwio::common::SelectiveColumnReader {
  void init(DwrfParams& params) {
    format_ = params.stripeStreams().format();
    if (format_ == DwrfFormat::kDwrf) {
      initDwrf(params);
    } else {
      VELOX_CHECK(format_ == DwrfFormat::kOrc);
      initOrc(params);
    }
  }

  void initDwrf(DwrfParams& params) {
    VELOX_FAIL("dwrf unsupport decimal");
  }

  void initOrc(DwrfParams& params) {
    const auto& stripe = params.stripeStreams();
    EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};

    auto values = encodingKey.forKind(proto::orc::Stream_Kind_DATA);
    auto scales = encodingKey.forKind(proto::orc::Stream_Kind_SECONDARY);

    bool valuesVInts = stripe.getUseVInts(values);
    bool scalesVInts = stripe.getUseVInts(scales);

    auto encoding = stripe.getEncodingOrc(encodingKey);
    auto encodingKind = encoding.kind();
    VELOX_CHECK(
        encodingKind == proto::orc::ColumnEncoding_Kind_DIRECT ||
        encodingKind == proto::orc::ColumnEncoding_Kind_DIRECT_V2);

    version_ = convertRleVersion(encodingKind);

    valueDecoder_ = createDirectDecoder<true>(
        stripe.getStream(values, true),
        valuesVInts,
        facebook::velox::dwio::common::LONG_BYTE_SIZE);

    scaleDecoder_ = createRleDecoder<true>(
        stripe.getStream(scales, true),
        version_,
        params.pool(),
        scalesVInts,
        facebook::velox::dwio::common::LONG_BYTE_SIZE);
  }

 public:
  using ValueType = int64_t;

  static const uint32_t MAX_PRECISION_64 = 18;
  static const uint32_t MAX_PRECISION_128 = 38;
  static const int64_t POWERS_OF_TEN[MAX_PRECISION_64 + 1];

  SelectiveShortDecimalColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& nodeType,
      const TypePtr& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec)
      : SelectiveColumnReader(nodeType, params, scanSpec, nodeType->type) {
    precision_ = dataType->asShortDecimal().precision();
    scale_ = dataType->asShortDecimal().scale();
    init(params);
  }

  bool hasBulkPath() const override {
    if (format_ == velox::dwrf::DwrfFormat::kDwrf) {
      return true;
    } else {
      // TODO: zuochunwei, need support useBulkPath() for kOrc
      return false;
    }
  }

  void seekToRowGroup(uint32_t index) override {
    auto positionsProvider = formatData_->seekToRowGroup(index);
    valueDecoder_->seekToRowGroup(positionsProvider);
    scaleDecoder_->seekToRowGroup(positionsProvider);
    // Check that all the provided positions have been consumed.
    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override {
    numValues = SelectiveColumnReader::skip(numValues);
    valueDecoder_->skip(numValues);
    scaleDecoder_->skip(numValues);
    return numValues;
  }

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  void getValues(RowSet rows, VectorPtr* result) override;

 private:
  template <bool dense>
  void processValueHook(RowSet rows, ValueHook* hook) {
    switch (hook->kind()) {
      case aggregate::AggregationHook::kShortDecimalMax:
        readHelper<dense, velox::common::AlwaysTrue>(
            &dwio::common::alwaysTrue(),
            rows,
            dwio::common::ExtractToHook<
                aggregate::MinMaxHook<UnscaledShortDecimal, false>>(hook));
        break;
      case aggregate::AggregationHook::kShortDecimalMin:
        readHelper<dense, velox::common::AlwaysTrue>(
            &dwio::common::alwaysTrue(),
            rows,
            dwio::common::ExtractToHook<
                aggregate::MinMaxHook<UnscaledShortDecimal, true>>(hook));
        break;
      default:
        readHelper<dense, velox::common::AlwaysTrue>(
            &dwio::common::alwaysTrue(),
            rows,
            dwio::common::ExtractToGenericHook(hook));
    }
  }

  template <bool dense, typename ExtractValues>
  void processFilter(
      velox::common::Filter* filter,
      ExtractValues extractValues,
      RowSet rows) {
    switch (filter ? filter->kind() : velox::common::FilterKind::kAlwaysTrue) {
      case velox::common::FilterKind::kAlwaysTrue:
        readHelper<dense, velox::common::AlwaysTrue>(
            filter, rows, extractValues);
        break;
      default:
        VELOX_FAIL("TODO: orc short decimal process filter unsupport cases");
        break;
    }
  }

  template <bool dense, typename Filter, typename ExtractValues>
  void readHelper(
      velox::common::Filter* filter,
      RowSet rows,
      ExtractValues extractValues) {
    VELOX_CHECK(filter->kind() == velox::common::FilterKind::kAlwaysTrue);

    vector_size_t numRows = rows.back() + 1;

    // step1: read scales
    // 1.1 read scales into values_(rawValues_)
    if (version_ == velox::dwrf::RleVersion_1) {
      auto scaleDecoderV1 =
          dynamic_cast<RleDecoderV1<true>*>(scaleDecoder_.get());
      if (nullsInReadRange_) {
        scaleDecoderV1->readWithVisitor<true>(
            nullsInReadRange_->as<uint64_t>(),
            facebook::velox::dwio::common::DirectRleColumnVisitor<
                int64_t,
                velox::common::AlwaysTrue,
                decltype(extractValues),
                dense>(dwio::common::alwaysTrue(), this, rows, extractValues));
      } else {
        scaleDecoderV1->readWithVisitor<false>(
            nullptr,
            facebook::velox::dwio::common::DirectRleColumnVisitor<
                int64_t,
                velox::common::AlwaysTrue,
                decltype(extractValues),
                dense>(dwio::common::alwaysTrue(), this, rows, extractValues));
      }
    } else {
      auto scaleDecoderV2 =
          dynamic_cast<RleDecoderV2<true>*>(scaleDecoder_.get());
      if (nullsInReadRange_) {
        scaleDecoderV2->readWithVisitor<true>(
            nullsInReadRange_->as<uint64_t>(),
            facebook::velox::dwio::common::DirectRleColumnVisitor<
                int64_t,
                velox::common::AlwaysTrue,
                decltype(extractValues),
                dense>(dwio::common::alwaysTrue(), this, rows, extractValues));
      } else {
        scaleDecoderV2->readWithVisitor<false>(
            nullptr,
            facebook::velox::dwio::common::DirectRleColumnVisitor<
                int64_t,
                velox::common::AlwaysTrue,
                decltype(extractValues),
                dense>(dwio::common::alwaysTrue(), this, rows, extractValues));
      }
    }

    // 1.2 copy scales from values_(rawValues_) into scaleBuffer_ before reading
    // values
    velox::dwio::common::ensureCapacity<int64_t>(
        scaleBuffer_, numValues_, &memoryPool_);
    scaleBuffer_->setSize(numValues_ * sizeof(int64_t));
    memcpy(
        scaleBuffer_->asMutable<char>(),
        rawValues_,
        numValues_ * sizeof(int64_t));

    // step2: read values
    auto numScales = numValues_;
    numValues_ = 0; // reset numValues_ before reading values

    // read values into values_(rawValues_)
    facebook::velox::dwio::common::ColumnVisitor<
        int64_t,
        velox::common::AlwaysTrue,
        decltype(extractValues),
        dense>
        columnVisitor(dwio::common::alwaysTrue(), this, rows, extractValues);

    auto valueDecoder = dynamic_cast<velox::dwio::common::DirectDecoder<true>*>(
        valueDecoder_.get());
    if (nullsInReadRange_) {
      valueDecoder->readWithVisitor<true>(
          nullsInReadRange_->as<uint64_t>(), columnVisitor, false);
    } else {
      valueDecoder->readWithVisitor<false>(nullptr, columnVisitor, false);
    }

    VELOX_CHECK(numScales == numValues_);

    // step3: change readOffset_
    readOffset_ += numRows;
  }

 private:
  dwrf::DwrfFormat format_;
  RleVersion version_;

  std::unique_ptr<dwio::common::IntDecoder<true>> valueDecoder_;
  std::unique_ptr<dwio::common::IntDecoder<true>> scaleDecoder_;

  BufferPtr scaleBuffer_; // to save scales

  int32_t precision_ = 0;
  int32_t scale_ = 0;
};

} // namespace facebook::velox::dwrf
