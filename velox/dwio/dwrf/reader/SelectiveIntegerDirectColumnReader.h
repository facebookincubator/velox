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

#include "velox/dwio/common/SelectiveIntegerColumnReader.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

class SelectiveIntegerDirectColumnReader
    : public dwio::common::SelectiveIntegerColumnReader {
  void init(DwrfParams& params, uint32_t numBytes) {
    format_ = params.stripeStreams().format();
    if (format_ == DwrfFormat::kDwrf) {
      initDwrf(params, numBytes);
    } else {
      VELOX_CHECK(format_ == DwrfFormat::kOrc);
      initOrc(params, numBytes);
    }
  }

  void initDwrf(DwrfParams& params, uint32_t numBytes) {
    auto& stripe = params.stripeStreams();
    EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};
    auto data = encodingKey.forKind(proto::Stream_Kind_DATA);
    bool dataVInts = stripe.getUseVInts(data);

    auto decoder = createDirectDecoder<true>(
        stripe.getStream(data, true), dataVInts, numBytes);
    directDecoder =
        dynamic_cast<dwio::common::DirectDecoder<true>*>(decoder.release());
    VELOX_CHECK(directDecoder);
    ints.reset(directDecoder);
  }

  void initOrc(DwrfParams& params, uint32_t numBytes) {
    auto& stripe = params.stripeStreams();
    EncodingKey encodingKey{nodeType_->id, params.flatMapContext().sequence};
    auto data = encodingKey.forKind(proto::orc::Stream_Kind_DATA);
    bool dataVInts = stripe.getUseVInts(data);

    auto encoding = stripe.getEncodingOrc(encodingKey);
    rleVersion_ = convertRleVersion(encoding.kind());
    auto decoder = createRleDecoder<true>(
        stripe.getStream(data, true),
        rleVersion_,
        params.pool(),
        dataVInts,
        numBytes);
    if (rleVersion_ == velox::dwrf::RleVersion_1) {
      rleDecoderV1 =
          dynamic_cast<velox::dwrf::RleDecoderV1<true>*>(decoder.release());
      VELOX_CHECK(rleDecoderV1);
      ints.reset(rleDecoderV1);
    } else {
      VELOX_CHECK(rleVersion_ == velox::dwrf::RleVersion_2);
      rleDecoderV2 =
          dynamic_cast<velox::dwrf::RleDecoderV2<true>*>(decoder.release());
      VELOX_CHECK(rleDecoderV2);
      ints.reset(rleDecoderV2);
    }
  }

 public:
  using ValueType = int64_t;

  SelectiveIntegerDirectColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      uint32_t numBytes,
      common::ScanSpec& scanSpec)
      : SelectiveIntegerColumnReader(
            std::move(requestedType),
            params,
            scanSpec,
            dataType->type) {
    init(params, numBytes);
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
    SelectiveColumnReader::seekToRowGroup(index);
    auto positionsProvider = formatData_->seekToRowGroup(index);
    ints->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());
  }

  uint64_t skip(uint64_t numValues) override;

  void read(vector_size_t offset, RowSet rows, const uint64_t* incomingNulls)
      override;

  template <typename ColumnVisitor>
  void readWithVisitor(RowSet rows, ColumnVisitor visitor);

 private:
  dwrf::DwrfFormat format_;
  RleVersion rleVersion_;

  union {
    dwio::common::DirectDecoder<true>* directDecoder;
    velox::dwrf::RleDecoderV1<true>* rleDecoderV1;
    velox::dwrf::RleDecoderV2<true>* rleDecoderV2;
  };

  std::unique_ptr<dwio::common::IntDecoder<true>> ints;
};

template <typename ColumnVisitor>
void SelectiveIntegerDirectColumnReader::readWithVisitor(
    RowSet rows,
    ColumnVisitor visitor) {
  vector_size_t numRows = rows.back() + 1;

  VELOX_CHECK(
      format_ == velox::dwrf::DwrfFormat::kDwrf ||
      format_ == velox::dwrf::DwrfFormat::kOrc);
  if (format_ == velox::dwrf::DwrfFormat::kDwrf) {
    if (nullsInReadRange_) {
      directDecoder->readWithVisitor<true>(
          nullsInReadRange_->as<uint64_t>(), visitor);
    } else {
      directDecoder->readWithVisitor<false>(nullptr, visitor);
    }
  } else {
    // orc format does not use int128
    if constexpr (!std::is_same_v<typename ColumnVisitor::DataType, int128_t>) {
      velox::dwio::common::DirectRleColumnVisitor<
          typename ColumnVisitor::DataType,
          typename ColumnVisitor::FilterType,
          typename ColumnVisitor::Extract,
          ColumnVisitor::dense>
          drVisitor(
              visitor.filter(),
              &visitor.reader(),
              visitor.rows(),
              visitor.numRows(),
              visitor.extractValues());

      if (nullsInReadRange_) {
        if (rleVersion_ == velox::dwrf::RleVersion_1) {
          rleDecoderV1->readWithVisitor<true>(
              nullsInReadRange_->as<uint64_t>(), drVisitor);
        } else {
          rleDecoderV2->readWithVisitor<true>(
              nullsInReadRange_->as<uint64_t>(), drVisitor);
        }
      } else {
        if (rleVersion_ == velox::dwrf::RleVersion_1) {
          rleDecoderV1->readWithVisitor<false>(nullptr, drVisitor);
        } else {
          rleDecoderV2->readWithVisitor<false>(nullptr, drVisitor);
        }
      }
    } else {
      VELOX_UNREACHABLE(
          "SelectiveIntegerDirectColumnReader::readWithVisitor get int128_t");
    }
  }
  readOffset_ += numRows;
}
} // namespace facebook::velox::dwrf
