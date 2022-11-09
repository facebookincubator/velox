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

#include "velox/dwio/parquet/reader/NestedIntegerColumnReader.h"

#include "velox/dwio/parquet/reader/ParquetColumnReader.h"

namespace facebook::velox::parquet {

void NestedIntegerColumnReader::prepareRead(vector_size_t offset, RowSet rows) {
  ParquetNestedLeafColumnReader::prepareRead(offset, rows);
  valueSize_ = parquetSizeOfIntKind(type_->kind());
}

void NestedIntegerColumnReader::decodePage(
    std::shared_ptr<ParquetDataPage> dataPage,
    uint32_t numValues,
    uint32_t outputOffset) {
  switch (valueSize_) {
    case 2:
      ensureValuesCapacity<int16_t>(numLeafRowsInBatch_);
      decodePageTyped<int16_t>(dataPage, numValues, outputOffset);
      break;

    case 4:
      ensureValuesCapacity<int32_t>(numLeafRowsInBatch_);
      decodePageTyped<int32_t>(dataPage, numValues, outputOffset);
      break;

    case 8:
      ensureValuesCapacity<int64_t>(numLeafRowsInBatch_);
      decodePageTyped<int64_t>(dataPage, numValues, outputOffset);
      break;

    default:
      VELOX_FAIL("Unsupported valueSize_ {}", valueSize_);
  }

  dataPage->pageData_ += numValues * valueSize_;
  dataPage->numRowsInPage_ -= numValues;
  dataPage->encodedDataSize_ -= numValues * valueSize_;
}

template <typename T>
void NestedIntegerColumnReader::decodePageTyped(
    std::shared_ptr<ParquetDataPage> dataPage,
    uint32_t numValues,
    uint32_t outputOffset) {
  if (dataPage->encoding_ == thrift::Encoding::PLAIN) {
    directDecoder_ = std::make_unique<dwio::common::DirectDecoder<true>>(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            dataPage->pageData_, dataPage->encodedDataSize_),
        valueSize_);

    directDecoder_->next<T>(
        reinterpret_cast<int64_t*>(
            reinterpret_cast<int8_t*>(rawValues_) + outputOffset * valueSize_),
        numValues,
        resultNulls_->as<uint64_t>(),
        outputOffset);
  } else {
    VELOX_NYI();
  }
}

void NestedIntegerColumnReader::getValues(RowSet /*rows*/, VectorPtr* result) {
  getIntValues(outputRows_, nodeType_->type, result);
}

} // namespace facebook::velox::parquet