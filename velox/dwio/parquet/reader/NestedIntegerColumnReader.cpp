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

uint64_t NestedIntegerColumnReader::decodePage(
    std::shared_ptr<ParquetDataPage> dataPage,
    uint32_t outputOffset) {
  uint32_t numNonEmptyValues = dataPage->numNonEmptyRowsInBatch;
  switch (valueSize_) {
    case 2:
      return decodePageTyped<int16_t>(
          dataPage, numNonEmptyValues, outputOffset);
    case 4:
      return decodePageTyped<int32_t>(
          dataPage, numNonEmptyValues, outputOffset);
    case 8:
      return decodePageTyped<int64_t>(
          dataPage, numNonEmptyValues, outputOffset);
    default:
      VELOX_FAIL("Unsupported valueSize_ {}", valueSize_);
  }
}

template <typename T>
uint64_t NestedIntegerColumnReader::decodePageTyped(
    std::shared_ptr<ParquetDataPage> dataPage,
    uint64_t numNonEmptyValues,
    uint64_t outputOffset) {
  auto numNonNullValues = numNonEmptyValues;
  if (dataPage->encoding == thrift::Encoding::PLAIN) {
    auto directDecoder = std::make_unique<dwio::common::DirectDecoder<true>>(
        std::make_unique<dwio::common::SeekableArrayInputStream>(
            dataPage->pageData, dataPage->encodedDataSize),
        valueSize_);

    auto bytesReadInStream = directDecoder->next<T>(
        reinterpret_cast<int64_t*>(rawValues_),
        outputOffset,
        numNonEmptyValues,
        resultNulls_->as<uint64_t>());

    return bytesReadInStream;
  } else {
    VELOX_NYI();
  }
}

void NestedIntegerColumnReader::getValues(RowSet /*rows*/, VectorPtr* result) {
  getIntValues(outputRows_, nodeType_->type, result);
}

} // namespace facebook::velox::parquet