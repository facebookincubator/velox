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

#include "velox/dwio/parquet/reader/NestedStructureDecoder.h"
#include "velox/dwio/parquet/reader/ParquetNestedColumnReader.h"

namespace facebook::velox::parquet {

class NestedIntegerColumnReader : public ParquetNestedLeafColumnReader {
 public:
  NestedIntegerColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      ParquetParams& params,
      common::ScanSpec& scanSpec,
      common::ScanSpec& topLevelScanSpec)
      : ParquetNestedLeafColumnReader(
            requestedType,
            params,
            scanSpec,
            topLevelScanSpec) {}

  void prepareRead(vector_size_t offset, RowSet /* rows */) override;

  uint64_t decodePage(
      std::shared_ptr<ParquetDataPage> dataPage,
      uint32_t outputOffset) override;

  void decodePage(
      std::shared_ptr<ParquetDataPage> dataPage,
      RowSet leafRowsInPage,
      uint32_t outputOffset) override {
    VELOX_NYI();
  }

  void getValues(RowSet rows, VectorPtr* FOLLY_NONNULL result) override;

 private:
  template <typename T>
  uint64_t decodePageTyped(
      std::shared_ptr<ParquetDataPage> dataPage,
      uint64_t numNonEmptyValues,
      uint64_t outputOffset);
};

} // namespace facebook::velox::parquet