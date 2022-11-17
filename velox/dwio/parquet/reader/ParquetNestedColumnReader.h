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
#include "velox/dwio/common/SelectiveColumnReader.h"
#include "velox/dwio/parquet/reader/NestedStructureDecoder.h"
#include "velox/dwio/parquet/reader/ParquetData.h"
#include "velox/dwio/parquet/reader/ParquetTypeWithId.h"

namespace facebook::velox::parquet {

class ParquetNestedColumnReader : public dwio::common::SelectiveColumnReader {
 public:
  ParquetNestedColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      ParquetParams& params,
      common::ScanSpec& scanSpec,
      common::ScanSpec& topLevelScanSpec)
      : SelectiveColumnReader(
            requestedType,
            params,
            scanSpec,
            requestedType->type),
        topLevelScanSpec_(&topLevelScanSpec) {
    auto type = &dynamic_cast<const ParquetTypeWithId&>(*requestedType);
    while (type->parent) {
      maxRepeats_.push_back(type->maxRepeat_);
      maxDefines_.push_back(type->maxDefine_);
      type = dynamic_cast<const ParquetTypeWithId*>(type->parent);
    }
    std::reverse(maxRepeats_.begin(), maxRepeats_.end());
    std::reverse(maxDefines_.begin(), maxDefines_.end());

    level_ = maxRepeats_.size() - 1;
  }

  virtual void enqueueRowGroup(
      uint32_t index,
      dwio::common::BufferedInput& input) = 0;

  virtual std::vector<std::shared_ptr<NestedData>> read(
      uint64_t offset,
      RowSet rows) = 0;

  void read(
      vector_size_t offset,
      RowSet rows,
      const uint64_t* FOLLY_NULLABLE /*incomingNulls*/) override {}

 protected:
  velox::common::ScanSpec* const FOLLY_NONNULL topLevelScanSpec_ = nullptr;

  std::vector<uint32_t> maxRepeats_;
  std::vector<uint32_t> maxDefines_;
  uint8_t level_;
};

class ParquetRepeatedColumnReader : public ParquetNestedColumnReader {
 public:
  ParquetRepeatedColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      ParquetParams& params,
      common::ScanSpec& scanSpec,
      common::ScanSpec& topLevelScanSpec)
      : ParquetNestedColumnReader(
            requestedType,
            params,
            scanSpec,
            topLevelScanSpec) {}

  void seekToRowGroup(uint32_t index) override;
};

class ParquetNestedLeafColumnReader : public ParquetNestedColumnReader {
 public:
  ParquetNestedLeafColumnReader(
      std::shared_ptr<const dwio::common::TypeWithId> requestedType,
      ParquetParams& params,
      common::ScanSpec& scanSpec,
      common::ScanSpec& topLevelScanSpec)
      : ParquetNestedColumnReader(
            requestedType,
            params,
            scanSpec,
            topLevelScanSpec) {
    dwio::common::ensureCapacity<uint8_t>(repetitionLevels_, 0, &memoryPool_);
    dwio::common::ensureCapacity<uint8_t>(definitionLevels_, 0, &memoryPool_);
  }

  void seekToRowGroup(uint32_t index) override;

  void enqueueRowGroup(uint32_t index, dwio::common::BufferedInput& input)
      override;

  std::vector<std::shared_ptr<NestedData>> read(
      uint64_t offset,
      RowSet topLevelRows) override;

 protected:
  virtual void prepareRead(vector_size_t offset, RowSet rows);

  void readPages(int64_t topLevelRowCount);

  std::shared_ptr<NestedData> parseNestedData(uint8_t level);

  void readRepDefs(std::shared_ptr<ParquetDataPage> page);

  void moveRepDefs();

  void readNulls(int32_t numValues);

  void readNoFilter(uint64_t offset, RowSet topRows);

  virtual uint64_t decodePage(
      std::shared_ptr<ParquetDataPage> dataPage,
      uint32_t outputOffset) = 0;

  virtual void decodePage(
      std::shared_ptr<ParquetDataPage> dataPage,
      RowSet leafRowsInPage,
      uint32_t outputOffset) = 0;

  std::unique_ptr<PageReader> reader_;
  std::deque<std::shared_ptr<ParquetDataPage>> dataPages_;
  std::shared_ptr<ParquetDictionaryPage> dictionaryPage_;

  BufferPtr repetitionLevels_;
  BufferPtr definitionLevels_;

  // The leaf level row counts in the current batch. May contain rows in
  // multiple pages. {Non-null rows} is a subset of {Non-empty rows}, and
  // {Non-empty rows} is a subset of all {rows}. The number of rows is the
  // number of repetition or definition levels.
  uint64_t numRowsInBatch_;
  uint64_t numNonEmptyRowsInBatch_;
  uint64_t numNonNullRowsInBatch_;
  uint64_t numRemainingRowsInLastBatch_ = 0;

  std::unique_ptr<NestedStructureDecoder> nestedStructureDecoder_;
  std::shared_ptr<ParquetDataPage> getDataPage();
};

} // namespace facebook::velox::parquet
