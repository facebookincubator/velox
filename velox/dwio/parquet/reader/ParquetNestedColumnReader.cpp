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

#include "velox/dwio/parquet/reader/ParquetNestedColumnReader.h"

#include "velox/dwio/common/BufferUtil.h"

namespace facebook::velox::parquet {

void ParquetRepeatedColumnReader::seekToRowGroup(uint32_t index) {
  SelectiveColumnReader::seekToRowGroup(index);
  scanState().clear();
  readOffset_ = 0;
}

void ParquetNestedLeafColumnReader::seekToRowGroup(uint32_t index) {
  SelectiveColumnReader::seekToRowGroup(index);
  scanState().clear();
  readOffset_ = 0;
  formatData_->as<ParquetData>().seekToRowGroup(index);
}

void ParquetNestedLeafColumnReader::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  formatData_->as<ParquetData>().enqueueRowGroup(index, input);
}

void ParquetNestedLeafColumnReader::prepareRead(
    vector_size_t offset,
    RowSet /* rows */) {
  seekTo(offset, scanSpec_->readsNullsOnly());

  outputRows_.clear();

  numValues_ = 0;
  mayGetValues_ = true;
  numLeafRowsInBatch_ = 0;
  numNonNullLeafRowsInBatch_ = 0;
}

std::vector<std::shared_ptr<NestedData>> ParquetNestedLeafColumnReader::read(
    uint64_t offset,
    RowSet topLevelRows) {
  prepareRead(offset, topLevelRows);

  auto maxNumTopLevelRows = topLevelRows.back() + 1;

  readPages(maxNumTopLevelRows);

  // TODO: check stats for mayHaveNulls
  auto nestedData = NestedStructureDecoder::readOffsetsAndNullsForAllLevels(
      repetitionLevels_->as<uint8_t>(),
      definitionLevels_->as<uint8_t>(),
      numLeafRowsInBatch_,
      maxRepeats_,
      maxDefines_,
      memoryPool_);

  // The leaf level nulls need to be set before the pages are decoded.
  resultNulls_ = nestedData[level_]->nulls;
  numNonNullLeafRowsInBatch_ = nestedData[level_]->numNonNulls;
  if (numNonNullLeafRowsInBatch_ == 0) {
    allNull_ = true;
  } else if (numNonNullLeafRowsInBatch_ < numLeafRowsInBatch_) {
    anyNulls_ = true;
  }
  nestedData[level_] = nullptr;

  if (!topLevelScanSpec_->hasFilter()) {
    readNoFilter(offset, topLevelRows);
  } else {
    VELOX_NYI();
  }

  return nestedData;
}

void ParquetNestedLeafColumnReader::readPages(int64_t numTopLevelRows) {
  int64_t numLeafRowsInPages = 0;

  while (numTopLevelRows >= 0) {
    // ParquetNestedLeafColumnReader must have non null repetition and
    // definitions
    auto currentRepDefSize = repetitionLevels_->size();
    std::shared_ptr<ParquetPage> page;

    if (numRemainingRepDefsInLastBatch_ > 0) {
      VELOX_CHECK_LE(dataPages_.size(), 1);

      page = dataPages_.front();
      // TODO: Avoid moving
      moveRepDefs();
    } else {
      auto& dataReader = formatData_->as<ParquetData>();
      page = dataReader.readNextPage();

      if (page != nullptr) {
        if (page->isDictionary_) {
          dictionaryPage_ = static_pointer_cast<ParquetDictionaryPage>(page);
          page = dataReader.readNextPage();
          VELOX_CHECK(page);
          VELOX_CHECK(static_pointer_cast<ParquetDataPage>(page));
        }

        dataPages_.push_back(static_pointer_cast<ParquetDataPage>(page));

        // Append this page's rep/def to repetitionLevels_ and definitionLevels_
        readRepDefs(static_pointer_cast<ParquetDataPage>(page));
      } else {
        // End of ColumnChunk means the top level row is ending, since a row
        // cannot stride multiple RowGroups.
        numTopLevelRows--;
        VELOX_CHECK_EQ(numTopLevelRows, -1);
      }
    }

    // TODO: push down to repeatDecoder_->next<uint8_t>()
    if (page) {
      uint32_t numLeafRowsInPage = 0;
      auto repetitions =
          repetitionLevels_->asMutable<uint8_t>() + currentRepDefSize;
      while (numLeafRowsInPage < page->numRowsInPage_ && numTopLevelRows >= 0) {
        bool isTopLevel = (repetitions[numLeafRowsInPage++] == 0);
        numTopLevelRows -= isTopLevel;
      }
      numLeafRowsInBatch_ += numLeafRowsInPage;
      numLeafRowsInPages += page->numRowsInPage_;
    }
  }

  numRemainingRepDefsInLastBatch_ = numLeafRowsInPages - numLeafRowsInBatch_;
}

void ParquetNestedLeafColumnReader::readRepDefs(
    std::shared_ptr<ParquetDataPage> page) {
  auto numRowsInPage = page->numRowsInPage_;

  if (maxRepeats_[level_] > 0) {
    auto currentSize = repetitionLevels_->size();

    // ensureCapacity would update the buffer size.
    dwio::common::ensureCapacity<uint8_t>(
        repetitionLevels_, currentSize + numRowsInPage, &memoryPool_);
    auto repetitions = repetitionLevels_->asMutable<uint8_t>() + currentSize;

    page->repeatDecoder_->next<uint8_t>(repetitions, page->numRowsInPage_);
  }

  if (maxDefines_[level_] > 0) {
    auto currentSize = definitionLevels_->size();
    dwio::common::ensureCapacity<uint8_t>(
        definitionLevels_, currentSize + numRowsInPage, &memoryPool_);
    auto definitions = definitionLevels_->asMutable<uint8_t>() + currentSize;

    // TODO: record number of nulls in each page
    page->defineDecoder_->next<uint8_t>(definitions, numRowsInPage);
  }
}

void ParquetNestedLeafColumnReader::moveRepDefs() {
  VELOX_CHECK_EQ(repetitionLevels_->size(), definitionLevels_->size());
  VELOX_CHECK_GT(numRemainingRepDefsInLastBatch_, 0);

  std::memmove(
      (void*)(repetitionLevels_->as<uint8_t>() + numLeafRowsInBatch_),
      (void*)(repetitionLevels_->asMutable<uint8_t>()),
      BaseVector::byteSize<uint8_t>(numRemainingRepDefsInLastBatch_));
  repetitionLevels_->setSize(numRemainingRepDefsInLastBatch_);

  std::memmove(
      (void*)(definitionLevels_->as<uint8_t>() + numLeafRowsInBatch_),
      (void*)(definitionLevels_->asMutable<uint8_t>()),
      BaseVector::byteSize<uint8_t>(numRemainingRepDefsInLastBatch_));
  definitionLevels_->setSize(numRemainingRepDefsInLastBatch_);

  numLeafRowsInBatch_ += numRemainingRepDefsInLastBatch_;
  numRemainingRepDefsInLastBatch_ = 0;
}

void ParquetNestedLeafColumnReader::readNoFilter(
    uint64_t /* offset */,
    RowSet topLevelRows) {
  auto maxNumTopLevelRows = topLevelRows.back() + 1;
  if (maxNumTopLevelRows == topLevelRows.size()) {
    // full
    int32_t numRowsRead = 0;
    //    for (int i = 0; i < dataPages_.size(); i++) {
    for (auto iter = dataPages_.begin(); iter < dataPages_.end(); iter++) {
      auto dataPage = *iter;

      auto numRowsToRead =
          std::min(dataPage->numRowsInPage_, numLeafRowsInBatch_ - numRowsRead);
      decodePage(dataPage, numRowsToRead, numRowsRead);

      if (dataPage->isEmpty()) {
        // This data page is used up. No need to keep it anymore.
        VELOX_CHECK(iter == dataPages_.begin());
        dataPages_.pop_front();
      }
    }

    // TODO: There is no need to allocate the RowSet here, but getFlatValues
    // requires a RowSet
    outputRows_.resize(numLeafRowsInBatch_);
    std::iota(outputRows_.begin(), outputRows_.end(), 0);
  } else {
    //   Populate leaf level RowSet
    VELOX_NYI();
  }

  numValues_ = numLeafRowsInBatch_;
  outputRows_.resize(numValues_);
}

} // namespace facebook::velox::parquet