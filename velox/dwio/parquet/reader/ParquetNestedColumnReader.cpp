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

#include "velox/common/base/Exceptions.h"
#include "velox/dwio/common/BufferUtil.h"
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"

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
  numRowsInBatch_ = 0;
  numNonEmptyRowsInBatch_ = 0;
  numNonNullRowsInBatch_ = 0;
}

std::vector<std::shared_ptr<NestedData>> ParquetNestedLeafColumnReader::read(
    uint64_t offset,
    RowSet topLevelRows) {
  prepareRead(offset, topLevelRows);

  auto maxNumTopLevelRows = topLevelRows.back() + 1;

  readPages(maxNumTopLevelRows);

  std::vector<std::shared_ptr<NestedData>> nestedDataAllLevels;
  for (int i = 0; i < maxRepeats_.size(); i++) {
    auto nestedData = parseNestedData(i);
    nestedDataAllLevels.push_back(nestedData);
  }

  // Prepare leaf level nulls before decoding the values. The offsets and
  // lengths are not used.
  auto leafNestedData = nestedDataAllLevels[level_];
  numNonEmptyRowsInBatch_ = leafNestedData->numNonEmptyCollections;
  numNonNullRowsInBatch_ = leafNestedData->numNonNullCollections;
  resultNulls_ = leafNestedData->nulls;
  allNull_ = numNonNullRowsInBatch_ == numNonEmptyRowsInBatch_ ? true : false;
  anyNulls_ = numNonNullRowsInBatch_ < numNonEmptyRowsInBatch_ ? true : false;
  nestedDataAllLevels[level_] = nullptr;

  if (!topLevelScanSpec_->hasFilter()) {
    readNoFilter(offset, topLevelRows);
  } else {
    VELOX_NYI();
  }

  return nestedDataAllLevels;
}

void ParquetNestedLeafColumnReader::readPages(int64_t numTopLevelRows) {
  int64_t numLeafRowsInPages = 0;

  // Look for numTopLevelRows + 1 number of rows that repetitionLevel == 0, or
  // the end of the RowGroup, whichever is first.
  while (numTopLevelRows >= 0) {
    std::shared_ptr<ParquetPage> page;

    if (numRemainingRowsInLastBatch_ > 0) {
      VELOX_CHECK_LE(dataPages_.size(), 1);
      page = dataPages_.front();
    } else {
      auto& dataReader = formatData_->as<ParquetData>();
      page = dataReader.readNextPage();

      if (page != nullptr) {
        if (page->isDictionary) {
          dictionaryPage_ =
              std::static_pointer_cast<ParquetDictionaryPage>(page);
          page = dataReader.readNextPage();
          VELOX_CHECK(page);
          VELOX_CHECK(std::static_pointer_cast<ParquetDataPage>(page));
        }

        dataPages_.push_back(std::static_pointer_cast<ParquetDataPage>(page));
      } else {
        // End of ColumnChunk means the top level row is ending, since a row
        // cannot stride multiple RowGroups.
        numTopLevelRows--;
        VELOX_CHECK_EQ(numTopLevelRows, -1);
      }
    }

    // Find out how many top level rows there are in this page, and record the
    // number of leaf rows (number of rep/def values) in this page that is
    // within the current batch. If the last top level row is ending in this
    // page, the dataPage's numRowsInBatch may be smaller than its numRowsInPage
    // TODO: push down to repeatDecoder_->next<uint8_t>()
    if (page) {
      auto dataPage = std::static_pointer_cast<ParquetDataPage>(page);

      uint32_t numLeafRowsInPage = 0;
      auto repetitions =
          dataPage->repetitions->asMutable<uint8_t>() + dataPage->repDefOffset;

      while (numLeafRowsInPage < dataPage->numRowsInPage &&
             numTopLevelRows >= 0) {
        // Top level rows cannot be empty, therefore only needs to check
        // repetition level
        bool isTopLevel = (repetitions[numLeafRowsInPage++] == 0);
        numTopLevelRows -= isTopLevel;
      }

      // If the page is not exhausted but enough top level rows collected, then
      // the numLeafRowsInPage's position marks the start of the next top level
      // row in the next batch.
      if (numLeafRowsInPage < dataPage->numRowsInPage && numTopLevelRows < 0) {
        numLeafRowsInPage -= 1;
      }

      numRowsInBatch_ += numLeafRowsInPage;
      dataPage->numRowsInBatch = numLeafRowsInPage;
      numLeafRowsInPages += dataPage->numRowsInPage;
    }
    numRemainingRowsInLastBatch_ = numLeafRowsInPages - numRowsInBatch_;
  }
}

std::shared_ptr<NestedData> ParquetNestedLeafColumnReader::parseNestedData(
    uint8_t level) {
  uint32_t maxRepeat = maxRepeats_[level];
  uint32_t maxDefine = maxDefines_[level];

  std::shared_ptr<NestedData> nestedData = std::make_shared<NestedData>();

  // Calculate offsets and nulls for the whole batch
  dwio::common::ensureCapacity<uint8_t>(
      nestedData->nulls, bits::nbytes(numRowsInBatch_), &memoryPool_);
  dwio::common::ensureCapacity<vector_size_t>(
      nestedData->offsets, numRowsInBatch_ + 1, &memoryPool_);
  auto offsets = nestedData->offsets->asMutable<uint32_t>();
  auto nulls = nestedData->nulls->asMutable<uint64_t>();

  //  uint64_t numRowsParsed = 0;
  uint64_t numNonEmptyCollections = 0;
  uint64_t numNonNullCollections = 0;
  bool wasLastCollectionNull = false;
  int64_t lastOffset = -1;
  for (auto page : dataPages_) {
    auto repetitions = page->repetitions->as<uint8_t>() + page->repDefOffset;
    auto definitions = page->definitions->as<uint8_t>() + page->repDefOffset;

    uint64_t nonEmptyOffset = numNonEmptyCollections;
    uint64_t nonNullOffset = numNonNullCollections;
    NestedStructureDecoder::readOffsetsAndNulls(
        repetitions,
        definitions,
        page->numRowsInBatch,
        maxRepeat,
        maxDefine,
        lastOffset,
        wasLastCollectionNull,
        offsets,
        nulls,
        numNonEmptyCollections,
        numNonNullCollections,
        memoryPool_);

    if (level == level_) {
      page->repDefOffset += page->numRowsInBatch;
      page->numNonEmptyRowsInBatch = numNonEmptyCollections - nonEmptyOffset;
      page->numNonNullRowsInBatch = numNonNullCollections - nonNullOffset;
    }
  }

  auto endOffset = lastOffset + !wasLastCollectionNull;
  offsets[numNonEmptyCollections] = endOffset;

  // Calculate lengths for the whole batch
  dwio::common::ensureCapacity<vector_size_t>(
      nestedData->lengths, numNonEmptyCollections, &memoryPool_);
  auto lengths = nestedData->lengths->asMutable<uint32_t>();
  for (int i = 0; i < numNonEmptyCollections; i++) {
    lengths[i] = offsets[i + 1] - offsets[i];
  }

  nestedData->offsets->setSize((numNonEmptyCollections + 1) * 4);
  nestedData->lengths->setSize(numNonEmptyCollections * 4);
  nestedData->nulls->setSize(bits::nbytes(numNonEmptyCollections));
  nestedData->numNonEmptyCollections = numNonEmptyCollections;
  nestedData->numNonNullCollections = numNonNullCollections;

  return nestedData;
}

void ParquetNestedLeafColumnReader::readNoFilter(
    uint64_t /* offset */,
    RowSet topLevelRows) {
  auto maxNumTopLevelRows = topLevelRows.back() + 1;
  if (maxNumTopLevelRows == topLevelRows.size()) {
    // full row range

    ensureValuesCapacity<uint8_t>(
        static_cast<vector_size_t>(numNonEmptyRowsInBatch_ * valueSize_));

    int32_t numNonEmptyRowsRead = 0;
    int i = 0;
    for (auto iter = dataPages_.begin(); iter < dataPages_.end(); iter++) {
      auto dataPage = *iter;

      auto bytesReadInStream = decodePage(dataPage, numNonEmptyRowsRead);

      numNonEmptyRowsRead += dataPage->numNonEmptyRowsInBatch;

      dataPage->pageData += bytesReadInStream;
      dataPage->encodedDataSize -= bytesReadInStream;
      dataPage->numRowsInPage -= dataPage->numRowsInBatch;
      dataPage->numRowsInBatch = 0;
      dataPage->numNonEmptyRowsInBatch = 0;
      dataPage->numNonNullRowsInBatch = 0;
      if (dataPage->isEmpty()) {
        // This data page is used up. No need to keep it anymore.
        VELOX_CHECK(iter == dataPages_.begin());
        dataPages_.pop_front();
      }
    }

    VELOX_CHECK_EQ(numNonEmptyRowsRead, numNonEmptyRowsInBatch_);

    // TODO: There is no need to allocate the RowSet here, but getFlatValues
    // requires a RowSet
    numValues_ = numNonEmptyRowsRead;
    outputRows_.resize(numNonEmptyRowsRead);
    std::iota(outputRows_.begin(), outputRows_.end(), 0);
  } else {
    //   Populate leaf level RowSet
    VELOX_NYI();
  }
}

} // namespace facebook::velox::parquet