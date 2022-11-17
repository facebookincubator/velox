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

#include "velox/dwio/parquet/reader/ListColumnReader.h"

#include "velox/dwio/parquet/reader/ParquetColumnReader.h"

namespace facebook::velox::parquet {

ListColumnReader::ListColumnReader(
    std::shared_ptr<const dwio::common::TypeWithId> requestedType,
    ParquetParams& params,
    common::ScanSpec& scanSpec,
    common::ScanSpec& topLevelScanSpec)
    : ParquetRepeatedColumnReader(
          requestedType,
          params,
          scanSpec,
          topLevelScanSpec) {
  auto& childType = requestedType->childAt(0);

  auto childReader = ParquetColumnReader::build(
      childType, params, *scanSpec.children()[0], scanSpec, true);

  ParquetNestedColumnReader* tmp =
      dynamic_cast<ParquetNestedColumnReader*>(childReader.get());
  if (tmp != nullptr) {
    childReader.release();
    childColumnReader_.reset(tmp);
  }
}

void ListColumnReader::enqueueRowGroup(
    uint32_t index,
    dwio::common::BufferedInput& input) {
  childColumnReader_->enqueueRowGroup(index, input);
}

void ListColumnReader::seekToRowGroup(uint32_t index) {
  ParquetRepeatedColumnReader::seekToRowGroup(index);
  childColumnReader_->seekToRowGroup(index);
}

void ListColumnReader::read(
    vector_size_t offset,
    RowSet rows,
    const uint64_t* /*incomingNulls*/) {
  read(offset, rows);
}

std::vector<std::shared_ptr<NestedData>> ListColumnReader::read(
    uint64_t offset,
    RowSet rows) {
  if (childColumnReader_ && !rows.empty()) {
    auto nestedData =
        childColumnReader_->read(childColumnReader_->readOffset(), rows);
    auto nestedDataForLevel = nestedData[level_];
    resultNulls_ = nestedDataForLevel->nulls;
    offsets_ = nestedDataForLevel->offsets;
    lengths_ = nestedDataForLevel->lengths;
    nestedData[level_] = nullptr;

    return nestedData;
  }

  return {};
}

void ListColumnReader::getValues(RowSet rows, VectorPtr* result) {
  VectorPtr elements;
  if (childColumnReader_) {
    childColumnReader_->getValues(rows, &elements);
  }
  *result = std::make_shared<ArrayVector>(
      &memoryPool_,
      nodeType_->type,
      resultNulls_,
      rows.size(),
      offsets_,
      lengths_,
      elements);
}

} // namespace facebook::velox::parquet
