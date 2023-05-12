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
#include "velox/dwio/common/SelectiveColumnReaderInternal.h"
#include "velox/dwio/common/SelectiveRepeatedColumnReader.h"
#include "velox/dwio/dwrf/common/DecoderUtil.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

class SelectiveListColumnReader
    : public dwio::common::SelectiveListColumnReader {
 public:
  SelectiveListColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec);

  void resetFilterCaches() override {
    child_->resetFilterCaches();
  }

  void seekToRowGroup(uint32_t index) override {
    SelectiveColumnReader::seekToRowGroup(index);
    auto positionsProvider = formatData_->seekToRowGroup(index);
    length_->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());

    child_->seekToRowGroup(index);
    child_->setReadOffsetRecursive(0);
    childTargetReadOffset_ = 0;
  }

  void readLengths(
      int32_t* FOLLY_NONNULL lengths,
      int32_t numLengths,
      const uint64_t* FOLLY_NULLABLE nulls) override {
    length_->next(lengths, numLengths, nulls);
  }

 private:
  std::unique_ptr<dwio::common::IntDecoder</*isSigned*/ false>> length_;
};

class SelectiveMapColumnReader : public dwio::common::SelectiveMapColumnReader {
 public:
  SelectiveMapColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec);

  void resetFilterCaches() override {
    keyReader_->resetFilterCaches();
    elementReader_->resetFilterCaches();
  }

  void seekToRowGroup(uint32_t index) override {
    SelectiveColumnReader::seekToRowGroup(index);
    auto positionsProvider = formatData_->seekToRowGroup(index);

    length_->seekToRowGroup(positionsProvider);

    VELOX_CHECK(!positionsProvider.hasNext());

    keyReader_->seekToRowGroup(index);
    keyReader_->setReadOffsetRecursive(0);
    elementReader_->seekToRowGroup(index);
    elementReader_->setReadOffsetRecursive(0);
    childTargetReadOffset_ = 0;
  }

  void readLengths(
      int32_t* FOLLY_NONNULL lengths,
      int32_t numLengths,
      const uint64_t* FOLLY_NULLABLE nulls) override {
    length_->next(lengths, numLengths, nulls);
  }

 private:
  std::unique_ptr<dwio::common::IntDecoder</*isSigned*/ false>> length_;
};

} // namespace facebook::velox::dwrf
