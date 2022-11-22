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

#include "velox/dwio/common/SelectiveStructColumnReader.h"
#include "velox/dwio/dwrf/reader/DwrfData.h"

namespace facebook::velox::dwrf {

class SelectiveStructColumnReaderBase
    : public dwio::common::SelectiveStructColumnReaderBase {
 public:
  SelectiveStructColumnReaderBase(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec)
      : dwio::common::SelectiveStructColumnReaderBase(
            requestedType,
            dataType,
            params,
            scanSpec),
        rowsPerRowGroup_(formatData_->rowsPerRowGroup().value()) {
    VELOX_CHECK_EQ(nodeType_->id, dataType->id, "working on the same node");
  }

  void seekTo(vector_size_t offset, bool readsNullsOnly) override;

  void seekToRowGroup(uint32_t index) override {
    SelectiveColumnReader::seekToRowGroup(index);
    if (isTopLevel_ && !formatData_->hasNulls()) {
      readOffset_ = index * rowsPerRowGroup_;
      return;
    }
    // There may be a nulls stream but no other streams for the struct.
    formatData_->seekToRowGroup(index);
    // Set the read offset recursively. Do this before seeking the
    // children because list/map children will reset the offsets for
    // their children.
    setReadOffsetRecursive(index * rowsPerRowGroup_);
    for (auto& child : children_) {
      child->seekToRowGroup(index);
    }
  }

  /// Advance field reader to the row group closest to specified offset by
  /// calling seekToRowGroup.
  void advanceFieldReader(SelectiveColumnReader* reader, vector_size_t offset)
      override {
    if (!reader->isTopLevel()) {
      return;
    }
    auto rowGroup = reader->readOffset() / rowsPerRowGroup_;
    auto nextRowGroup = offset / rowsPerRowGroup_;
    if (nextRowGroup > rowGroup) {
      reader->seekToRowGroup(nextRowGroup);
      reader->setReadOffset(nextRowGroup * rowsPerRowGroup_);
    }
  }

 private:
  const int32_t rowsPerRowGroup_;
};

struct SelectiveStructColumnReader : SelectiveStructColumnReaderBase {
  SelectiveStructColumnReader(
      const std::shared_ptr<const dwio::common::TypeWithId>& requestedType,
      const std::shared_ptr<const dwio::common::TypeWithId>& dataType,
      DwrfParams& params,
      common::ScanSpec& scanSpec);

 private:
  void addChild(std::unique_ptr<SelectiveColumnReader> child) {
    children_.push_back(child.get());
    childrenOwned_.push_back(std::move(child));
  }

  std::vector<std::unique_ptr<SelectiveColumnReader>> childrenOwned_;
};

} // namespace facebook::velox::dwrf
