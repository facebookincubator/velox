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

#include "velox/dwio/common/Arena.h"
#include "velox/dwio/common/OutputStream.h"
#include "velox/dwio/dwrf/common/wrap/dwrf-proto-wrapper.h"
#include "velox/dwio/dwrf/writer/StatisticsBuilder.h"

namespace facebook::velox::dwrf {

using dwio::common::ArenaCreate;
using dwio::common::BufferedOutputStream;
using dwio::common::PositionRecorder;

namespace {

constexpr int32_t PRESENT_STREAM_INDEX_ENTRIES_UNPAGED = 3;
constexpr int32_t PRESENT_STREAM_INDEX_ENTRIES_PAGED =
    PRESENT_STREAM_INDEX_ENTRIES_UNPAGED + 1;

} // namespace

class IndexBuilder : public PositionRecorder {
 public:
  IndexBuilder(
      std::unique_ptr<BufferedOutputStream> out,
      dwio::common::FileFormat fileFormat = dwio::common::FileFormat::DWRF)
      : out_{std::move(out)},
        arena_(std::make_unique<google::protobuf::Arena>()) {
    auto rowIndex = ArenaCreate<proto::RowIndex>(arena_.get());
    auto rowIndexEntry = ArenaCreate<proto::RowIndexEntry>(arena_.get());

    index_ = std::make_unique<RowIndexWriteWrapper>(rowIndex);
    entry_ = std::make_unique<RowIndexEntryWriteWrapper>(rowIndexEntry);
  }

  virtual ~IndexBuilder() = default;

  void add(uint64_t pos, int32_t index = -1) override {
    getEntry(index).addPositions(pos);
  }

  virtual void addEntry(const StatisticsBuilder& writer) {
    auto stats = entry_->mutableStatistics();
    writer.toProto(stats);
    index_->addEntry(entry_);
    entry_->clear();
  }

  virtual size_t getEntrySize() const {
    const int32_t size = index_->entrySize() + 1;
    VELOX_CHECK_GT(size, 0, "Invalid entry size or missing current entry.");
    return size;
  }

  virtual void flush() {
    // remove isPresent positions if none is null
    index_->SerializeToZeroCopyStream(out_.get());
    out_->flush();
    index_->clear();
    entry_->clear();
  }

  void capturePresentStreamOffset() {
    if (!presentStreamOffset_.has_value()) {
      presentStreamOffset_ = entry_->positionsSize();
    } else {
      DWIO_ENSURE_EQ(presentStreamOffset_.value(), entry_->positionsSize());
    }
  }

  void removePresentStreamPositions(bool isPaged) {
    DWIO_ENSURE(presentStreamOffset_.has_value());
    const auto streamCount = isPaged ? PRESENT_STREAM_INDEX_ENTRIES_PAGED
                                     : PRESENT_STREAM_INDEX_ENTRIES_UNPAGED;

    // Only need to process entries that have been added to the row index
    for (uint32_t i = 0; i < index_->entrySize(); ++i) {
      index_->mutableEntry(i).mutablePositions(
          presentStreamOffset_.value(), streamCount);
    }
  }

 private:
  RowIndexEntryWriteWrapper getEntry(int32_t index) {
    if (index < 0) {
      return *entry_;
    } else if (index < index_->entrySize()) {
      return index_->mutableEntry(index);
    } else {
      VELOX_CHECK_EQ(index, index_->entrySize());
      return *entry_;
    }
  }

  const std::unique_ptr<BufferedOutputStream> out_;
  std::unique_ptr<RowIndexWriteWrapper> index_;
  std::unique_ptr<RowIndexEntryWriteWrapper> entry_;
  std::unique_ptr<google::protobuf::Arena> arena_;
  std::optional<int32_t> presentStreamOffset_;

  friend class IndexBuilderTest;
};

} // namespace facebook::velox::dwrf
