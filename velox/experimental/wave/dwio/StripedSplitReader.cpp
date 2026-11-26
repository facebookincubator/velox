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

#include "velox/experimental/wave/dwio/StripedSplitReader.h"
#include "velox/experimental/wave/dwio/StructColumnReader.h"

// NOLINTNEXTLINE(facebook-hte-PreventUseOfDeclareGflag)
DECLARE_int32(wave_max_reader_batch_rows);

namespace facebook::velox::wave {

bool StripedSplitReader::nextStripe() {
  if (stripeIdx_ + 1 >= numStripes()) {
    return false;
  }
  if (columnReader_) {
    completedRows_ += stripeRows_;
    previousStripes_.push_back(
        {stripeIdx_, std::move(columnReader_), std::move(stripeUse_)});
  }
  ++stripeIdx_;
  nextRow_ = 0;
  WithSubfieldMap subfields(subfields_);
  columnReader_ = startStripe(stripeIdx_, stripeRows_);
  stripeUse_ = std::make_shared<bool>(true);
  return true;
}

void StripedSplitReader::freeUnusedStripes() {
  // use_count() is exact enough here: only 'this' copies a use, on the thread
  // that calls schedule(), so a use with no other owner never gets one again.
  // A ReadStream that drops its copy on another thread at the same time at
  // worst delays the free to a later call.
  for (auto i = 0; i < previousStripes_.size();) {
    if (previousStripes_[i].use.use_count() > 1) {
      ++i;
      continue;
    }
    const auto index = previousStripes_[i].index;
    previousStripes_.erase(previousStripes_.begin() + i);
    releaseStripe(index);
  }
}

int32_t StripedSplitReader::canAdvance(WaveStream& /*stream*/) {
  while (!columnReader_ || nextRow_ >= stripeRows_) {
    if (!nextStripe()) {
      return 0;
    }
  }
  return std::min<int32_t>(
      FLAGS_wave_max_reader_batch_rows, stripeRows_ - nextRow_);
}

void StripedSplitReader::schedule(WaveStream& waveStream, int32_t maxRows) {
  const auto numRows = std::min<int32_t>(maxRows, stripeRows_ - nextRow_);
  auto rowSet = folly::Range<const int32_t*>(iota(numRows, rows_), numRows);
  std::unique_ptr<ReadStream> exe(
      reinterpret_cast<ReadStream*>(
          waveStream.recycleExecutable(nullptr, 0).release()));
  if (exe &&
      reinterpret_cast<ColumnReader*>(exe->reader()) != columnReader_.get()) {
    // The previous read on the WaveStream is for a different stripe or split.
    exe.reset();
  }
  if (!exe) {
    exe = std::make_unique<ReadStream>(
        reinterpret_cast<StructColumnReader*>(columnReader_.get()),
        waveStream,
        params_.ioStatistics.get(),
        fileInfo_);
    exe->setStripeUse(stripeUse_);
  }
  freeUnusedStripes();
  ReadStream::launch(std::move(exe), nextRow_, rowSet);
  nextRow_ += numRows;
}

bool StripedSplitReader::isFinished() const {
  return stripeIdx_ + 1 >= numStripes() &&
      (!columnReader_ || nextRow_ >= stripeRows_);
}

} // namespace facebook::velox::wave
