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

#include "velox/experimental/wave/dwio/ColumnReader.h"
#include "velox/experimental/wave/exec/WaveSplitReader.h"

namespace facebook::velox::wave {

/// A WaveSplitReader that reads the stripes (row groups) of a split one at a
/// time with a reader tree per stripe. A file format provides the stripes and
/// makes the reader tree of a stripe. The data and reader trees of previous
/// stripes stay alive while their batches may still be decoding. Reader
/// trees of later stripes are made on other threads than the one that adds
/// the split, with the subfield map of the thread that adds the split.
class StripedSplitReader : public WaveSplitReader {
 public:
  explicit StripedSplitReader(const SplitReaderParams& params)
      : params_(params), subfields_(threadSubfieldMap()) {}

  int32_t canAdvance(WaveStream& stream) override;

  void schedule(WaveStream& stream, int32_t maxRows) override;

  bool isFinished() const override;

  uint64_t getCompletedBytes() override {
    return params_.ioStatistics->rawBytesRead();
  }

  uint64_t getCompletedRows() override {
    return completedRows_ + nextRow_;
  }

  std::unordered_map<std::string, RuntimeCounter> runtimeStats() override {
    return {};
  }

 protected:
  /// Number of stripes of the split.
  virtual int32_t numStripes() const = 0;

  /// Loads stripe 'index' of the split and makes its reader tree. Sets
  /// 'numRows' to the number of rows of the stripe. The previous stripe's
  /// data must stay alive until releaseStripe() or the format's destructor.
  virtual std::unique_ptr<ColumnReader> startStripe(
      int32_t index,
      int32_t& numRows) = 0;

  /// Frees the data of stripe 'index'. Called when no batch of the stripe is
  /// decoding, after its reader tree is freed.
  virtual void releaseStripe(int32_t /*index*/) {}

  SplitReaderParams params_;
  FileInfo fileInfo_;

 private:
  const SubfieldMap* const subfields_;

  // Makes the reader tree of the next stripe. Returns false if there are no
  // more stripes.
  bool nextStripe();

  struct PreviousStripe {
    int32_t index;
    std::unique_ptr<ColumnReader> reader;
    std::shared_ptr<void> use;
  };

  // Frees the previous stripes that no ReadStream uses.
  void freeUnusedStripes();

  std::unique_ptr<ColumnReader> columnReader_;
  // Held by the ReadStreams of the current stripe.
  std::shared_ptr<void> stripeUse_;
  // Stripes whose batches may still be decoding.
  std::vector<PreviousStripe> previousStripes_;
  // Position of the current stripe, -1 before the first.
  int32_t stripeIdx_{-1};
  int32_t stripeRows_{0};
  // First unscheduled row of the current stripe.
  int32_t nextRow_{0};
  // Rows of the stripes before the current one.
  uint64_t completedRows_{0};
  raw_vector<int32_t> rows_;
};

} // namespace facebook::velox::wave
