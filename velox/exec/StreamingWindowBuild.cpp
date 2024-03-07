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

#include "velox/exec/StreamingWindowBuild.h"
#include "velox/exec/MemoryReclaimer.h"

namespace facebook::velox::exec {

StreamingWindowBuild::StreamingWindowBuild(
    const std::shared_ptr<const core::WindowNode>& windowNode,
    velox::memory::MemoryPool* pool,
    const common::SpillConfig* spillConfig,
    tsan_atomic<bool>* nonReclaimableSection)
    : WindowBuild(windowNode, pool, spillConfig, nonReclaimableSection),
      pool_(pool) {
  allKeyInfo_.reserve(partitionKeyInfo_.size() + sortKeyInfo_.size());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), partitionKeyInfo_.begin(), partitionKeyInfo_.end());
  allKeyInfo_.insert(
      allKeyInfo_.cend(), sortKeyInfo_.begin(), sortKeyInfo_.end());
}

void StreamingWindowBuild::ensureInputFits(const RowVectorPtr& input) {
  if (spillConfig_ == nullptr) {
    // Spilling is disabled.
    return;
  }

  if (data_->numRows() == 0) {
    // Nothing to spill.
    return;
  }

  // Test-only spill path.
  if (inputRows_.size() > 0 && testingTriggerSpill()) {
    spill();
    return;
  }

  auto [freeRows, outOfLineFreeBytes] = data_->freeSpace();
  const auto outOfLineBytes =
      data_->stringAllocator().retainedSize() - outOfLineFreeBytes;
  const auto outOfLineBytesPerRow = outOfLineBytes / data_->numRows();

  const auto currentUsage = data_->pool()->currentBytes();
  const auto minReservationBytes =
      currentUsage * spillConfig_->minSpillableReservationPct / 100;
  const auto availableReservationBytes = data_->pool()->availableReservation();
  const auto incrementBytes =
      data_->sizeIncrement(input->size(), outOfLineBytesPerRow * input->size());

  // First to check if we have sufficient minimal memory reservation.
  if (availableReservationBytes >= minReservationBytes) {
    if ((freeRows > input->size()) &&
        (outOfLineBytes == 0 ||
         outOfLineFreeBytes >= outOfLineBytesPerRow * input->size())) {
      // Enough free rows for input rows and enough variable length free space.
      return;
    }
  }

  // Check if we can increase reservation. The increment is the largest of twice
  // the maximum increment from this input and 'spillableReservationGrowthPct_'
  // of the current memory usage.
  const auto targetIncrementBytes = std::max<int64_t>(
      incrementBytes * 2,
      currentUsage * spillConfig_->spillableReservationGrowthPct / 100);
  {
    memory::ReclaimableSectionGuard guard(nonReclaimableSection_);
    if (data_->pool()->maybeReserve(targetIncrementBytes)) {
      return;
    }
  }

  LOG(WARNING) << "Failed to reserve " << succinctBytes(targetIncrementBytes)
               << " for memory pool " << data_->pool()->name()
               << ", usage: " << succinctBytes(data_->pool()->currentBytes())
               << ", reservation: "
               << succinctBytes(data_->pool()->reservedBytes());
}

void StreamingWindowBuild::setupSpiller() {
  spillers_.push_back(std::make_unique<Spiller>(
      Spiller::Type::kOrderByOutput, data_.get(), inputType_, spillConfig_));
}

void StreamingWindowBuild::spill() {
  if (spillers_.size() < currentSpilledPartition_ + 1) {
    setupSpiller();
  }

  spillers_[currentSpilledPartition_]->spill(inputRows_, lastRun);
  inputRows_.clear();
}

void StreamingWindowBuild::buildNextPartition() {
  if (currentSpilledPartition_ < spillers_.size() &&
      spillers_[currentSpilledPartition_] != nullptr) {
    partitionStartRows_.push_back(sortedRows_.size());
    lastRun = true;

    // Spill remaining data to avoid running out of memory while
    // sort-merging
    // spilled data.
    spill();

    auto spillPartition = spillers_[currentSpilledPartition_]->finishSpill();
    merges_.push_back(spillPartition.createOrderedReader(pool_));
    return;
  }

  merges_.push_back(nullptr);
  spillers_.push_back(nullptr);
  partitionStartRows_.push_back(sortedRows_.size());
  sortedRows_.insert(sortedRows_.end(), inputRows_.begin(), inputRows_.end());
  inputRows_.clear();
}

void StreamingWindowBuild::addInput(RowVectorPtr input) {
  for (auto i = 0; i < inputChannels_.size(); ++i) {
    decodedInputVectors_[i].decode(*input->childAt(inputChannels_[i]));
  }

  ensureInputFits(input);

  for (auto row = 0; row < input->size(); ++row) {
    char* newRow = data_->newRow();

    for (auto col = 0; col < input->childrenSize(); ++col) {
      data_->store(decodedInputVectors_[col], row, newRow, col);
    }

    if (previousRow_ != nullptr &&
        compareRowsWithKeys(previousRow_, newRow, partitionKeyInfo_)) {
      buildNextPartition();
      currentSpilledPartition_++;
    }

    inputRows_.push_back(newRow);

    previousRow_ = newRow;
  }
}

void StreamingWindowBuild::noMoreInput() {
  buildNextPartition();

  // Help for last partition related calculations.
  partitionStartRows_.push_back(sortedRows_.size());
}

std::unique_ptr<WindowPartition> StreamingWindowBuild::nextPartition() {
  currentPartition_++;

  if (currentPartition_ < merges_.size() &&
      merges_[currentPartition_] != nullptr) {
    loadNextPartitionFromSpill();
    VELOX_CHECK(!spilledSortedRows_.empty(), "No window partitions available")

    // Currently, when the data of the same window partition is spilled multiple
    // times, the order of the data loaded here is incorrect. For example, the
    // data spilled for the first time is 1, 2, 3. The second time it is spilled
    // is 4, 5. The data obtained here is 4, 5, 1, 2, 3. Therefore, a sort is
    // added here to sort the data.
    std::sort(
        spilledSortedRows_.begin(),
        spilledSortedRows_.end(),
        [this](const char* leftRow, const char* rightRow) {
          return compareRowsWithKeys(leftRow, rightRow, allKeyInfo_);
        });

    auto partition =
        folly::Range(spilledSortedRows_.data(), spilledSortedRows_.size());
    return std::make_unique<WindowPartition>(
        data_.get(), partition, inputColumns_, sortKeyInfo_);
  }

  VELOX_CHECK_GT(
      partitionStartRows_.size(), 0, "No window partitions available")

  VELOX_CHECK_LE(
      currentPartition_,
      partitionStartRows_.size() - 2,
      "All window partitions consumed");

  // Erase previous partition.
  if (currentPartition_ > 0) {
    auto numPreviousPartitionRows = partitionStartRows_[currentPartition_];
    data_->eraseRows(
        folly::Range<char**>(sortedRows_.data(), numPreviousPartitionRows));
    sortedRows_.erase(
        sortedRows_.begin(), sortedRows_.begin() + numPreviousPartitionRows);
    for (int i = currentPartition_; i < partitionStartRows_.size(); i++) {
      partitionStartRows_[i] =
          partitionStartRows_[i] - numPreviousPartitionRows;
    }
  }

  auto partitionSize = partitionStartRows_[currentPartition_ + 1] -
      partitionStartRows_[currentPartition_];
  auto partition = folly::Range(
      sortedRows_.data() + partitionStartRows_[currentPartition_],
      partitionSize);

  return std::make_unique<WindowPartition>(
      data_.get(), partition, inputColumns_, sortKeyInfo_);
}

void StreamingWindowBuild::loadNextPartitionFromSpill() {
  spilledSortedRows_.clear();
  data_->clear();

  while (auto next = merges_[currentPartition_]->next()) {
    if (next == nullptr) {
      break;
    }

    auto* newRow = data_->newRow();
    for (auto i = 0; i < inputChannels_.size(); ++i) {
      data_->store(next->decoded(i), next->currentIndex(), newRow, i);
    }
    spilledSortedRows_.push_back(newRow);
    next->pop();
  }
}

bool StreamingWindowBuild::hasNextPartition() {
  return partitionStartRows_.size() > 0 &&
      currentPartition_ < int(partitionStartRows_.size() - 2);
}

} // namespace facebook::velox::exec
