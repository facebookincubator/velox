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

#include <velox/exec/BatchExchange.h>

namespace facebook::velox::exec {

void BatchExchangeSource::request() {
  VELOX_CHECK(requestPending_);
  if (!currentIterator_) {
    // When it is the first time, we grab the block iterator for the given
    // shuffle partition and metadata
    currentIterator_ =
        shuffleReader_->getPartitionBlocks(partition_, metaData_);
  }

  // Assume if the partition (metadata) has no row, the returned iterator is
  // null
  if (currentIterator_ != nullptr) {
    // For the time being the Shuffle API for reading blocks is blocking
    // @TODO we need to update this part ff that API becomes async (future
    // based) we need to update this logic as well
    std::lock_guard<std::mutex> l(queue_->mutex());
    do {
      queue_->enqueue(
          std::move(std::make_unique<BlockWrapper>(currentIterator_->block())));
      currentIterator_->next();
    } while (currentIterator_->hasNext());
  }
}

RowVectorPtr BatchExchange::getOutput() {
  if (!currentPage_) {
    return nullptr;
  }

  auto wrappedBlock = (BlockWrapper*)(currentPage_.get());
  VELOX_CHECK(wrappedBlock != nullptr);
  auto rowIterator = wrappedBlock->block()->iterator();
  // Construct the row block pointers from the wrapped block
  std::vector<std::optional<std::string_view>> rowPointers;
  if (rowIterator != nullptr) {
    do {
      auto rowPointer = std::string_view();
      rowPointers.emplace_back(rowPointer);
    } while (rowIterator->hasNext());
  }

  // Creating the row vector from the row pointers
  if (!vectorSerde_) {
    vectorSerde_ = vectorSerdeFactory_->createVectorSerde();
  }
  vectorSerde_->deserializeVector(rowPointers, outputType_, &result_);

  return result_;
}

} // namespace facebook::velox::exec