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

#include <velox/exec/BatchShuffleFactory.h>
#include <velox/exec/PartitionedOutput.h>
#include <velox/vector/VectorBatch.h>

namespace facebook::velox::exec {

// This class implement a basic batch destination for the PartitionedOutput
// operator. Internally it uses shared serializer and shuffler writer objects
class BatchDestination : public Destination {
 public:
  BatchDestination(
      const std::string& taskId,
      int destination,
      memory::MappedMemory* FOLLY_NONNULL memory)
      : Destination(taskId, destination, memory) {}

  BlockingReason advance(
      uint64_t maxBytes,
      const std::vector<vector_size_t>& sizes,
      const RowVectorPtr& output,
      PartitionedOutputBufferManager& bufferManager,
      bool* FOLLY_NONNULL atEnd,
      ContinueFuture* FOLLY_NONNULL future) override;

  BlockingReason flush(
      PartitionedOutputBufferManager& bufferManager,
      ContinueFuture* FOLLY_NULLABLE future) override {
    // Given the flush is not really used in this setup
    return BlockingReason::kNotBlocked;
  };

  virtual ~BatchDestination() = default;

  static void createSerializer(
      std::unique_ptr<velox::batch::VectorSerdeFactory> vectorSerdeFactory) {
    serializer_s = vectorSerdeFactory->createVectorSerde();
  }

  static void createShuffleWriter(
      std::unique_ptr<velox::exec::BatchShuffleFactory> shuffleFactory,
      std::shared_ptr<memory::MemoryPool> pool) {
    shuffleWriter_s = shuffleFactory->createShuffleWriter(0, pool);
  }

 private:
  // A shared serializer for writing the rows to shuffle writer
  static std::unique_ptr<velox::batch::VectorSerde> serializer_s;
  // Since shuffle writer is stateless, we use one for all destination
  static std::unique_ptr<velox::exec::BatchShuffleWriter> shuffleWriter_s;
};

class BatchDestinationFactory : public DestinationFactory {
 public:
  bool needsBufferManager() override {
    return false;
  }
  std::unique_ptr<Destination> createDestination(
      const std::string& taskId,
      int destination,
      memory::MappedMemory* FOLLY_NONNULL memory) override {
    return std::make_unique<StreamingDestination>(taskId, destination, memory);
  }
  virtual ~BatchDestinationFactory() = default;
};
} // namespace facebook::velox::exec