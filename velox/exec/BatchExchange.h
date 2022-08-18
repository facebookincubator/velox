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

#include <velox/common/memory/MappedMemory.h>
#include <velox/common/memory/Memory.h>
#include <velox/exec/BatchShuffleFactory.h>
#include <velox/exec/Exchange.h>
#include <velox/vector/VectorBatch.h>

#include <memory>

namespace facebook::velox::exec {
// In order to be able to use all the queuing infra structure in exchange
// for the batching mode in which Presto Pages are not used, we use inheritance
// so that the data read from batch shuffle can be used in the exchange queue
// Another option is to create a common base class and change all the use cases
// of SerializedPage throughout the code.
class BlockWrapper : public SerializedPage {
 public:
  BlockWrapper(const std::unique_ptr<BatchBlock>& block)
      : SerializedPage(nullptr), block_(block) {}

  ~BlockWrapper() = default;

  // Returns the size of the serialized data in bytes.
  uint64_t size() const override {
    return block_->size();
  }

  void prepareStreamForDeserialize(ByteStream* input) override {}

  const std::unique_ptr<BatchBlock>& block() {
    return block_;
  };

 private:
  const std::unique_ptr<BatchBlock>& block_;
};

class BatchExchangeSource : public ExchangeSource {
  using shuffleMetaDataType = long;

 public:
  BatchExchangeSource(
      const std::string& taskId,
      int destination,
      std::shared_ptr<ExchangeQueue> queue,
      std::shared_ptr<velox::exec::BatchShuffleFactory>& shuffleFactory,
      std::shared_ptr<velox::memory::MemoryPool>& pool,
      int partition,
      std::vector<shuffleMetaDataType>& metaData)
      : ExchangeSource(taskId, destination, queue),
        partition_(partition),
        metaData_(metaData),
        currentIterator_(nullptr) {
    VELOX_CHECK(shuffleFactory);
    VELOX_CHECK(pool);
    shuffleReader_ = shuffleFactory->createShuffleReader(partition, pool);
  }

  bool shouldRequestLocked() override {
    if (atEnd_) {
      return false;
    }
    bool pending = requestPending_;
    requestPending_ = true;
    return !pending;
  }

  void request() override;

  void close() override {}

 private:
  // For now assume there's one partition per ExchangeSource
  const int partition_;
  // for now set of integers should be enough. We may need to templatize it in
  // the future
  const std::vector<shuffleMetaDataType> metaData_;
  std::unique_ptr<velox::exec::BatchShuffleReader> shuffleReader_;
  std::unique_ptr<velox::exec::BatchShuffleBlockIterator> currentIterator_;
};

class BatchExchange : public Exchange {
 public:
  BatchExchange(
      int32_t operatorId,
      DriverCtx* ctx,
      const std::shared_ptr<const core::ExchangeNode>& exchangeNode,
      std::shared_ptr<ExchangeClient> exchangeClient,
      std::unique_ptr<velox::batch::VectorSerdeFactory> serdeFactory)
      : Exchange(operatorId, ctx, exchangeNode, exchangeClient),
        vectorSerdeFactory_(std::move(serdeFactory)) {}

  RowVectorPtr getOutput() override;

 protected:
  std::unique_ptr<velox::batch::VectorSerde> vectorSerde_ = nullptr;
  std::unique_ptr<velox::batch::VectorSerdeFactory> vectorSerdeFactory_;
};

} // namespace facebook::velox::exec