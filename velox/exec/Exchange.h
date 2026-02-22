/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <random>

#include "velox/exec/ExchangeClient.h"
#include "velox/exec/Operator.h"
#include "velox/exec/OperatorType.h"
#include "velox/exec/OutputBufferManager.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/RowSerializer.h"

namespace facebook::velox::exec {

struct RemoteConnectorSplit : public connector::ConnectorSplit {
  const std::string taskId;

  explicit RemoteConnectorSplit(const std::string& remoteTaskId)
      : ConnectorSplit(""), taskId(remoteTaskId) {}

  static std::shared_ptr<RemoteConnectorSplit> create(
      const folly::dynamic& obj);

  static void registerSerDe();

  folly::dynamic serialize() const override;

  std::string toString() const override {
    return fmt::format("Remote: {}", taskId);
  }
};

class Exchange : public SourceOperator {
 public:
  Exchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::ExchangeNode>& exchangeNode,
      std::shared_ptr<ExchangeClient> exchangeClient,
      std::string_view operatorType = OperatorType::kExchange);

  ~Exchange() override {
    close();
  }

  RowVectorPtr getOutput() override;

  void close() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

 protected:
  virtual VectorSerde* getSerde();

  // When 'estimatedRowSize_' is unset, meaning we haven't materialized
  // and returned any output from this exchange operator, we return this
  // conservative number of output rows, to make sure memory does not grow too
  // much.
  static constexpr uint64_t kInitialOutputRows = 64;

  // Invoked to create exchange client for remote tasks. The function shuffles
  // the source task ids first to randomize the source tasks we fetch data from.
  // This helps to avoid different tasks fetching from the same source task in a
  // distributed system.
  void addRemoteTaskIds(std::vector<std::string>& remoteTaskIds);

  // Fetches splits from the task until there are no more splits or task returns
  // a future that will be complete when more splits arrive. Adds splits to
  // exchangeClient_.
  void getSplits(ContinueFuture* future);

  // Fetches runtime stats from ExchangeClient and replaces these in this
  // operator's stats.
  void recordExchangeClientStats();

  void recordInputStats(uint64_t rawInputBytes);

  RowVectorPtr getOutputFromColumnarPages(VectorSerde* serde);

  RowVectorPtr getOutputFromRowPages(VectorSerde* serde);

  const uint64_t preferredOutputBatchBytes_;

  const VectorSerde::Kind serdeKind_;

  const std::unique_ptr<VectorSerde::Options> serdeOptions_;

  /// True if this operator is responsible for fetching splits from the Task
  /// and passing these to ExchangeClient.
  const bool processSplits_;

  const int driverId_;

  bool noMoreSplits_ = false;

  std::shared_ptr<ExchangeClient> exchangeClient_;

  // A future received from Task::getSplitOrFuture(). It will be complete when
  // there are more splits available or no-more-splits signal has arrived.
  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  // Reusable result vector.
  RowVectorPtr result_;

  std::vector<std::unique_ptr<SerializedPageBase>> currentPages_;
  bool atEnd_{false};
  std::default_random_engine rng_{std::random_device{}()};

  // Memory holders for deserialization across 'getOutput' calls.
  // The merged pages for row serialization.
  std::unique_ptr<SerializedPageBase> mergedRowPage_;
  std::unique_ptr<RowIterator> rowIterator_;

  // State for columnar page deserialization.
  // Index of the current page in 'currentPages_' being processed.
  size_t columnarPageIdx_{0};

  // Stream for deserialization used by both row and columnar.
  std::unique_ptr<ByteInputStream> inputStream_;

  // The estimated bytes per row of the output of this exchange operator
  // computed from the last processed output.
  std::optional<uint64_t> estimatedRowSize_;
};

} // namespace facebook::velox::exec
