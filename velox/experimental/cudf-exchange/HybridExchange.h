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

#include <folly/Uri.h>
#include <random>
#include "velox/exec/ExchangeClient.h"
#include "velox/exec/Operator.h"
#include "velox/exec/Task.h"
#include "velox/experimental/cudf-exchange/CudfExchangeClient.h"
#include "velox/experimental/cudf-exchange/CudfExchangeSource.h"
#include "velox/experimental/cudf-exchange/CudfQueues.h"
#include "velox/experimental/cudf-exchange/ExchangeClientFacade.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/serializers/PrestoSerializer.h"
#include "velox/serializers/RowSerializer.h"

using namespace facebook::velox::exec;

namespace facebook::velox::cudf_exchange {

/// @brief The combined Cudf and HTTP exchange operator
/// is required to process "metadata" queries such as
/// "show schemas in catalog;" where the query plan contains
/// exchanges from other workers that are using CudfExchange
/// and also an exchange from the coordinator that is based
/// on HTTP. The HybridExchange assumes that it will always
/// only get remote splits to other worker nodes or only ever
/// to the coordinator but never a combination of the two.
/// The decision is taken when the first remote split is added.
/// Then, the nature of the operator is fixed.
///
/// The implementation contains both a CudfExchangeClient and a
/// ExchangeClient reference, only one of which is ever used.
/// When the first remote split points to the coordinator, then the
/// HTTP based ExchangeClient is used, otherwise the CudfExchangeClient.
/// The return type of the getOutput method is always a CudfVector.
/// If data origins from the HTTP ExchangeClient, then this data is
/// converted into a CudfVector using the CudfFromVelox operator.
class HybridExchange : public SourceOperator {
 public:
  HybridExchange(
      int32_t operatorId,
      DriverCtx* driverCtx,
      const std::shared_ptr<const core::PlanNode>& planNode,
      std::shared_ptr<ExchangeClientFacade> exchangeClientFacade,
      const std::string& operatorType = "HybridExchange");

  ~HybridExchange() override;

  BlockingReason isBlocked(ContinueFuture* future) override;

  bool isFinished() override;

  RowVectorPtr getOutput() override;

  void close() override;

 private:
  // When 'estimatedCompactRowSize_' is unset, meaning we haven't materialized
  // and returned any output from this exchange operator, we return this
  // conservative number of output rows, to make sure memory does not grow too
  // much.
  static constexpr uint64_t kInitialOutputCompactRows = 64;

  // Returns true if the result is empty.
  bool resultIsEmpty();

  // Empty the current result.
  void emptyResult();

  // Returns the pages vector from the result variant. This fails if the
  // result does not contain a vector of pages.
  const SerPageVector* getResultPages();

  // Returns the packed table with stream pointer from the result variant. This
  // fails if the result does not contain a packed table with stream pointer.
  const PackedTableWithStreamPtr* getResultPackedTable();

  // Invoked to create exchange client for remote tasks. The function shuffles
  // the source task ids first to randomize the source tasks we fetch data from.
  // This helps to avoid different tasks fetching from the same source task in a
  // distributed system.
  void addRemoteTaskIds(std::vector<std::string>& remoteTaskIds);

  // This is a no-op except when called from driver 0.
  // Fetches splits from the task until there are no more splits or task returns
  // a future that will be complete when more splits arrive. Adds splits to
  // exchangeClient_. Sets "noMoreSplits_" if the task returns not blocked
  // without a split, this is the end-of-splits signal. Decides on the first
  // split whether the exchangeClient_ is backed by the HTTP exchange client or
  // the Cudf exchange client.
  void getSplits(ContinueFuture* future);

  // Converts the results from the HTTP exchange client into a cudf vector.
  RowVectorPtr getOutputFromPages(const SerPageVector* pages);

  // Helper method used by getOutputFromPages
  RowVectorPtr getOutputFromCompactRows(
      VectorSerde* serde,
      const SerPageVector* pages);

  // Helper method used by getOutputFromPages
  RowVectorPtr getOutputFromUnsafeRows(
      VectorSerde* serde,
      const SerPageVector* pages);

  // Converts the results from the Cudf exchange client into a cudf vector.
  RowVectorPtr getOutputFromPackedTable(
      const PackedTableWithStreamPtr* dataPtr);

  // Fetches runtime stats from ExchangeClient and replaces these in this
  // operator's stats.
  void recordExchangeClientStats();

  void recordInputStats(uint64_t rawInputBytes, RowVectorPtr result = nullptr);

  // Abstracts away the http and the cudf exchange client.
  std::shared_ptr<ExchangeClientFacade> exchangeClient_;

  const uint64_t preferredOutputBatchBytes_;
  VectorSerde::Kind serdeKind_;
  std::unique_ptr<VectorSerde::Options> serdeOptions_;

  /// True if this operator is responsible for fetching splits from the Task
  /// and passing these to ExchangeClient. When running with multile drivers,
  /// this is done by the exchange running on driver 0.
  const bool processSplits_;
  const int pipelineId_;
  const int driverId_;
  bool noMoreSplits_ = false;

  // A future received from Task::getSplitOrFuture(). It will be complete when
  // there are more splits available or no-more-splits signal has arrived.
  ContinueFuture splitFuture_{ContinueFuture::makeEmpty()};

  // data returned from exchangeClient->next();
  ResultVariant currentData_;

  // Reusable result vector.
  RowVectorPtr result_;

  bool atEnd_{false};
  std::default_random_engine rng_{std::random_device{}()};

  // Memory holders needed by compact row serde to perform cursor like reads
  // across 'getOutputFromCompactRows' calls.
  std::unique_ptr<SerializedPage> compactRowPages_;
  std::unique_ptr<ByteInputStream> compactRowInputStream_;
  std::unique_ptr<RowIterator> compactRowIterator_;

  // The estimated bytes per row of the output of this exchange operator
  // computed from the last processed output.
  std::optional<uint64_t> estimatedCompactRowSize_;
};

} // namespace facebook::velox::cudf_exchange
