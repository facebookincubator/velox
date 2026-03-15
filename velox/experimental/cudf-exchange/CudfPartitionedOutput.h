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

#include "velox/exec/Operator.h"
#include "velox/experimental/cudf-exchange/CudfOutputQueueManager.h"
#include "velox/experimental/cudf/exec/NvtxHelper.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::cudf_exchange {

/// This is the cudf equivalent of the PartitionedOutput operator for cudf.
/// Instead of serializing and segmenting the partitioned data into an
/// OutputBuffer, the CudfPartitionedOutput operator transfers entire
/// cudf::packed_columns corresponding to CudfVectors to other workers.
class CudfPartitionedOutput : public exec::Operator,
                              public cudf_velox::NvtxHelper {
 public:
  CudfPartitionedOutput(
      int32_t operatorId,
      exec::DriverCtx* ctx,
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
      bool eagerFlush);

  void addInput(RowVectorPtr input) override;

  /// Always returns nullptr. The action is to further process
  /// unprocessed input. If all input has been processed, 'this' is in
  /// a non-blocked state, otherwise blocked.
  RowVectorPtr getOutput() override;

  /// always true but the caller will check isBlocked before adding input, hence
  /// the blocked state does not accumulate input.
  bool needsInput() const override {
    return true;
  }

  // the operator is blocked if the queues are full, we are ignoring this so
  // always return kNotBlocked
  exec::BlockingReason isBlocked(ContinueFuture* future) override;

  // The operaor is finished when the queue manager say the queues have all been
  // drained ?
  bool isFinished() override;

 private:
  std::shared_ptr<facebook::velox::cudf_exchange::CudfOutputQueueManager>
  sharedQueueManager();

  // Heuristic method to derive the partition keys from the PartitionNode
  // specification.
  void initPartitionKeys(
      const std::shared_ptr<const core::PartitionedOutputNode>& planNode);

  // Partitions the cudf table view using the partition keys and a hash
  // function using the given stream.
  void hashPartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Splits the cudf table view into equal sizes. This is used when
  // RoundRobin partitioning is requested but round robin on a
  // row-by-row basis is not meaningful for cudf exchange.
  void equalPartition(cudf::table_view tableView, rmm::cuda_stream_view stream);

  // Splits the table along the given offsets and enqueues each offset
  // to the corresponding partition, i.e. first split to the partition 0,
  // second split to partition 1 etc.
  void splitAndEnqueue(
      cudf::table_view tableView,
      std::vector<cudf::size_type> offsets,
      rmm::cuda_stream_view stream);

  const std::weak_ptr<CudfOutputQueueManager> queueManager_;
  std::vector<column_index_t> partitionKeyIndices_;
  const size_t numPartitions_;

  const int pipelineId_;
  const int driverId_;

  exec::BlockingReason blockingReason_;
  ContinueFuture future_;

  bool finished_{false};
  std::string spec_;

  // Used for switching columns when column order differs between input and
  // output.
  std::vector<uint32_t> remap_;
};

} // namespace facebook::velox::cudf_exchange
