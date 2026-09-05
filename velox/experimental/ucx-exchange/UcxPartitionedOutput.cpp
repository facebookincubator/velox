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
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include <fmt/format.h>
#include <limits>
#include "velox/core/PlanNode.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Driver.h"
#include "velox/exec/Operator.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/exec/Utilities.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/binaryop.hpp>
#include <cudf/concatenate.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/copying.hpp>
#include <cudf/detail/utilities/stream_pool.hpp>
#include <cudf/filling.hpp>
#include <cudf/partitioning.hpp>
#include <cudf/scalar/scalar.hpp>
#include <cudf/stream_compaction.hpp>
#include <cudf/unary.hpp>
#include <cudf/utilities/memory_resource.hpp>

using namespace facebook::velox::cudf_velox;
using facebook::velox::exec::Task;
namespace facebook::velox::ucx_exchange {

// Computes a mapping from names in n2 to names in n1
// and returns that mapping in remap.
// Names in n2 must occurs in n1.
static void getRemapping(
    const std::vector<std::string>& n1,
    const std::vector<std::string>& n2,
    std::vector<uint32_t>& remap) {
  std::unordered_map<std::string, uint32_t> lookup;
  for (uint32_t i = 0; i < n1.size(); ++i) {
    lookup[n1[i]] = i;
  }

  remap.clear();
  remap.reserve(n2.size());
  for (const auto& key : n2) {
    remap.push_back(lookup.at(key));
  }
}

UcxPartitionedOutput::UcxPartitionedOutput(
    int32_t operatorId,
    exec::DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode,
    const std::shared_ptr<UcxOutputQueueManager>& queueManager)
    : Operator(
          ctx,
          planNode->outputType(),
          operatorId,
          planNode->id(),
          "cudfPartitionedOutput"),
      NvtxHelper(
          nvtx3::rgb{255, 215, 0}, // Gold
          operatorId,
          fmt::format("[{}]", planNode->id())),
      queueManager_(queueManager),
      numPartitions_(planNode->numPartitions()),
      replicateNullsAndAny_(planNode->isReplicateNullsAndAny()),
      pipelineId_(ctx->pipelineId),
      driverId_(ctx->driverId),
      targetRowsPerChunk_(ctx->queryConfig().get<int64_t>(
          CudfConfig::kUcxPartitionedOutputBatchRows,
          CudfConfig::getInstance().partitionedOutputBatchRows)) {
  VELOX_CHECK_NOT_NULL(queueManager, "UCX output queue manager is null");
  this->initPartitionKeys(planNode);
  auto sources = planNode->sources();
  std::vector<std::string> inNames, outNames;
  inNames.reserve(planNode->inputType()->size());
  for (int i = 0; i < planNode->inputType()->size(); ++i) {
    inNames.push_back(planNode->inputType()->nameOf(i));
  }
  outNames.reserve(planNode->outputType()->size());
  for (int i = 0; i < planNode->outputType()->size(); ++i) {
    outNames.push_back(planNode->outputType()->nameOf(i));
  }
  if (inNames != outNames) {
    getRemapping(inNames, outNames, remap_);
  }
}

void UcxPartitionedOutput::addInput(RowVectorPtr input) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " addInput";
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  auto cudfVector = std::dynamic_pointer_cast<CudfVector>(input);
  VELOX_CHECK_NOT_NULL(cudfVector, "Input must be a CudfVector");
  VELOX_CHECK(
      !future_.valid() || future_.hasValue(),
      "addInput with outstanding future!");

  // Record stats per-input (before buffering).
  {
    auto lockedStats = stats_.wlock();
    lockedStats->addOutputVector(input->estimateFlatSize(), input->size());
  }

  // CudfVector::size(), not the table view: a table with no columns derives
  // num_rows() from them and so reports 0 however many rows it holds.
  pendingRows_ += cudfVector->size();
  pendingInputs_.push_back(std::move(cudfVector));

  if (targetRowsPerChunk_ <= 0 || pendingRows_ >= targetRowsPerChunk_) {
    flushPending();
  }
}

void UcxPartitionedOutput::flushPending() {
  if (pendingInputs_.empty()) {
    return;
  }

  try {
    cudf::table_view tableView;
    rmm::cuda_stream_view stream = pendingInputs_.back()->stream();
    // Keeps the merged table alive while tableView references it.
    std::unique_ptr<cudf::table> mergedTable;

    // Logical rows being flushed. Summed from CudfVector::size() before
    // anything clears pendingInputs_, and before any cuDF call, because
    // neither cudf::concatenate nor a table view can carry the count of a
    // column-less table. Accumulated in 64 bits and checked before narrowing,
    // because a column-less payload is not bounded by a cuDF table the way a
    // column-bearing one is: cudf::concatenate would refuse to build a table
    // past the cudf::size_type ceiling, but nothing stops the logical count of
    // an empty layout from running past it. Mirrors zeroColumnBuildRows() in
    // CudfNestedLoopJoin.cpp.
    int64_t totalRows = 0;
    for (const auto& input : pendingInputs_) {
      totalRows += input->size();
    }
    VELOX_CHECK_LE(
        totalRows,
        std::numeric_limits<vector_size_t>::max(),
        "UCX exchange page exceeds the cuDF row limit: {} rows. Lower {} to "
        "split the payload into smaller pages.",
        totalRows,
        CudfConfig::kUcxPartitionedOutputBatchRows);
    const auto numRows = static_cast<vector_size_t>(totalRows);

    // An empty output layout means there is no GPU data at all, only rows to
    // account for. Nothing below can be asked to concatenate, hash, slice or
    // split such a table: cudf::concatenate collapses it to 0 rows and
    // cudf::contiguous_split returns no partitions at all, which
    // splitAndEnqueue would then index out of bounds.
    const bool hasColumns = outputType_->size() > 0;

    if (pendingInputs_.size() == 1 || !hasColumns) {
      // Fast path: use the single input's view directly (no GPU alloc). With no
      // columns there is nothing to merge either, so any input's view will do.
      auto& cv = pendingInputs_[0];
      stream = cv->stream();
      tableView = remap_.empty()
          ? cv->getTableView()
          : cv->getTableView().select(remap_.begin(), remap_.end());
    } else {
      // Collect (remapped) table views.
      std::vector<cudf::table_view> views;
      std::vector<rmm::cuda_stream_view> inputStreams;
      views.reserve(pendingInputs_.size());
      inputStreams.reserve(pendingInputs_.size());
      for (auto& v : pendingInputs_) {
        inputStreams.push_back(v->stream());
        views.push_back(
            remap_.empty()
                ? v->getTableView()
                : v->getTableView().select(remap_.begin(), remap_.end()));
      }

      cudf::detail::join_streams(inputStreams, stream);
      mergedTable = cudf::concatenate(
          views, stream, cudf::get_current_device_resource_ref());

      orderCudfVectorDeallocationsAfterStream(
          pendingInputs_, inputStreams, stream);

      // Free input GPU memory before partitioning (peak = 2x -> 1x).
      pendingInputs_.clear();

      tableView = mergedTable->view();
    }

    // Partition + enqueue (identical to previous addInput logic).
    auto queueManager = sharedQueueManager();
    if (numPartitions_ > 1) {
      if (replicateNullsAndAny_) {
        // Replicating null partition keys presupposes partition keys, and keys
        // live in the payload, so an empty layout cannot reach here.
        VELOX_CHECK(
            hasColumns,
            "A partitioned output with an empty layout has no partition key to "
            "replicate nulls for");
        replicateNullsAndAnyThenPartition(tableView, numRows, stream);
      } else {
        partitionAndEnqueue(tableView, numRows, stream);
      }
    } else if (numRows > 0) {
      auto packedCols = cudf::pack(
          tableView, stream, cudf::get_current_device_resource_ref());
      stream.synchronize();
      auto packedColsPtr = std::make_unique<cudf::packed_columns>(
          std::move(packedCols.metadata), std::move(packedCols.gpu_data));
      queueManager->enqueue(
          this->taskId(), 0, std::move(packedColsPtr), numRows);
    }

    // Check backpressure after enqueue.
    auto blocked = queueManager->checkBlocked(this->taskId(), &future_);
    if (blocked) {
      VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
              << " is blocked, can no longer write to output!";
    }
    blockingReason_ = blocked ? exec::BlockingReason::kWaitForConsumer
                              : exec::BlockingReason::kNotBlocked;

    pendingInputs_.clear();
    pendingRows_ = 0;

  } catch (const rmm::bad_alloc& e) {
    VLOG(1)
        << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
        << " caught memory alloc error, removing all memory in output queues";
    pendingInputs_.clear();
    pendingRows_ = 0;
    for (int i = 0; i < numPartitions_; i++) {
      sharedQueueManager()->deleteResults(this->taskId(), i);
    }
    throw;
  }
}

exec::BlockingReason UcxPartitionedOutput::isBlocked(ContinueFuture* future) {
  if (blockingReason_ != exec::BlockingReason::kNotBlocked) {
    *future = std::move(future_);
    blockingReason_ = exec::BlockingReason::kNotBlocked;
    return exec::BlockingReason::kWaitForConsumer;
  }
  return exec::BlockingReason::kNotBlocked;
}

RowVectorPtr UcxPartitionedOutput::getOutput() {
  VELOX_NVTX_OPERATOR_FUNC_RANGE();
  if (finished_) {
    return nullptr;
  }
  if (noMoreInput_) {
    flushPending(); // drain any remaining buffered inputs
    sharedQueueManager()->noMoreData(this->taskId());
    finished_ = true;
  }
  return nullptr;
}

bool UcxPartitionedOutput::isFinished() {
  return finished_;
}

std::shared_ptr<facebook::velox::ucx_exchange::UcxOutputQueueManager>
UcxPartitionedOutput::sharedQueueManager() {
  auto shared_queueManager = queueManager_.lock();
  VELOX_CHECK_NOT_NULL(
      shared_queueManager, "OutputQueueManager was already destructed");
  return shared_queueManager;
}

void UcxPartitionedOutput::initPartitionKeys(
    const std::shared_ptr<const core::PartitionedOutputNode>& planNode) {
  // Following Logic copied direcly from CudLocalPartition (!)

  // Following is IMO a hacky way to get the partition key indices. It is to
  // workaround the fact that the partition spec constructs the hash function
  // directly and has no public methods to get the partition key indices.

  // When the operator is of type kRepartition, the partition spec is a string
  // in the format "HASH(key1, key2, ...)"
  // We're going to extract the keys between HASH( and ) and find their indices
  // in the output row type.

  // When operator is of type kGather, we don't need to store any partition key
  // indices because we're going to merge all the incoming streams together.

  // Get partition function specification string
  spec_ = planNode->partitionFunctionSpec().toString();

  // Only parse keys if it's a hash function
  if (spec_.find("HASH(") != std::string::npos) {
    // Extract keys between HASH( and )
    size_t start = spec_.find("HASH(") + 5;
    size_t end = spec_.find(")", start);
    if (start != std::string::npos && end != std::string::npos) {
      std::string keysStr = spec_.substr(start, end - start);

      // Split by comma to get individual keys.
      std::vector<std::string> keys;
      size_t pos = 0;
      while ((pos = keysStr.find(",")) != std::string::npos) {
        std::string key = keysStr.substr(0, pos);
        keys.push_back(key);
        keysStr.erase(0, pos + 1);
      }
      keys.push_back(keysStr); // Add the last key.

      // Find field indices for each key.
      const auto& rowType = planNode->outputType();
      for (const auto& key : keys) {
        auto trimmedKey = key;
        // Trim whitespace
        trimmedKey.erase(0, trimmedKey.find_first_not_of(" "));
        trimmedKey.erase(trimmedKey.find_last_not_of(" ") + 1);

        auto fieldIndex = rowType->getChildIdx(trimmedKey);
        partitionKeyIndices_.push_back(fieldIndex);
      }
    }
  }
}

void UcxPartitionedOutput::partitionAndEnqueue(
    cudf::table_view tableView,
    vector_size_t numRows,
    rmm::cuda_stream_view stream) {
  if (tableView.num_columns() == 0) {
    // No columns means no partition key to hash on -- initPartitionKeys()
    // resolves keys through the output row type, so a HASH spec over an empty
    // layout fails there long before this point -- and no data to split. Only
    // the row count has to reach the destinations.
    equalPartitionRowCountOnly(tableView, numRows, stream);
    return;
  }
  if (partitionKeyIndices_.size() > 0 || spec_ == "gather") {
    hashPartition(tableView, stream);
  } else {
    equalPartition(tableView, stream);
  }
}

void UcxPartitionedOutput::equalPartitionRowCountOnly(
    cudf::table_view tableView,
    vector_size_t numRows,
    rmm::cuda_stream_view stream) {
  VELOX_CHECK_EQ(
      tableView.num_columns(), 0, "Expected a column-less payload here");
  if (numRows == 0) {
    return;
  }

  auto mr = cudf::get_current_device_resource_ref();
  // Same boundaries equalPartition() computes, so the split is identical to the
  // column-bearing case and the rows still add up to numRows.
  // The products are formed in 64 bits: numRows * (destination + 1) overflows
  // int32 well before numRows itself does. Each share fits vector_size_t
  // because it cannot exceed numRows.
  std::vector<vector_size_t> rowsPerDestination(numPartitions_);
  int64_t start = 0;
  for (size_t destination = 0; destination < numPartitions_; ++destination) {
    const int64_t end = static_cast<int64_t>(numRows) *
        static_cast<int64_t>(destination + 1) /
        static_cast<int64_t>(numPartitions_);
    rowsPerDestination[destination] = static_cast<vector_size_t>(end - start);
    start = end;
  }

  // One private packed copy per destination: the intra-node transfer path moves
  // the members out of a packed_columns, which would corrupt a shared one.
  std::vector<std::unique_ptr<cudf::packed_columns>> perDestination(
      numPartitions_);
  for (size_t destination = 0; destination < numPartitions_; ++destination) {
    if (rowsPerDestination[destination] == 0) {
      continue;
    }
    auto packed = cudf::pack(tableView, stream, mr);
    perDestination[destination] = std::make_unique<cudf::packed_columns>(
        std::move(packed.metadata), std::move(packed.gpu_data));
  }
  // UCX is not stream aware, so the packs must be complete before enqueueing.
  stream.synchronize();

  auto queueManager = sharedQueueManager();
  for (size_t destination = 0; destination < numPartitions_; ++destination) {
    if (perDestination[destination] == nullptr) {
      continue;
    }
    queueManager->enqueue(
        this->taskId(),
        static_cast<int>(destination),
        std::move(perDestination[destination]),
        rowsPerDestination[destination]);
  }
}

void UcxPartitionedOutput::replicateNullsAndAnyThenPartition(
    cudf::table_view tableView,
    vector_size_t numRows,
    rmm::cuda_stream_view stream) {
  // This path only runs for a payload with partition keys, so the table can
  // report its own rows and the two counts must agree.
  VELOX_CHECK_EQ(tableView.num_rows(), numRows);
  if (numRows == 0) {
    return;
  }

  bool anyKeyHasNulls = false;
  for (const auto keyIndex : partitionKeyIndices_) {
    if (tableView.column(static_cast<cudf::size_type>(keyIndex)).null_count() >
        0) {
      anyKeyHasNulls = true;
      break;
    }
  }
  const bool needsArbitraryRow = !replicatedAnyRow_;

  // Nothing to replicate, so route exactly as an operator without the flag.
  if (!anyKeyHasNulls && !needsArbitraryRow) {
    partitionAndEnqueue(tableView, numRows, stream);
    return;
  }

  auto mr = cudf::get_current_device_resource_ref();

  // Only the arbitrary row needs replicating, so slicing avoids a gather.
  if (!anyKeyHasNulls) {
    // num_rows() is safe to slice on here: the check above established that it
    // equals numRows, because this path always has partition key columns.
    const auto slices = cudf::slice(tableView, {0, 1, 1, tableView.num_rows()});
    packAndEnqueueToAllDestinations(slices[0], stream);
    replicatedAnyRow_ = true;
    if (slices[1].num_rows() > 0) {
      partitionAndEnqueue(slices[1], slices[1].num_rows(), stream);
    }
    return;
  }

  // A row is replicated when any of its partition keys is null, matching
  // exec::PartitionedOutput::collectNullRows(). cudf::is_null yields a
  // non-nullable BOOL8 column, which is what the stream compaction below needs.
  std::unique_ptr<cudf::column> replicateMask;
  for (const auto keyIndex : partitionKeyIndices_) {
    const auto keyColumn =
        tableView.column(static_cast<cudf::size_type>(keyIndex));
    if (keyColumn.null_count() == 0) {
      continue;
    }
    auto keyIsNull = cudf::is_null(keyColumn, stream, mr);
    if (replicateMask == nullptr) {
      replicateMask = std::move(keyIsNull);
    } else {
      replicateMask = cudf::binary_operation(
          replicateMask->view(),
          keyIsNull->view(),
          cudf::binary_operator::LOGICAL_OR,
          cudf::data_type{cudf::type_id::BOOL8},
          stream,
          mr);
    }
  }
  VELOX_CHECK_NOT_NULL(replicateMask, "Null partition key mask is null");

  // The arbitrary row rides along in the same mask, so it is replicated exactly
  // once per destination even when its own key is null.
  if (needsArbitraryRow) {
    auto maskView = replicateMask->mutable_view();
    const auto trueScalar = cudf::numeric_scalar<bool>(true, true, stream);
    cudf::fill_in_place(maskView, 0, 1, trueScalar, stream);
  }

  // apply_boolean_mask keeps the true rows and apply_deletion_mask keeps the
  // false ones, so the two results are an exact partition of the input: no row
  // is both replicated and routed, and none is dropped.
  const auto replicatedRows =
      cudf::apply_boolean_mask(tableView, replicateMask->view(), stream, mr);
  const auto routedRows =
      cudf::apply_deletion_mask(tableView, replicateMask->view(), stream, mr);

  packAndEnqueueToAllDestinations(replicatedRows->view(), stream);
  replicatedAnyRow_ = true;

  // Removing the replicated rows before hashing leaves every remaining row on
  // the destination it would have had otherwise, so co-partitioned joins that
  // rely on this partitioning still line up.
  if (routedRows->num_rows() > 0) {
    partitionAndEnqueue(routedRows->view(), routedRows->num_rows(), stream);
  }
}

void UcxPartitionedOutput::packAndEnqueueToAllDestinations(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  // Only reached for a payload with partition keys, so num_rows() is the real
  // count here. A column-less payload goes through equalPartitionRowCountOnly.
  VELOX_CHECK_GT(tableView.num_columns(), 0);
  if (tableView.num_rows() == 0) {
    return;
  }

  auto mr = cudf::get_current_device_resource_ref();
  std::vector<std::unique_ptr<cudf::packed_columns>> perDestination;
  perDestination.reserve(numPartitions_);
  for (size_t destination = 0; destination < numPartitions_; ++destination) {
    auto packed = cudf::pack(tableView, stream, mr);
    perDestination.push_back(
        std::make_unique<cudf::packed_columns>(
            std::move(packed.metadata), std::move(packed.gpu_data)));
  }
  // UCX is not stream aware, so the packs must be complete before enqueueing.
  stream.synchronize();

  auto queueManager = sharedQueueManager();
  for (size_t destination = 0; destination < numPartitions_; ++destination) {
    queueManager->enqueue(
        this->taskId(),
        static_cast<int>(destination),
        std::move(perDestination[destination]),
        tableView.num_rows());
  }
}

void UcxPartitionedOutput::hashPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Hashing and partitioning into " << numPartitions_ << " chunks";

  // Use cudf hash partitioning
  std::vector<cudf::size_type> partitionKeyIndices;
  for (const auto& idx : partitionKeyIndices_) {
    partitionKeyIndices.push_back(static_cast<cudf::size_type>(idx));
  }

  auto [partitionedTable, partitionOffsets] = cudf::hash_partition(
      tableView,
      partitionKeyIndices,
      numPartitions_,
      cudf::hash_id::HASH_MURMUR3,
      cudf::DEFAULT_HASH_SEED,
      stream);

  VELOX_CHECK_EQ(partitionOffsets.size(), numPartitions_ + 1);
  VELOX_CHECK_EQ(partitionOffsets[0], 0);

  // Erase first element since it's always 0 and we don't need it.
  partitionOffsets.erase(partitionOffsets.begin());
  partitionOffsets.pop_back();

  splitAndEnqueue(partitionedTable->view(), partitionOffsets, stream);
}

void UcxPartitionedOutput::equalPartition(
    cudf::table_view tableView,
    rmm::cuda_stream_view stream) {
  VLOG(3) << "@" << taskId() << "#" << pipelineId_ << "/" << driverId_
          << " Splitting into " << numPartitions_ << " chunks";
  std::vector<cudf::size_type> offsets;
  cudf::size_type size = tableView.num_rows();
  for (int i = 1; i < numPartitions_; ++i) {
    cudf::size_type idx = size * i / numPartitions_;
    offsets.push_back(idx);
  }
  splitAndEnqueue(tableView, offsets, stream);
}

void UcxPartitionedOutput::splitAndEnqueue(
    cudf::table_view tableView,
    std::vector<cudf::size_type> offsets,
    rmm::cuda_stream_view stream) {
  // cudf::contiguous_split returns no partitions at all for a column-less
  // table, which the loop below would index out of bounds. Such payloads are
  // routed to equalPartitionRowCountOnly instead and never arrive here.
  VELOX_CHECK_GT(tableView.num_columns(), 0);
  auto contiguousTables = cudf::contiguous_split(
      tableView, offsets, stream, cudf::get_current_device_resource_ref());

  // Synchronize the stream to ensure CUDA operations complete before enqueuing.
  // UCXX/UCX is not stream-aware, so without syncing, data could be sent before
  // the GPU kernels have finished writing to the buffers.
  stream.synchronize();

  VELOX_CHECK_EQ(
      offsets.size() + 1, numPartitions_, "mismatch in numPartitions_");
  auto queueManager = sharedQueueManager();
  for (int i = 0; i < numPartitions_; ++i) {
    auto const& partitionTable = contiguousTables[i];
    if (partitionTable.table.num_rows() == 0) {
      // Skip empty partitions.
      continue;
    }

    auto packedColsPtr = std::make_unique<cudf::packed_columns>(
        std::move(contiguousTables[i].data.metadata),
        std::move(contiguousTables[i].data.gpu_data));

    // enqueue partition data on Ucx Output Buffer
    queueManager->enqueue(
        this->taskId(),
        i,
        std::move(packedColsPtr),
        partitionTable.table.num_rows());
  }
}

} // namespace facebook::velox::ucx_exchange
