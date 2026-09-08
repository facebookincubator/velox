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
#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#include <cudf/null_mask.hpp>
#include <cudf/table/table.hpp>
#include <cudf/types.hpp>
#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <rmm/device_buffer.hpp>
#include <algorithm>
#include <memory>
#include <vector>
#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/Driver.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"
#include "velox/experimental/ucx-exchange/UcxOutputQueueManager.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

using facebook::velox::cudf_velox::CudfConfig;
using facebook::velox::exec::Task;

namespace facebook::velox::ucx_exchange {
namespace {

// Rows whose partition key is null, plus one arbitrary row, must reach every
// destination so that an anti-join can tell on every worker whether the build
// side was empty and whether it held a null key. Without that,
// cudf::hash_partition puts all null keys in one bucket and the other
// destinations wrongly report their rows as unmatched.
class UcxPartitionedOutputTest : public testing::Test {
 protected:
  static constexpr int kNumPartitions = 3;
  static constexpr cudf::size_type kNumRows = 6;
  // The sentinel sits under every null slot so a raw host read is unambiguous.
  static constexpr int32_t kNullSentinel = -1;

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    pool_ = memory::memoryManager()->addLeafPool();
    queueManager_ = UcxOutputQueueManager::getInstanceRef();
    CudfConfig::getInstance().exchange = true;
  }

  void TearDown() override {
    for (int destination = 0; destination < kNumPartitions; ++destination) {
      queueManager_->deleteResults(taskId_, destination);
    }
    queueManager_->removeTask(taskId_);
  }

  // Builds the key column, marking the rows in 'nullRows' invalid.
  std::unique_ptr<cudf::column> makeKeyColumn(
      const std::vector<int32_t>& values,
      const std::vector<cudf::size_type>& nullRows,
      rmm::cuda_stream_view stream) {
    const auto numRows = static_cast<cudf::size_type>(values.size());
    auto column = cudf::make_fixed_width_column(
        cudf::data_type{cudf::type_id::INT32},
        numRows,
        cudf::mask_state::ALL_VALID,
        stream);
    auto view = column->mutable_view();
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        view.data<int32_t>(),
        values.data(),
        values.size() * sizeof(int32_t),
        cudaMemcpyHostToDevice,
        stream.value()));
    for (const auto row : nullRows) {
      cudf::set_null_mask(view.null_mask(), row, row + 1, false, stream);
    }
    // Counted off the mask instead of taken from 'nullRows.size()', so that the
    // null_count every test asserts on reflects the bits actually cleared
    // above. Trusting the requested count would let a key column that carries
    // no nulls at all still report the expected number of them, and the
    // replicate-nulls probes would then be vacuous.
    column->set_null_count(
        cudf::null_count(view.null_mask(), 0, numRows, stream));
    stream.synchronize();
    return column;
  }

  // Creates one UcxPartitionedOutput driver over 'task'. Kept separate from
  // feeding batches so a single operator instance can take several of them:
  // the arbitrary replicated row is owed once per operator, and no test could
  // observe that through a helper that builds a fresh operator per batch.
  std::unique_ptr<UcxPartitionedOutput> makePartitionedOutput(
      const std::shared_ptr<Task>& task) {
    auto partitionedOutputNode =
        std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
            task->planFragment().planNode);
    VELOX_CHECK_NOT_NULL(partitionedOutputNode);

    // Outlives the operator, which holds a raw pointer to it.
    driverCtx_ = std::make_shared<exec::DriverCtx>(
        task,
        /*driverId=*/0,
        /*pipelineId=*/0,
        exec::kUngroupedGroupId,
        /*partitionId=*/0);
    // The operator normally gets its manager from the OutputTransportEntry
    // registered under core::TransportKind::kUcx; here it must be the same
    // process-wide instance the test drains from.
    return std::make_unique<UcxPartitionedOutput>(
        /*operatorId=*/0,
        driverCtx_.get(),
        partitionedOutputNode,
        queueManager_);
  }

  // Wraps 'keyColumn' in a CudfVector of kTestRowType and feeds it as one
  // batch, mirroring how SourceDriverMock drives the operator.
  void feedBatch(
      UcxPartitionedOutput* partitionedOutput,
      std::unique_ptr<cudf::column> keyColumn,
      rmm::cuda_stream_view stream) {
    const auto numRows = keyColumn->size();
    std::vector<std::unique_ptr<cudf::column>> columns;
    columns.push_back(std::move(keyColumn));
    columns.push_back(
        cudf::make_fixed_width_column(
            cudf::data_type{cudf::type_id::FLOAT64},
            numRows,
            cudf::mask_state::ALL_VALID,
            stream));
    columns.push_back(make_strings_column_from_host(
        std::vector<std::string>(numRows, "payload")));
    auto table = std::make_unique<cudf::table>(std::move(columns));
    stream.synchronize();

    auto cudfVector = std::make_shared<cudf_velox::CudfVector>(
        partitionedOutput->pool(),
        UcxTestData::kTestRowType,
        numRows,
        std::move(table),
        stream);

    partitionedOutput->addInput(cudfVector);
    partitionedOutput->getOutput();
  }

  // Runs the operator to completion, flushing whatever is still buffered.
  void finishPartitionedOutput(UcxPartitionedOutput* partitionedOutput) {
    partitionedOutput->noMoreInput();
    while (!partitionedOutput->isFinished()) {
      partitionedOutput->getOutput();
    }
  }

  // Feeds one batch through a single operator and runs it to completion.
  void runPartitionedOutput(
      const std::shared_ptr<Task>& task,
      std::unique_ptr<cudf::column> keyColumn,
      rmm::cuda_stream_view stream) {
    auto partitionedOutput = makePartitionedOutput(task);
    feedBatch(partitionedOutput.get(), std::move(keyColumn), stream);
    finishPartitionedOutput(partitionedOutput.get());
  }

  // Drains one destination and returns the key column of every packet that
  // arrived, one entry per packet, in arrival order. Packet boundaries carry
  // information a flat row list loses: each flush packs its replicated rows
  // separately, so a test can tell one flush from two by looking at which rows
  // travelled together.
  std::vector<std::vector<int32_t>> drainKeyPackets(int destination) {
    std::vector<std::vector<int32_t>> packets;
    while (true) {
      std::shared_ptr<cudf::packed_columns> payload;
      queueManager_->getData(
          taskId_,
          destination,
          [&payload](
              std::shared_ptr<cudf::packed_columns> data,
              vector_size_t /*numRows*/,
              std::vector<int64_t> /*remainingBytes*/) {
            payload = std::move(data);
          });
      // A null payload is the end-of-stream marker.
      if (payload == nullptr) {
        break;
      }
      auto unpacked = cudf::unpack(*payload);
      packets.push_back(
          getColVector<int32_t>(
              unpacked.column(0),
              unpacked.num_rows(),
              rmm::cuda_stream_default));
    }
    return packets;
  }

  // Flattens drainKeyPackets() for the tests that do not care which packet a
  // row arrived in.
  std::vector<int32_t> drainKeyColumn(int destination) {
    std::vector<int32_t> keys;
    for (const auto& packet : drainKeyPackets(destination)) {
      keys.insert(keys.end(), packet.begin(), packet.end());
    }
    return keys;
  }

  // Counts how many of the drained keys are the replicated sentinel.
  static int countSentinels(const std::vector<int32_t>& keys) {
    return static_cast<int>(
        std::count(keys.begin(), keys.end(), kNullSentinel));
  }

  static int countValue(const std::vector<int32_t>& keys, int32_t value) {
    return static_cast<int>(std::count(keys.begin(), keys.end(), value));
  }

  const std::string taskId_{"ucx-partitioned-output-test"};
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<UcxOutputQueueManager> queueManager_;
  std::shared_ptr<exec::DriverCtx> driverCtx_;
};

// The bug: both null-keyed rows land in a single hash bucket, so two of the
// three destinations receive none of them.
TEST_F(UcxPartitionedOutputTest, replicatesNullPartitionKeysToAllDestinations) {
  auto stream = rmm::cuda_stream_default;
  const std::vector<int32_t> keyValues{
      10, 20, kNullSentinel, 30, kNullSentinel, 40};
  const std::vector<cudf::size_type> nullRows{2, 4};
  const auto numNullRows = static_cast<int>(nullRows.size());

  auto keyColumn = makeKeyColumn(keyValues, nullRows, stream);
  // Guard against a vacuous probe: the null keys must really exist.
  ASSERT_EQ(keyColumn->view().null_count(), numNullRows);

  auto task = createPartitionedOutputTask(
      taskId_,
      pool_,
      UcxTestData::kTestRowType,
      kNumPartitions,
      {"c0"},
      FOUR_GBYTES,
      {},
      /*replicateNullsAndAny=*/true);
  queueManager_->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      kNumPartitions,
      /*numDrivers=*/1);

  runPartitionedOutput(task, std::move(keyColumn), stream);

  int totalRows = 0;
  std::vector<int> nullKeysPerDestination;
  for (int destination = 0; destination < kNumPartitions; ++destination) {
    const auto keys = drainKeyColumn(destination);
    totalRows += static_cast<int>(keys.size());
    nullKeysPerDestination.push_back(countSentinels(keys));
  }

  // Every destination must see both null-keyed rows. Deliberately says nothing
  // about which destination the non-null rows reach, since that depends on the
  // cudf hash.
  EXPECT_THAT(nullKeysPerDestination, testing::Each(numNullRows));
  // The replicated and routed halves are an exact partition of the input, so
  // the two null keys and the arbitrary row reach all three destinations while
  // the remaining three rows are routed once each. A fix that replicated the
  // null rows and also let them be hashed would push this over.
  EXPECT_EQ(
      totalRows,
      kNumPartitions * (numNullRows + 1) + (kNumRows - numNullRows - 1));
}

// Negative control for the two tests around it: with the flag off, the operator
// must not replicate anything. Without this, a harness that lost the ability to
// pass replicateNullsAndAny=false -- or a plan builder that quietly forced it
// on -- would leave every assertion above passing for the wrong reason.
TEST_F(UcxPartitionedOutputTest, routesNullPartitionKeysByHashWithoutTheFlag) {
  auto stream = rmm::cuda_stream_default;
  const std::vector<int32_t> keyValues{
      10, 20, kNullSentinel, 30, kNullSentinel, 40};
  const std::vector<cudf::size_type> nullRows{2, 4};

  auto keyColumn = makeKeyColumn(keyValues, nullRows, stream);
  ASSERT_EQ(keyColumn->view().null_count(), static_cast<int>(nullRows.size()));

  auto task = createPartitionedOutputTask(
      taskId_,
      pool_,
      UcxTestData::kTestRowType,
      kNumPartitions,
      {"c0"},
      FOUR_GBYTES,
      {},
      /*replicateNullsAndAny=*/false);
  queueManager_->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      kNumPartitions,
      /*numDrivers=*/1);

  runPartitionedOutput(task, std::move(keyColumn), stream);

  int totalRows = 0;
  int totalNullKeys = 0;
  for (int destination = 0; destination < kNumPartitions; ++destination) {
    const auto keys = drainKeyColumn(destination);
    totalRows += static_cast<int>(keys.size());
    totalNullKeys += countSentinels(keys);
  }

  // Every row is delivered exactly once, so nothing was replicated. Says
  // nothing about which destination received the null keys, only that they were
  // not copied.
  EXPECT_EQ(totalRows, kNumRows);
  EXPECT_EQ(totalNullKeys, static_cast<int>(nullRows.size()));
}

// The "and any" half of the contract, which no existing test covers: with no
// null keys at all, one arbitrary row still reaches every destination.
TEST_F(UcxPartitionedOutputTest, replicatesOneArbitraryRowWithNullFreeKeys) {
  auto stream = rmm::cuda_stream_default;
  const std::vector<int32_t> keyValues{10, 20, 30, 40, 50, 60};

  auto keyColumn = makeKeyColumn(keyValues, {}, stream);
  ASSERT_EQ(keyColumn->view().null_count(), 0);

  auto task = createPartitionedOutputTask(
      taskId_,
      pool_,
      UcxTestData::kTestRowType,
      kNumPartitions,
      {"c0"},
      FOUR_GBYTES,
      {},
      /*replicateNullsAndAny=*/true);
  queueManager_->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      kNumPartitions,
      /*numDrivers=*/1);

  runPartitionedOutput(task, std::move(keyColumn), stream);

  int totalRows = 0;
  std::vector<int> firstRowCopies;
  for (int destination = 0; destination < kNumPartitions; ++destination) {
    const auto keys = drainKeyColumn(destination);
    totalRows += static_cast<int>(keys.size());
    firstRowCopies.push_back(countValue(keys, keyValues[0]));
  }

  // Row 0 reaches every destination exactly once, and is not additionally
  // routed by hash, so the total is the 5 remaining rows plus 3 copies.
  EXPECT_THAT(firstRowCopies, testing::Each(1));
  EXPECT_EQ(totalRows, kNumRows - 1 + kNumPartitions);
}

// The arbitrary row is owed once per operator, not once per flush. A fix that
// forces row 0 into the replicate mask on every flush duplicates the first row
// of every later batch across all destinations, and the join on the other side
// then counts those rows several times.
TEST_F(UcxPartitionedOutputTest, replicatesArbitraryRowOncePerOperator) {
  auto stream = rmm::cuda_stream_default;
  // Disjoint value ranges, so each batch's row 0 is identifiable at the
  // destination. The second batch carries the nulls, which makes the two
  // flushes distinguishable: see the packet check below.
  const std::vector<int32_t> firstKeyValues{10, 20, 30, 40, 50, 60};
  const std::vector<int32_t> secondKeyValues{
      110, 120, kNullSentinel, 140, kNullSentinel, 160};
  const std::vector<cudf::size_type> secondNullRows{2, 4};
  const auto numNullRows = static_cast<int>(secondNullRows.size());

  auto firstKeyColumn = makeKeyColumn(firstKeyValues, {}, stream);
  auto secondKeyColumn = makeKeyColumn(secondKeyValues, secondNullRows, stream);
  // The first batch owes only the arbitrary row, the second only its nulls.
  ASSERT_EQ(firstKeyColumn->view().null_count(), 0);
  ASSERT_EQ(secondKeyColumn->view().null_count(), numNullRows);

  // One row per chunk, so each addInput flushes on its own. Without this the
  // default 10'000-row threshold concatenates both batches into a single
  // flush, and no assertion below could tell per-operator from per-flush.
  const std::unordered_map<std::string, std::string> extraConfig{
      {CudfConfig::kUcxPartitionedOutputBatchRows, "1"}};

  auto task = createPartitionedOutputTask(
      taskId_,
      pool_,
      UcxTestData::kTestRowType,
      kNumPartitions,
      {"c0"},
      FOUR_GBYTES,
      extraConfig,
      /*replicateNullsAndAny=*/true);
  queueManager_->initializeTask(
      task,
      core::PartitionedOutputNode::Kind::kPartitioned,
      kNumPartitions,
      /*numDrivers=*/1);

  auto partitionedOutput = makePartitionedOutput(task);
  feedBatch(partitionedOutput.get(), std::move(firstKeyColumn), stream);
  feedBatch(partitionedOutput.get(), std::move(secondKeyColumn), stream);
  finishPartitionedOutput(partitionedOutput.get());

  int totalRows = 0;
  int totalSecondBatchFirstRowCopies = 0;
  std::vector<int> firstBatchFirstRowCopies;
  std::vector<int> nullKeysPerDestination;
  for (int destination = 0; destination < kNumPartitions; ++destination) {
    int firstRowCopiesHere = 0;
    int nullKeysHere = 0;
    for (const auto& packet : drainKeyPackets(destination)) {
      totalRows += static_cast<int>(packet.size());
      const auto firstRowCopiesInPacket = countValue(packet, firstKeyValues[0]);
      firstRowCopiesHere += firstRowCopiesInPacket;
      nullKeysHere += countSentinels(packet);
      totalSecondBatchFirstRowCopies += countValue(packet, secondKeyValues[0]);

      // Proves the premise of this test, that the batches really were flushed
      // separately. The first batch is null-free, so its flush replicates the
      // arbitrary row alone; the second batch's null keys are replicated by a
      // later flush, in their own packet. Were both batches merged into one
      // flush, all three rows would be replicated in a single packet and the
      // arbitrary row would travel next to the sentinels -- and then the
      // per-flush assertion below would hold vacuously.
      if (firstRowCopiesInPacket > 0) {
        EXPECT_EQ(countSentinels(packet), 0);
      }
    }
    firstBatchFirstRowCopies.push_back(firstRowCopiesHere);
    nullKeysPerDestination.push_back(nullKeysHere);
  }

  // The first batch's row 0 is the arbitrary row and reaches everybody.
  EXPECT_THAT(firstBatchFirstRowCopies, testing::Each(1));
  // The second batch's row 0 is not, because the debt was already paid. A
  // per-flush implementation delivers it kNumPartitions times instead. Summed
  // over destinations, so it does not depend on where the hash puts it.
  EXPECT_EQ(totalSecondBatchFirstRowCopies, 1);
  // Null keys are still replicated on every flush, unlike the arbitrary row.
  EXPECT_THAT(nullKeysPerDestination, testing::Each(numNullRows));
  EXPECT_EQ(
      totalRows,
      /*arbitrary row to every destination*/ kNumPartitions +
          /*first batch routed*/ (kNumRows - 1) +
          /*null keys to every destination*/ kNumPartitions * numNullRows +
          /*second batch routed*/ (kNumRows - numNullRows));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
