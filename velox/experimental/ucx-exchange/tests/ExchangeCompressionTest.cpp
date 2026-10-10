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
#include "velox/experimental/ucx-exchange/ExchangeCompression.h"
#include "velox/experimental/ucx-exchange/UcxExchangeSource.h"
#include "velox/experimental/ucx-exchange/UcxPartitionedOutput.h"
#include "velox/experimental/ucx-exchange/tests/UcxTestHelpers.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/contiguous_split.hpp>
#if __has_include(<cudf/detail/fused_for.hpp>)
#define VELOX_UCX_TEST_HAS_FUSED_FOR 1
#include <cudf/detail/fused_for.hpp>
#else
#define VELOX_UCX_TEST_HAS_FUSED_FOR 0
#endif
#include <cudf/table/table.hpp>
#include <gtest/gtest.h>
#include <rmm/cuda_stream.hpp>
#include <cstring>
#include <string_view>
#include <utility>

#include "velox/common/memory/MemoryPool.h"
#include "velox/exec/Driver.h"
#include "velox/experimental/cudf/CudfConfig.h"
#include "velox/experimental/cudf/vector/CudfVector.h"

namespace facebook::velox::ucx_exchange {
namespace {

using cudf_velox::CudfConfig;

class ExchangeCompressionTest : public testing::Test {
 protected:
  using Config = std::unordered_map<std::string, std::string>;
  struct Packet {
    std::shared_ptr<cudf::packed_columns> data;
    vector_size_t rows;
    int destination;
  };

  static void SetUpTestCase() {
    memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});
  }

  void SetUp() override {
    auto& config = CudfConfig::getInstance();
    previousCompression_ = config.exchangeCompression;
    previousExchange_ = config.exchange;
    config.exchangeCompression = "none";
    config.exchange = true;
    pool_ = memory::memoryManager()->addLeafPool();
    manager_ = UcxOutputQueueManager::getInstanceRef();
  }

  void TearDown() override {
    for (const auto& [id, partitions] : tasks_) {
      for (int destination = 0; destination < partitions; ++destination) {
        manager_->deleteResults(id, destination);
      }
      manager_->removeTask(id);
    }
    auto& config = CudfConfig::getInstance();
    config.exchangeCompression = previousCompression_;
    config.exchange = previousExchange_;
  }

  std::unique_ptr<UcxPartitionedOutput>
  start(const Config& config, int partitions, const RowTypePtr& type) {
    taskId_ = "cascaded-consumer-test-" + std::to_string(tasks_.size());
    task_ = createPartitionedOutputTask(
        taskId_, pool_, type, partitions, {}, FOUR_GBYTES, config);
    manager_->initializeTask(
        task_, core::PartitionedOutputNode::Kind::kPartitioned, partitions, 1);
    tasks_.emplace_back(taskId_, partitions);
    driver_ = std::make_shared<exec::DriverCtx>(
        task_, 0, 0, exec::kUngroupedGroupId, 0);
    auto plan = std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
        task_->planFragment().planNode);
    return std::make_unique<UcxPartitionedOutput>(
        0, driver_.get(), plan, manager_);
  }

  std::vector<Packet> produce(
      const std::vector<int32_t>& values,
      const Config& config = {},
      int partitions = 1,
      bool emptyLayout = false) {
    const auto type = emptyLayout ? ROW({}, {}) : ROW({"c0"}, {INTEGER()});
    auto output = start(config, partitions, type);
    const auto rows = static_cast<vector_size_t>(values.size());
    std::vector<std::unique_ptr<cudf::column>> columns;
    if (!emptyLayout) {
      auto column = cudf::make_fixed_width_column(
          cudf::data_type{cudf::type_id::INT32},
          rows,
          cudf::mask_state::UNALLOCATED,
          stream_,
          cudf::get_current_device_resource_ref());
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          column->mutable_view().data<int32_t>(),
          values.data(),
          values.size() * sizeof(int32_t),
          cudaMemcpyHostToDevice,
          stream_.value()));
      columns.push_back(std::move(column));
    }
    auto table = std::make_unique<cudf::table>(std::move(columns));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream_.value()));
    auto input = std::make_shared<cudf_velox::CudfVector>(
        output->pool(), type, rows, std::move(table), stream_);
    output->addInput(std::move(input));
    output->noMoreInput();
    while (!output->isFinished()) {
      output->getOutput();
    }

    std::vector<Packet> packets;
    for (int destination = 0; destination < partitions; ++destination) {
      while (true) {
        Packet packet{nullptr, 0, destination};
        manager_->getData(
            taskId_,
            destination,
            [&packet](
                std::shared_ptr<cudf::packed_columns> data,
                vector_size_t rows,
                std::vector<int64_t>) {
              packet.data = std::move(data);
              packet.rows = rows;
            });
        if (!packet.data) {
          break;
        }
        packets.push_back(std::move(packet));
      }
    }
    return packets;
  }

  PackedTableWithStreamPtr restore(Packet& packet, cuda::stream_ref stream) {
    auto received = detail::restoreReceivedTable(
        std::move(packet.data->metadata),
        std::move(packet.data->gpu_data),
        stream,
        packet.rows);
    packet.data.reset(); // Returned storage must not depend on the wire owner.
    return received;
  }

  static std::vector<int32_t> readColumn(
      const cudf::column_view& column,
      cuda::stream_ref stream) {
    std::vector<int32_t> values(column.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        values.data(),
        column.data<int32_t>(),
        values.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost,
        stream.get()));
    CUDF_CUDA_TRY(cudaStreamSynchronize(stream.get()));
    return values;
  }

  void expectCascaded(std::vector<Packet>& packets, vector_size_t totalRows) {
    vector_size_t rows = 0;
    rmm::cuda_stream consumer;
    for (auto& packet : packets) {
      auto decoded = unwrapExchangePayloadMetadata(
          std::make_unique<std::vector<uint8_t>>(*packet.data->metadata));
      ASSERT_EQ(decoded.codec, ExchangePayloadCodec::kCascaded);
      EXPECT_LT(packet.data->gpu_data->size(), decoded.logicalDataSize);
      // resize must change the transmitted size, not just metadata. Retained
      // allocation capacity is not the payload size and is not sent.
      EXPECT_LT(
          packet.data->gpu_data->size(), packet.data->gpu_data->capacity());
      auto received = restore(packet, consumer);
      ASSERT_TRUE(received->table);
      EXPECT_FALSE(received->packedTable);
      EXPECT_EQ(received->gpuDataSize(), decoded.logicalDataSize);
      EXPECT_EQ(received->numRows, packet.rows);
      EXPECT_EQ(received->tableView().num_rows(), packet.rows);
      EXPECT_EQ(
          readColumn(received->tableView().column(0), consumer),
          std::vector<int32_t>(packet.rows, 7));
      rows += packet.rows;
    }
    EXPECT_EQ(rows, totalRows);
  }

  static std::vector<int32_t> forValues() {
    // Every 32 KiB INT32 tile spans exactly five bits, including both halves
    // of the round-robin split. This distinguishes layouts and tile sizes.
    std::vector<int32_t> values(65536);
    for (size_t i = 0; i < values.size(); ++i) {
      values[i] = static_cast<int32_t>(i % 32) - 10;
    }
    return values;
  }

#if VELOX_UCX_TEST_HAS_FUSED_FOR
  void expectFusedFor(
      std::vector<Packet>& packets,
      bool bitpacked,
      const std::vector<int32_t>& values) {
    size_t rowStart = 0;
    rmm::cuda_stream consumer;
    for (auto& packet : packets) {
      auto metadata = unwrapExchangePayloadMetadata(
          std::make_unique<std::vector<uint8_t>>(*packet.data->metadata));
      ASSERT_EQ(metadata.codec, ExchangePayloadCodec::kFusedFor);
      ASSERT_GT(metadata.auxiliaryCount, 0);
      std::vector<cudf::detail::fused_for_segment> segments(
          metadata.auxiliaryCount);
      CUDF_CUDA_TRY(cudaMemcpyAsync(
          segments.data(),
          packet.data->gpu_data->data(),
          segments.size() * sizeof(segments.front()),
          cudaMemcpyDeviceToHost,
          consumer.value()));
      CUDF_CUDA_TRY(cudaStreamSynchronize(consumer.value()));
      constexpr size_t rowsPerTile = 32 * 1024 / sizeof(int32_t);
      size_t numericTiles = 0;
      size_t numericRows = 0;
      for (const auto& segment : segments) {
        if (segment.logical_width != sizeof(int32_t)) {
          continue;
        }
        ++numericTiles;
        numericRows += segment.element_count;
        EXPECT_LE(segment.element_count, rowsPerTile);
        EXPECT_EQ(segment.is_bitpacked(), bitpacked);
        EXPECT_EQ(segment.wire_width, bitpacked ? 5 : 1);
      }
      EXPECT_EQ(numericTiles, (packet.rows + rowsPerTile - 1) / rowsPerTile);
      EXPECT_EQ(numericRows, packet.rows);
      EXPECT_LT(packet.data->gpu_data->size(), metadata.logicalDataSize);

      auto received = restore(packet, consumer);
      ASSERT_TRUE(received->packedTable);
      EXPECT_FALSE(received->table);
      EXPECT_EQ(received->numRows, packet.rows);
      ASSERT_LE(rowStart + packet.rows, values.size());
      EXPECT_EQ(
          readColumn(received->tableView().column(0), consumer),
          std::vector<int32_t>(
              values.begin() + rowStart,
              values.begin() + rowStart + packet.rows));
      rowStart += packet.rows;
    }
    EXPECT_EQ(rowStart, values.size());
  }
#endif

  rmm::cuda_stream stream_;
  std::shared_ptr<memory::MemoryPool> pool_;
  std::shared_ptr<UcxOutputQueueManager> manager_;
  std::shared_ptr<exec::Task> task_;
  std::shared_ptr<exec::DriverCtx> driver_;
  std::string taskId_;
  std::vector<std::pair<std::string, int>> tasks_;
  std::string previousCompression_;
  bool previousExchange_;
};

TEST(ExchangeCompressionConfigTest, parsesSupportedSelectorNames) {
  std::vector<std::pair<std::string_view, ExchangeCompression>> modes = {
      {"none", ExchangeCompression::kNone},
      {"cascaded", ExchangeCompression::kCascaded}};
#if VELOX_UCX_TEST_HAS_FUSED_FOR
  modes.emplace_back(
      "fused-for-bitpacked", ExchangeCompression::kFusedForBitpacked);
  modes.emplace_back(
      "fused-for-byte-aligned", ExchangeCompression::kFusedForByteAligned);
#endif
  for (const auto& [name, mode] : modes) {
    EXPECT_EQ(parseExchangeCompression(name), mode);
  }
  EXPECT_EQ(fusedForAvailable(), VELOX_UCX_TEST_HAS_FUSED_FOR != 0);
#if !VELOX_UCX_TEST_HAS_FUSED_FOR
  EXPECT_THROW(parseExchangeCompression("fused-for-bitpacked"), VeloxUserError);
  EXPECT_THROW(
      parseExchangeCompression("fused-for-byte-aligned"), VeloxUserError);
#endif
}

TEST_F(ExchangeCompressionTest, disabledByDefaultRetainsRawPacking) {
  EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, "none");
  const std::vector<int32_t> values(65536, 7);
  auto packets = produce(values);
  ASSERT_EQ(packets.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*packets[0].data->metadata));
  EXPECT_EQ(metadata.codec, ExchangePayloadCodec::kNone);
  auto* allocation = packets[0].data->gpu_data.get();
  auto received = restore(packets[0], stream_);
  EXPECT_FALSE(received->table);
  ASSERT_TRUE(received->packedTable);
  EXPECT_EQ(received->packedTable->data.gpu_data.get(), allocation);
  EXPECT_EQ(readColumn(received->tableView().column(0), stream_), values);
}

TEST_F(ExchangeCompressionTest, cascadedSingleDestinationUsesOwningPath) {
  auto packets = produce(
      std::vector<int32_t>(65536, 7),
      {{CudfConfig::kUcxExchangeCompression, "cascaded"}});
  ASSERT_EQ(packets.size(), 1);
  expectCascaded(packets, 65536);
}

TEST_F(ExchangeCompressionTest, cascadedSplitDestinationsUseOwningPath) {
  // Establish that the fixture actually produces both destinations without
  // compression before checking the new codec path.
  auto raw = produce(std::vector<int32_t>(65536, 7), {}, 2);
  ASSERT_EQ(raw.size(), 2);
  for (auto& packet : raw) {
    EXPECT_EQ(packet.rows, 32768);
    auto received = restore(packet, stream_);
    ASSERT_TRUE(received->packedTable);
    EXPECT_EQ(
        readColumn(received->tableView().column(0), stream_),
        std::vector<int32_t>(32768, 7));
  }
  auto packets = produce(
      std::vector<int32_t>(65536, 7),
      {{CudfConfig::kUcxExchangeCompression, "cascaded"}},
      2);
  ASSERT_EQ(packets.size(), 2);
  EXPECT_EQ(packets[0].destination, 0);
  EXPECT_EQ(packets[1].destination, 1);
  EXPECT_EQ(packets[0].rows, 32768);
  EXPECT_EQ(packets[1].rows, 32768);
  expectCascaded(packets, 65536);
}

TEST_F(ExchangeCompressionTest, cascadedWithoutReductionKeepsRawPayload) {
  // Deterministic full-width values do not leave Cascaded enough redundancy
  // to offset its framing. This exercises the global no-gain fallback without
  // depending on the codec's exact header size.
  std::vector<int32_t> values(65536);
  uint32_t state = 0x9e3779b9;
  for (auto& value : values) {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    std::memcpy(&value, &state, sizeof(value));
  }
  auto packets =
      produce(values, {{CudfConfig::kUcxExchangeCompression, "cascaded"}});
  ASSERT_EQ(packets.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*packets[0].data->metadata));
  EXPECT_EQ(metadata.codec, ExchangePayloadCodec::kNone);
  auto received = restore(packets[0], stream_);
  ASSERT_TRUE(received->packedTable);
  EXPECT_FALSE(received->table);
  EXPECT_EQ(readColumn(received->tableView().column(0), stream_), values);
}

TEST_F(ExchangeCompressionTest, emptyLayoutKeepsProducerRowsAndRawStorage) {
  std::vector<const char*> modes{"none", "cascaded"};
#if VELOX_UCX_TEST_HAS_FUSED_FOR
  modes.push_back("fused-for-bitpacked");
  modes.push_back("fused-for-byte-aligned");
#endif
  for (const auto* mode : modes) {
    SCOPED_TRACE(mode);
    for (int partitions : {1, 2}) {
      SCOPED_TRACE(partitions);
      auto packets = produce(
          std::vector<int32_t>(7),
          {{CudfConfig::kUcxExchangeCompression, mode}},
          partitions,
          true);
      ASSERT_EQ(packets.size(), partitions);
      vector_size_t rows = 0;
      for (auto& packet : packets) {
        EXPECT_EQ(packet.data->gpu_data->size(), 0);
        auto received = restore(packet, stream_);
        ASSERT_TRUE(received->packedTable);
        EXPECT_FALSE(received->table);
        EXPECT_EQ(received->numRows, packet.rows);
        EXPECT_EQ(received->tableView().num_columns(), 0);
        rows += received->numRows;
      }
      EXPECT_EQ(rows, 7);
    }
  }
}

#if VELOX_UCX_TEST_HAS_FUSED_FOR
TEST_F(ExchangeCompressionTest, fusedForEnvelopeUsesDirectDecoder) {
  const std::vector<int32_t> values(65536, 7);
  auto packets = produce(
      values,
      {{CudfConfig::kUcxExchangeCompression, "fused-for-byte-aligned"}});
  ASSERT_EQ(packets.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*packets[0].data->metadata));
  ASSERT_EQ(metadata.codec, ExchangePayloadCodec::kFusedFor);
  EXPECT_GT(metadata.auxiliaryCount, 0);
  auto received = restore(packets[0], stream_);
  ASSERT_TRUE(received->packedTable);
  EXPECT_FALSE(received->table);
  EXPECT_EQ(readColumn(received->tableView().column(0), stream_), values);
}
#endif

TEST_F(ExchangeCompressionTest, rejectsInvalidSelector) {
  CudfConfig::getInstance().exchangeCompression = "cascaded";
  for (const auto* value :
       {"", "true", "fused-for", "automatic", "cascaded,fused-for-bitpacked"}) {
    SCOPED_TRACE(value);
    EXPECT_THROW(parseExchangeCompression(value), VeloxUserError);
    EXPECT_THROW(
        CudfConfig::getInstance().initialize(
            {{CudfConfig::kUcxExchangeCompression, value}}),
        VeloxUserError);
    EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, "cascaded");
    EXPECT_THROW(
        start(
            {{CudfConfig::kUcxExchangeCompression, value}},
            1,
            ROW({"c0"}, {INTEGER()})),
        VeloxUserError);
  }
}

TEST_F(ExchangeCompressionTest, processDefaultSupportsQueryOverride) {
  CudfConfig::getInstance().initialize(
      {{CudfConfig::kUcxExchangeCompression, "cascaded"}});
  EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, "cascaded");
  auto compressed = produce(std::vector<int32_t>(65536, 7));
  ASSERT_EQ(compressed.size(), 1);
  expectCascaded(compressed, 65536);
  auto raw = produce(
      std::vector<int32_t>(65536, 7),
      {{CudfConfig::kUcxExchangeCompression, "none"}});
  ASSERT_EQ(raw.size(), 1);
  auto metadata = unwrapExchangePayloadMetadata(
      std::make_unique<std::vector<uint8_t>>(*raw[0].data->metadata));
  EXPECT_EQ(metadata.codec, ExchangePayloadCodec::kNone);
#if VELOX_UCX_TEST_HAS_FUSED_FOR
  // A query chooses one codec, so FOR overrides Cascaded instead of
  // conflicting.
  auto fused = produce(
      forValues(),
      {{CudfConfig::kUcxExchangeCompression, "fused-for-bitpacked"}});
  ASSERT_EQ(fused.size(), 1);
  expectFusedFor(fused, true, forValues());
#endif
  EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, "cascaded");
}

#if VELOX_UCX_TEST_HAS_FUSED_FOR
TEST_F(
    ExchangeCompressionTest,
    bothForLayoutsUseExplicit32KiBTilesOnBothRoutes) {
  const auto values = forValues();
  for (const auto* mode : {"fused-for-bitpacked", "fused-for-byte-aligned"}) {
    SCOPED_TRACE(mode);
    for (int partitions : {1, 2}) {
      SCOPED_TRACE(partitions);
      auto packets = produce(
          values, {{CudfConfig::kUcxExchangeCompression, mode}}, partitions);
      ASSERT_EQ(packets.size(), partitions);
      for (int destination = 0; destination < partitions; ++destination) {
        EXPECT_EQ(packets[destination].destination, destination);
        EXPECT_EQ(packets[destination].rows, values.size() / partitions);
      }
      expectFusedFor(
          packets, std::string_view{mode} == "fused-for-bitpacked", values);
    }
  }
}

TEST_F(
    ExchangeCompressionTest,
    processForDefaultsAndQueryChoicesRemainIndependent) {
  const auto values = forValues();
  for (const auto* mode : {"fused-for-bitpacked", "fused-for-byte-aligned"}) {
    SCOPED_TRACE(mode);
    CudfConfig::getInstance().initialize(
        {{CudfConfig::kUcxExchangeCompression, mode}});
    const auto defaultMode = parseExchangeCompression(mode);
    EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, mode);
    // An unrelated initialization must not reset the configured default.
    CudfConfig::getInstance().initialize({});

    auto inherited = produce(values, {}, 2);
    ASSERT_EQ(inherited.size(), 2);
    expectFusedFor(
        inherited,
        defaultMode == ExchangeCompression::kFusedForBitpacked,
        values);

    const auto* otherMode =
        defaultMode == ExchangeCompression::kFusedForBitpacked
        ? "fused-for-byte-aligned"
        : "fused-for-bitpacked";
    auto overridden =
        produce(values, {{CudfConfig::kUcxExchangeCompression, otherMode}}, 2);
    ASSERT_EQ(overridden.size(), 2);
    expectFusedFor(
        overridden,
        std::string_view{otherMode} == "fused-for-bitpacked",
        values);

    auto cascaded = produce(
        std::vector<int32_t>(65536, 7),
        {{CudfConfig::kUcxExchangeCompression, "cascaded"}});
    ASSERT_EQ(cascaded.size(), 1);
    expectCascaded(cascaded, 65536);
    auto raw = produce(values, {{CudfConfig::kUcxExchangeCompression, "none"}});
    ASSERT_EQ(raw.size(), 1);
    EXPECT_EQ(
        unwrapExchangePayloadMetadata(
            std::make_unique<std::vector<uint8_t>>(*raw[0].data->metadata))
            .codec,
        ExchangePayloadCodec::kNone);
    auto received = restore(raw[0], stream_);
    ASSERT_TRUE(received->packedTable);
    EXPECT_FALSE(received->table);
    EXPECT_EQ(readColumn(received->tableView().column(0), stream_), values);
    EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, mode);
  }

  CudfConfig::getInstance().initialize(
      {{CudfConfig::kUcxExchangeCompression, "none"}});
  EXPECT_EQ(CudfConfig::getInstance().exchangeCompression, "none");
}
#endif

TEST_F(ExchangeCompressionTest, emptyInputProducesNoPacketsForAnyMode) {
  std::vector<const char*> modes{"none", "cascaded"};
#if VELOX_UCX_TEST_HAS_FUSED_FOR
  modes.push_back("fused-for-bitpacked");
  modes.push_back("fused-for-byte-aligned");
#endif
  for (const auto* mode : modes) {
    SCOPED_TRACE(mode);
    for (int partitions : {1, 2}) {
      EXPECT_TRUE(
          produce({}, {{CudfConfig::kUcxExchangeCompression, mode}}, partitions)
              .empty());
    }
  }
}

TEST_F(ExchangeCompressionTest, receiverRejectsTruncatedCascadedPayload) {
  auto packets = produce(
      std::vector<int32_t>(65536, 7),
      {{CudfConfig::kUcxExchangeCompression, "cascaded"}});
  ASSERT_EQ(packets.size(), 1);
  auto& wire = *packets[0].data->gpu_data;
  ASSERT_GT(wire.size(), 1);
  wire.resize(wire.size() - 1, stream_);
  EXPECT_ANY_THROW(restore(packets[0], stream_));
}

} // namespace
} // namespace facebook::velox::ucx_exchange
