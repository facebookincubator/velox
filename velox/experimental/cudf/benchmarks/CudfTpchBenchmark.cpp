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

#include "velox/experimental/cudf/connectors/parquet/ParquetConfig.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetConnector.h"
#include "velox/experimental/cudf/connectors/parquet/ParquetTableHandle.h"
#include "velox/experimental/cudf/exec/CudfConversion.h"
#include "velox/experimental/cudf/exec/ToCudf.h"
#include "velox/experimental/cudf/tests/utils/ParquetConnectorTestBase.h"

#include "velox/benchmarks/tpch/TpchBenchmark.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::exec::test;
using namespace facebook::velox::dwio::common;

DEFINE_uint64(
    cudf_chunk_read_limit,
    0,
    "Output table chunk read limit for cudf::parquet_chunked_reader.");

DEFINE_uint64(
    cudf_pass_read_limit,
    0,
    "Pass read limit for cudf::parquet_chunked_reader.");

DEFINE_int32(
    cudf_gpu_batch_size_rows,
    100000,
    "Preferred output batch size in rows for cudf operators.");

class CudfTpchBenchmark : public TpchBenchmark {
 public:
  void initialize() override {
    TpchBenchmark::initialize();

    // Add new values into the parquet configuration...
    auto parquetConfigurationValues =
        std::unordered_map<std::string, std::string>();
    parquetConfigurationValues
        [cudf_velox::connector::parquet::ParquetConfig::kMaxChunkReadLimit] =
            std::to_string(FLAGS_cudf_chunk_read_limit);
    parquetConfigurationValues
        [cudf_velox::connector::parquet::ParquetConfig::kMaxPassReadLimit] =
            std::to_string(FLAGS_cudf_pass_read_limit);
    parquetConfigurationValues[cudf_velox::connector::parquet::ParquetConfig::
                                   kAllowMismatchedParquetSchemas] =
        std::to_string(true);
    auto parquetProperties = std::make_shared<const config::ConfigBase>(
        std::move(parquetConfigurationValues));

    // Create parquet connector with config...
    connector::registerConnectorFactory(
        std::make_shared<
            cudf_velox::connector::parquet::ParquetConnectorFactory>());
    auto parquetConnector =
        connector::getConnectorFactory(
            cudf_velox::connector::parquet::ParquetConnectorFactory::
                kParquetConnectorName)
            ->newConnector(
                cudf_velox::exec::test::kParquetConnectorId,
                parquetProperties,
                ioExecutor_.get());
    connector::registerConnector(parquetConnector);

    // Enable cuDF operators
    cudf_velox::registerCudf();

    // Add custom configs
    queryConfigs_
        [facebook::velox::cudf_velox::CudfFromVelox::kGpuBatchSizeRows] =
            std::to_string(FLAGS_cudf_gpu_batch_size_rows);
  }

  exec::test::TpchPlan editQueryPlan(
      exec::test::TpchPlan planContext) override {
    auto plan = planContext.plan;

    // Function to recursively replace TableScanNode instances
    std::function<core::PlanNodePtr(const core::PlanNodePtr&)>
        replaceTableScans =
            [&](const core::PlanNodePtr& node) -> core::PlanNodePtr {
      // Check if this is a TableScanNode
      if (auto tableScanNode =
              std::dynamic_pointer_cast<const core::TableScanNode>(node)) {
        // Create a new TableScanNode with "cudf" appended to the ID
        auto handle = tableScanNode->tableHandle();
        if (auto hiveHandle = std::dynamic_pointer_cast<
                const connector::hive::HiveTableHandle>(handle)) {
          // Convert to parquet handle
          // Get remaining filters
          auto remainingFilter = hiveHandle->remainingFilter();

          // subfieldFilters() returns a const reference. We need a modifiable
          // copy that we can move into the ParquetTableHandle without invoking
          // a copy constructor on the Subfield keys (which are not
          // copy-constructible). Perform a deep copy using Subfield::clone().

          const auto& subfieldFilters = hiveHandle->subfieldFilters();

          common::SubfieldFilters parquetSubfieldFilters;
          parquetSubfieldFilters.reserve(subfieldFilters.size());
          for (const auto& [k, v] : subfieldFilters) {
            parquetSubfieldFilters.emplace(k.clone(), v);
          }

          // Create ParquetTableHandle using the movable copy of filters.
          auto parquetHandle = std::make_shared<
              cudf_velox::connector::parquet::ParquetTableHandle>(
              cudf_velox::exec::test::kParquetConnectorId,
              hiveHandle->tableName(),
              true,
              std::move(parquetSubfieldFilters),
              remainingFilter,
              hiveHandle->dataColumns());
          // Create new TableScanNode with parquet handle
          auto newTableScanNode = core::TableScanNode::Builder(*tableScanNode)
                                      .tableHandle(parquetHandle)
                                      .build();
          return newTableScanNode;
        }
        return tableScanNode;
      }

      // For non-TableScanNode, recursively process sources
      auto sources = node->sources();
      if (sources.empty()) {
        // Leaf node (not TableScan), return as-is
        return node;
      }

      // Process all sources
      std::vector<core::PlanNodePtr> newSources;
      bool hasChanges = false;

      for (const auto& source : sources) {
        auto newSource = replaceTableScans(source);
        newSources.push_back(newSource);
        if (newSource != source) {
          hasChanges = true;
        }
      }

      // If no changes to sources, return original node
      if (!hasChanges) {
        return node;
      }

      // Create new node with updated sources
      // This is a simplified approach - for a complete implementation,
      // you would need to handle each specific node type
      // For now, we'll use dynamic casting and builders for common types

      if (auto filterNode =
              std::dynamic_pointer_cast<const core::FilterNode>(node)) {
        return core::FilterNode::Builder(*filterNode)
            .source(newSources[0])
            .build();
      } else if (
          auto projectNode =
              std::dynamic_pointer_cast<const core::ProjectNode>(node)) {
        return core::ProjectNode::Builder(*projectNode)
            .source(newSources[0])
            .build();
      } else if (
          auto aggNode =
              std::dynamic_pointer_cast<const core::AggregationNode>(node)) {
        return core::AggregationNode::Builder(*aggNode)
            .source(newSources[0])
            .build();
      } else if (
          auto orderByNode =
              std::dynamic_pointer_cast<const core::OrderByNode>(node)) {
        return core::OrderByNode::Builder(*orderByNode)
            .source(newSources[0])
            .build();
      } else if (
          auto limitNode =
              std::dynamic_pointer_cast<const core::LimitNode>(node)) {
        return core::LimitNode::Builder(*limitNode)
            .source(newSources[0])
            .build();
      } else if (
          auto hashJoinNode =
              std::dynamic_pointer_cast<const core::HashJoinNode>(node)) {
        return core::HashJoinNode::Builder(*hashJoinNode)
            .left(newSources[0])
            .right(newSources[1])
            .build();
      } else if (
          auto localPartitionNode =
              std::dynamic_pointer_cast<const core::LocalPartitionNode>(node)) {
        return core::LocalPartitionNode::Builder(*localPartitionNode)
            .sources(newSources)
            .build();
      } else if (
          auto partitionedOutputNode =
              std::dynamic_pointer_cast<const core::PartitionedOutputNode>(
                  node)) {
        return core::PartitionedOutputNode::Builder(*partitionedOutputNode)
            .source(newSources[0])
            .build();
      } else if (
          auto valuesNode =
              std::dynamic_pointer_cast<const core::ValuesNode>(node)) {
        // ValuesNode has no sources, return as-is
        return valuesNode;
      } else if (
          auto arrowStreamNode =
              std::dynamic_pointer_cast<const core::ArrowStreamNode>(node)) {
        // ArrowStreamNode has no sources, return as-is
        return arrowStreamNode;
      } else if (
          auto traceScanNode =
              std::dynamic_pointer_cast<const core::TraceScanNode>(node)) {
        // TraceScanNode has no sources, return as-is
        return traceScanNode;
      } else if (
          auto tableWriteNode =
              std::dynamic_pointer_cast<const core::TableWriteNode>(node)) {
        return core::TableWriteNode::Builder(*tableWriteNode)
            .source(newSources[0])
            .build();
      } else if (
          auto tableWriteMergeNode =
              std::dynamic_pointer_cast<const core::TableWriteMergeNode>(
                  node)) {
        return core::TableWriteMergeNode::Builder(*tableWriteMergeNode)
            .source(newSources[0])
            .build();
      } else if (
          auto expandNode =
              std::dynamic_pointer_cast<const core::ExpandNode>(node)) {
        return core::ExpandNode::Builder(*expandNode)
            .source(newSources[0])
            .build();
      } else if (
          auto groupIdNode =
              std::dynamic_pointer_cast<const core::GroupIdNode>(node)) {
        return core::GroupIdNode::Builder(*groupIdNode)
            .source(newSources[0])
            .build();
      } else if (
          auto exchangeNode =
              std::dynamic_pointer_cast<const core::ExchangeNode>(node)) {
        // ExchangeNode has no sources, return as-is
        return exchangeNode;
      } else if (
          auto mergeExchangeNode =
              std::dynamic_pointer_cast<const core::MergeExchangeNode>(node)) {
        // MergeExchangeNode has no sources, return as-is
        return mergeExchangeNode;
      } else if (
          auto localMergeNode =
              std::dynamic_pointer_cast<const core::LocalMergeNode>(node)) {
        return core::LocalMergeNode::Builder(*localMergeNode)
            .sources(newSources)
            .build();
      } else if (
          auto mergeJoinNode =
              std::dynamic_pointer_cast<const core::MergeJoinNode>(node)) {
        return core::MergeJoinNode::Builder(*mergeJoinNode)
            .left(newSources[0])
            .right(newSources[1])
            .build();
      } else if (
          auto indexLookupJoinNode =
              std::dynamic_pointer_cast<const core::IndexLookupJoinNode>(
                  node)) {
        return core::IndexLookupJoinNode::Builder(*indexLookupJoinNode)
            .left(newSources[0])
            .right(std::dynamic_pointer_cast<const core::TableScanNode>(
                newSources[1]))
            .build();
      } else if (
          auto nestedLoopJoinNode =
              std::dynamic_pointer_cast<const core::NestedLoopJoinNode>(node)) {
        return core::NestedLoopJoinNode::Builder(*nestedLoopJoinNode)
            .left(newSources[0])
            .right(newSources[1])
            .build();
      } else if (
          auto topNNode =
              std::dynamic_pointer_cast<const core::TopNNode>(node)) {
        return core::TopNNode::Builder(*topNNode).source(newSources[0]).build();
      } else if (
          auto unnestNode =
              std::dynamic_pointer_cast<const core::UnnestNode>(node)) {
        return core::UnnestNode::Builder(*unnestNode)
            .source(newSources[0])
            .build();
      } else if (
          auto enforceSingleRowNode =
              std::dynamic_pointer_cast<const core::EnforceSingleRowNode>(
                  node)) {
        return core::EnforceSingleRowNode::Builder(*enforceSingleRowNode)
            .source(newSources[0])
            .build();
      } else if (
          auto assignUniqueIdNode =
              std::dynamic_pointer_cast<const core::AssignUniqueIdNode>(node)) {
        return core::AssignUniqueIdNode::Builder(*assignUniqueIdNode)
            .source(newSources[0])
            .build();
      } else if (
          auto windowNode =
              std::dynamic_pointer_cast<const core::WindowNode>(node)) {
        return core::WindowNode::Builder(*windowNode)
            .source(newSources[0])
            .build();
      } else if (
          auto rowNumberNode =
              std::dynamic_pointer_cast<const core::RowNumberNode>(node)) {
        return core::RowNumberNode::Builder(*rowNumberNode)
            .source(newSources[0])
            .build();
      } else if (
          auto markDistinctNode =
              std::dynamic_pointer_cast<const core::MarkDistinctNode>(node)) {
        return core::MarkDistinctNode::Builder(*markDistinctNode)
            .source(newSources[0])
            .build();
      } else if (
          auto topNRowNumberNode =
              std::dynamic_pointer_cast<const core::TopNRowNumberNode>(node)) {
        return core::TopNRowNumberNode::Builder(*topNRowNumberNode)
            .source(newSources[0])
            .build();
      }

      // For any other node type, return the original (fallback)
      // In a complete implementation, you'd handle all node types
      return node;
    };

    // Replace the plan with the modified version
    planContext.plan = replaceTableScans(plan);

    return planContext;
  }

  std::vector<std::shared_ptr<connector::ConnectorSplit>> listSplits(
      const std::string& path,
      int32_t numSplitsPerFile,
      const exec::test::TpchPlan& plan) override {
    if (facebook::velox::cudf_velox::cudfIsRegistered() &&
        facebook::velox::connector::getAllConnectors().count(
            cudf_velox::exec::test::kParquetConnectorId) > 0 &&
        facebook::velox::cudf_velox::cudfTableScanEnabled()) {
      std::vector<std::shared_ptr<connector::ConnectorSplit>> result;
      auto temp = cudf_velox::exec::test::ParquetConnectorTestBase::
          makeParquetConnectorSplits(path, 1);
      for (auto& i : temp) {
        result.push_back(i);
      }
      return result;
    }

    return TpchBenchmark::listSplits(path, numSplitsPerFile, plan);
  }

  void shutdown() override {
    cudf_velox::unregisterCudf();
    connector::unregisterConnector(cudf_velox::exec::test::kParquetConnectorId);
    connector::unregisterConnectorFactory(
        cudf_velox::connector::parquet::ParquetConnectorFactory::
            kParquetConnectorName);
    TpchBenchmark::shutdown();
  }
};

int main(int argc, char** argv) {
  std::string kUsage(
      "This program benchmarks TPC-H queries. Run 'velox_cudf_tpch_benchmark -helpon=TpchBenchmark' for available options.\n");
  gflags::SetUsageMessage(kUsage);
  folly::Init init{&argc, &argv, false};
  benchmark = std::make_unique<CudfTpchBenchmark>();
  tpchBenchmarkMain();
}
