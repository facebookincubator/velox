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
#include "velox4j/init/Init.h"

#include <folly/Conv.h>
#include <velox/common/base/Exceptions.h>
#include <velox/common/base/VeloxException.h>
#include <velox/common/config/Config.h>
#include <velox/common/file/FileSystems.h>
#include <velox/common/memory/Memory.h>
#include <velox/common/memory/MemoryPool.h>
#include <velox/connectors/Connector.h>
#include <velox/connectors/hive/HiveConnector.h>
#include <velox/connectors/hive/HiveConnectorSplit.h>
#include <velox/connectors/hive/HiveDataSink.h>
#include <velox/connectors/hive/TableHandle.h>
#include <velox/core/ITypedExpr.h>
#include <velox/core/PlanNode.h>
#include <velox/dwio/common/FileSink.h>
#include <velox/dwio/parquet/RegisterParquetReader.h>
#include <velox/dwio/parquet/RegisterParquetWriter.h>
#include <velox/exec/PartitionFunction.h>
#include <velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h>
#include <velox/functions/prestosql/window/WindowFunctionsRegistration.h>
#include <velox/functions/sparksql/aggregates/Register.h>
#include <velox/functions/sparksql/registration/Register.h>
#include <velox/functions/sparksql/window/WindowFunctionsRegistration.h>
#include <velox/type/Filter.h>
#include <velox/type/Type.h>
#include <atomic>
#include <functional>
#include <string>
#include <unordered_map>

#include "velox4j/conf/Config.h"
#include "velox4j/connector/ExternalStream.h"
#include "velox4j/eval/Evaluation.h"
#include "velox4j/init/Config.h"
#include "velox4j/query/Query.h"

namespace facebook::velox4j {

using namespace facebook::velox;

namespace {

// A utility function to ensure the initialization code body, i.e., f(),
// is called once and only once when initialize(...) is being invoked.
void init(const std::function<void()>& f) {
  static std::atomic<bool> initialized{false};
  bool expected = false;
  if (!initialized.compare_exchange_strong(expected, true)) {
    VELOX_FAIL("Velox4J was already initialized");
  }
  f();
}

void initForSpark() {
  FLAGS_velox_memory_leak_check_enabled = true;
  FLAGS_velox_memory_pool_capacity_transfer_across_tasks = true;
  FLAGS_velox_exception_user_stacktrace_enabled = true;
  FLAGS_velox_exception_system_stacktrace_enabled = true;
  filesystems::registerLocalFileSystem();
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  dwio::common::registerFileSinks();
  parquet::registerParquetReaderFactory();
  parquet::registerParquetWriterFactory();
  functions::sparksql::registerFunctions();
  aggregate::prestosql::registerAllAggregateFunctions(
      "",
      true /*registerCompanionFunctions*/,
      false /*onlyPrestoSignatures*/,
      true /*overwrite*/);
  functions::aggregate::sparksql::registerAggregateFunctions(
      "", true /*registerCompanionFunctions*/, true /*overwrite*/);
  window::prestosql::registerAllWindowFunctions();
  functions::window::sparksql::registerWindowFunctions("");

  ConfigArray::registerSerDe();
  ConnectorConfigArray::registerSerDe();
  Evaluation::registerSerDe();
  Query::registerSerDe();
  Type::registerSerDe();
  common::Filter::registerSerDe();
  connector::hive::HiveTableHandle::registerSerDe();
  connector::hive::LocationHandle::registerSerDe();
  connector::hive::HiveColumnHandle::registerSerDe();
  connector::hive::HiveConnectorSplit::registerSerDe();
  connector::hive::registerHivePartitionFunctionSerDe();
  connector::hive::HiveInsertTableHandle::registerSerDe();
  connector::hive::LocationHandle::registerSerDe();
  connector::hive::HiveSortingColumn::registerSerDe();
  connector::hive::HiveBucketProperty::registerSerDe();
  connector::hive::HiveInsertFileNameGenerator::registerSerDe();
  connector::registerConnector(std::make_shared<connector::hive::HiveConnector>(
      "connector-hive",
      std::make_shared<facebook::velox::config::ConfigBase>(
          std::unordered_map<std::string, std::string>()),
      nullptr));
  ExternalStreamConnectorSplit::registerSerDe();
  ExternalStreamTableHandle::registerSerDe();
  connector::registerConnector(std::make_shared<ExternalStreamConnector>(
      "connector-external-stream",
      std::make_shared<facebook::velox::config::ConfigBase>(
          std::unordered_map<std::string, std::string>())));
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  exec::registerPartitionFunctionSerDe();
}
} // namespace

void initialize(const std::shared_ptr<ConfigArray>& configArray) {
  init([&]() -> void {
    auto vConfig = std::make_shared<facebook::velox::config::ConfigBase>(
        configArray->toMap());
    auto preset = vConfig->get(VELOX4J_INIT_PRESET);
    switch (preset) {
      case SPARK:
        initForSpark();
        break;
      default:
        VELOX_FAIL("Unknown preset: {}", folly::to<std::string>(preset));
    }
  });
}
} // namespace facebook::velox4j
