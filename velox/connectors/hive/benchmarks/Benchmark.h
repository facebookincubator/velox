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

#include <folly/Benchmark.h>
#include "velox/connectors/hive/HiveConfig.h"
#include "velox/connectors/hive/HiveConnectorUtil.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"
#include "velox/exec/tests/utils/TempDirectoryPath.h"
#include "velox/type/Type.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::connector::hive::benchmark {

struct PartitionField {
  int32_t id;
  connector::hive::iceberg::TransformType type;
  std::optional<int32_t> parameter;
};

struct BenchmarkStats {
  std::chrono::microseconds duration;
  int64_t memoryUsedMB;
  int64_t peakMemoryMB;
};

class Benchmark {
 public:
  Benchmark();
  ~Benchmark();

 protected:
  void setUp();
  void tearDown();
  std::vector<RowVectorPtr> createTestData(
      const TypePtr& dataType,
      int32_t numBatches,
      vector_size_t rowsPerBatch,
      std::optional<int32_t> range);

  std::shared_ptr<exec::test::TempDirectoryPath> testDir_;
  RowTypePtr rowType_;
  std::shared_ptr<memory::MemoryPool> rootPool_;
  std::shared_ptr<memory::MemoryPool> opPool_;
  std::shared_ptr<memory::MemoryPool> connectorPool_;
  std::shared_ptr<config::ConfigBase> connectorSessionProperties_;
  std::shared_ptr<connector::hive::HiveConfig> connectorConfig_;
  std::unique_ptr<connector::ConnectorQueryCtx> connectorQueryCtx_;
  std::unique_ptr<VectorFuzzer> fuzzer_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
};

} // namespace facebook::velox::connector::hive::benchmark
