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

#include "velox/connectors/hive/benchmarks/Benchmark.h"

#ifdef VELOX_ENABLE_PARQUET
#include "velox/dwio/parquet/RegisterParquetWriter.h"
#endif

#include "velox/common/fuzzer/Utils.h"
#include "velox/type/Timestamp.h"

namespace facebook::velox::connector::hive::benchmark {

namespace {

constexpr int32_t kMaxValues = 128;

class LimitedUniqueGenerator : public AbstractInputGenerator {
 public:
  LimitedUniqueGenerator(const TypePtr& targetType, int maxUniqueValues)
      : AbstractInputGenerator(0, targetType, nullptr, 0.0),
        maxUniqueValues_(maxUniqueValues) {}

  variant generate() override {
    auto uniqueValue =
        std::uniform_int_distribution<int32_t>(1, maxUniqueValues_)(rng_);
    switch (type_->kind()) {
      case TypeKind::INTEGER:
        if (type_->isDate()) {
          const int32_t baseDay = 18'000;
          const int32_t stepDays = 1;
          const int32_t days = baseDay + (uniqueValue - 1) * stepDays;
          return variant(static_cast<int32_t>(days));
        }
        return variant(static_cast<int32_t>(uniqueValue));
      case TypeKind::BIGINT:
        return variant(static_cast<int64_t>(uniqueValue));
      case TypeKind::TIMESTAMP: {
        const int64_t baseMillis = 1'700'000'000'000;
        const int64_t stepMillis = 1'000; // 1-second increments.
        const int64_t millis = baseMillis + (uniqueValue - 1) * stepMillis;
        return variant(Timestamp::fromMillisNoError(millis));
      }
      case TypeKind::SMALLINT:
        return variant(static_cast<int16_t>(uniqueValue));
      case TypeKind::VARCHAR:
        return variant(fmt::format("string_0000_1111_2222_{}", uniqueValue));
      case TypeKind::VARBINARY: {
        std::string v(fmt::format("binary_0000_1111_2222_{}", uniqueValue));
        return variant::binary(v);
      }
      case TypeKind::BOOLEAN:
        return variant((uniqueValue % 2) == 0);
      case TypeKind::HUGEINT:
        return variant(static_cast<int128_t>(uniqueValue));
      case TypeKind::TINYINT:
        return variant(static_cast<int8_t>(uniqueValue));
      default:
        return variant(static_cast<int32_t>(uniqueValue));
    }
  }

 private:
  int maxUniqueValues_;
};

} // namespace

Benchmark::Benchmark() {
  setUp();
}

Benchmark::~Benchmark() {
  tearDown();
}

void Benchmark::setUp() {
  filesystems::registerLocalFileSystem();
  parquet::registerParquetWriterFactory();
  dwio::common::registerFileSinks();
  Type::registerSerDe();

  testDir_ = exec::test::TempDirectoryPath::create();
  rootPool_ = memory::memoryManager()->addRootPool("Benchmark", 1L << 30);
  opPool_ = rootPool_->addLeafChild("operator");
  connectorPool_ = rootPool_->addAggregateChild("connector");
  connectorSessionProperties_ = std::make_shared<config::ConfigBase>(
      std::unordered_map<std::string, std::string>(), true);
  connectorConfig_ = std::make_shared<connector::hive::HiveConfig>(
      std::make_shared<config::ConfigBase>(
          std::unordered_map<std::string, std::string>()));
  connectorQueryCtx_ = std::make_unique<connector::ConnectorQueryCtx>(
      opPool_.get(),
      connectorPool_.get(),
      connectorSessionProperties_.get(),
      nullptr,
      common::PrefixSortConfig(),
      nullptr,
      nullptr,
      "queryBenchmark",
      "taskBenchmark",
      "planNodeBenchmark",
      0,
      "");

  VectorFuzzer::Options fuzzerOptions;
  fuzzerOptions.vectorSize = 100;
  fuzzerOptions.nullRatio = 0.0;
  fuzzerOptions.stringVariableLength = true;
  fuzzer_ = std::make_unique<VectorFuzzer>(fuzzerOptions, opPool_.get());
  vectorMaker_ = std::make_unique<velox::test::VectorMaker>(opPool_.get());
}

void Benchmark::tearDown() {
  vectorMaker_.reset();
  fuzzer_.reset();
  connectorQueryCtx_.reset();
  connectorPool_.reset();
  opPool_.reset();
  rootPool_.reset();
}

std::vector<RowVectorPtr> Benchmark::createTestData(
    const TypePtr& dataType,
    int32_t numBatches,
    vector_size_t rowsPerBatch,
    std::optional<int32_t> range) {
  VectorFuzzer::Options fuzzerOptions;
  fuzzerOptions.vectorSize = rowsPerBatch;
  fuzzerOptions.nullRatio = 0.0;
  fuzzerOptions.stringVariableLength = true;
  auto fuzzer = std::make_unique<VectorFuzzer>(fuzzerOptions, opPool_.get());

  std::vector<RowVectorPtr> batches;
  batches.reserve(numBatches);

  std::shared_ptr<LimitedUniqueGenerator> generator = nullptr;
  if (range.has_value()) {
    generator = make_shared<LimitedUniqueGenerator>(dataType, range.value());
  }
  std::vector<AbstractInputGeneratorPtr> generators = {
      generator,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      nullptr};
  for (auto i = 0; i < numBatches; ++i) {
    auto batch = fuzzer->fuzzRow(rowType_, rowsPerBatch, false, generators);
    batches.push_back(batch);
  }
  return batches;
}

} // namespace facebook::velox::connector::hive::benchmark
