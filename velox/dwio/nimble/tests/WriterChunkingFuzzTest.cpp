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

#include <folly/executors/CPUThreadPoolExecutor.h>
#include <folly/system/HardwareConcurrency.h>
#include <gtest/gtest.h>
#include <random>

#include "folly/Random.h"
#include "velox/common/memory/MemoryArbitrator.h"
#include "velox/common/memory/SharedArbitrator.h"
#include "velox/dwio/nimble/reader/BatchReader.h"
#include "velox/dwio/nimble/tests/WriterTestUtils.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/Writer.h"

namespace facebook {
DEFINE_uint32(
    writer_tests_seed,
    0,
    "If provided, this seed will be used when executing tests. "
    "Otherwise, a random seed will be used.");

class WriterChunkingFuzzTest : public ::testing::Test {
 protected:
  static void SetUpTestCase() {
    velox::memory::SharedArbitrator::registerFactory();
    velox::memory::MemoryManager::Options options;
    options.arbitratorKind = "SHARED";
    velox::memory::MemoryManager::testingSetInstance(options);
  }

  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool("default_root");
    leafPool_ = rootPool_->addLeafChild("default_leaf");
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
};

TEST_F(WriterChunkingFuzzTest, fuzzComplex) {
  auto type = velox::ROW(
      {{"array", velox::ARRAY(velox::VARCHAR())},
       {"dict_array", velox::ARRAY(velox::REAL())},
       {"map", velox::MAP(velox::INTEGER(), velox::DOUBLE())},
       {"row",
        velox::ROW({
            {"a", velox::REAL()},
            {"b", velox::INTEGER()},
        })},
       {"row",
        velox::ROW(
            {{"nested_row",
              velox::ROW(
                  {{"nested_nested_row", velox::ROW({{"a", velox::INTEGER()}})},
                   {"b", velox::INTEGER()}})}})},
       {"map",
        velox::MAP(velox::INTEGER(), velox::ROW({{"a", velox::INTEGER()}}))},
       {"nested",
        velox::ARRAY(
            velox::ROW({
                {"a", velox::INTEGER()},
                {"b", velox::MAP(velox::REAL(), velox::VARBINARY())},
            }))},
       {"nested_map_array1",
        velox::MAP(velox::INTEGER(), velox::ARRAY(velox::REAL()))},
       {"nested_map_array2",
        velox::MAP(velox::INTEGER(), velox::ARRAY(velox::INTEGER()))},
       {"dict_map", velox::MAP(velox::INTEGER(), velox::INTEGER())}});

  auto rowType = std::dynamic_pointer_cast<const velox::RowType>(type);
  const uint32_t seed = FLAGS_writer_tests_seed > 0 ? FLAGS_writer_tests_seed
                                                    : folly::Random::rand32();
  LOG(INFO) << "seed: " << seed;
  std::mt19937 rng{seed};
  for (bool disableSharedStringBuffers : {true, false}) {
    LOG(INFO) << "disableSharedStringBuffers: " << disableSharedStringBuffers;
    for (auto parallelismFactor : {0U, 1U, folly::available_concurrency()}) {
      std::shared_ptr<folly::CPUThreadPoolExecutor> executor;
      nimble::WriterOptions writerOptions;
      writerOptions.enableChunking = true;
      writerOptions.disableSharedStringBuffers = disableSharedStringBuffers;
      writerOptions.flushPolicyFactory =
          []() -> std::unique_ptr<nimble::FlushPolicy> {
        return std::make_unique<nimble::ChunkFlushPolicy>(
            nimble::ChunkFlushPolicyConfig{
                .writerMemoryHighThresholdBytes = 200 << 10,
                .writerMemoryLowThresholdBytes = 100 << 10,
                .targetStripeSizeBytes = 100 << 10,
                .estimatedCompressionFactor = 1.7,
            });
      };

      LOG(INFO) << "Parallelism Factor: " << parallelismFactor;
      writerOptions.dictionaryArrayColumns.insert("nested_map_array1");
      writerOptions.dictionaryArrayColumns.insert("nested_map_array2");
      writerOptions.dictionaryArrayColumns.insert("dict_array");
      writerOptions.deduplicatedMapColumns.insert("dict_map");

      if (parallelismFactor > 0) {
        executor =
            std::make_shared<folly::CPUThreadPoolExecutor>(parallelismFactor);
        writerOptions.encodingExecutor = folly::getKeepAliveToken(*executor);
      }

      const auto iterations = 20;
      // provide sufficient buffer between min and max chunk size thresholds
      constexpr uint64_t chunkThresholdBuffer =
          sizeof(std::string_view) + sizeof(bool);
      for (auto i = 0; i < iterations; ++i) {
        writerOptions.minStreamChunkRawSize =
            std::uniform_int_distribution<uint64_t>(10, 4096)(rng);
        writerOptions.maxStreamChunkRawSize =
            std::uniform_int_distribution<uint64_t>(
                writerOptions.minStreamChunkRawSize + chunkThresholdBuffer,
                8192)(rng);
        const auto batchSize =
            std::uniform_int_distribution<uint32_t>(10, 400)(rng);
        const auto batchCount = 5;

        std::string file;
        auto writeFile = std::make_unique<velox::InMemoryWriteFile>(&file);
        nimble::Writer writer(
            type, std::move(writeFile), *leafPool_.get(), writerOptions);
        const auto batches = generateBatches(
            type,
            /*batchCount=*/batchCount,
            /*size=*/batchSize,
            /*seed=*/seed,
            *leafPool_);

        for (const auto& batch : batches) {
          writer.write(batch);
        }
        writer.close();

        velox::InMemoryReadFile readFile(file);
        nimble::BatchReader reader(&readFile, *leafPool_);
        validateChunkSize(
            reader,
            writerOptions.minStreamChunkRawSize,
            writerOptions.maxStreamChunkRawSize);
      }
    }
  }
}

} // namespace facebook
