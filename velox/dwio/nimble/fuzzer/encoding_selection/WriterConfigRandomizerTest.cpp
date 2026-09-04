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

#include "velox/dwio/nimble/fuzzer/encoding_selection/WriterConfigRandomizer.h"

#include <gtest/gtest.h>

#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"
#include "velox/dwio/nimble/velox/NimbleConfig.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"

namespace facebook::nimble::fuzzer {
namespace {

bool isPowerOfTwo(uint64_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

TEST(WriterConfigRandomizerTest, deterministicAndValid) {
  EXPECT_EQ(randomizeWriterConfig(123), randomizeWriterConfig(123));

  for (uint64_t seed = 0; seed < 1'000; ++seed) {
    const auto config = randomizeWriterConfig(seed);
    EXPECT_GE(config.compressionAcceptRatio, 0.5);
    EXPECT_LT(config.compressionAcceptRatio, 1.0);
    EXPECT_GE(config.internalCompressionLevel, 1);
    EXPECT_LE(config.internalCompressionLevel, 9);
    EXPECT_TRUE(
        config.minStreamChunkRawSize == 0 ||
        isPowerOfTwo(config.minStreamChunkRawSize));
    EXPECT_LT(config.minStreamChunkRawSize, config.maxStreamChunkRawSize);
    EXPECT_TRUE(isPowerOfTwo(config.maxStreamChunkRawSize));
    EXPECT_GE(config.maxStreamChunkRawSize, uint64_t{1} << 10);
    EXPECT_LE(config.maxStreamChunkRawSize, uint64_t{1} << 21);
    EXPECT_EQ(
        config.maxStreamChunkRawSize, config.wideSchemaMaxStreamChunkRawSize);
    EXPECT_FALSE(
        config.flushMode == FuzzedFlushMode::kRawSize && config.enableChunking);
    EXPECT_FALSE(config.enableChunkIndex && !config.enableChunking);
    if (config.flushMode == FuzzedFlushMode::kPerBatch) {
      EXPECT_TRUE(config.enableChunking);
    }
    EXPECT_TRUE(isPowerOfTwo(config.blockBitPackingBlockSize));
    EXPECT_GE(config.blockBitPackingBlockSize, 32);
    EXPECT_LE(config.blockBitPackingBlockSize, 1024);
    EXPECT_TRUE(config.encodingSelectionSeed.has_value());
  }
}

TEST(WriterConfigRandomizerTest, encodingSelectionFrequency) {
  bool observedRandomSelection = false;
  bool observedManualSelection = false;
  for (uint64_t seed = 0; seed < 100; ++seed) {
    const auto config =
        randomizeWriterConfig(seed, {.randomEncodingSelectionOneIn = 2});
    observedRandomSelection |= config.encodingSelectionSeed.has_value();
    observedManualSelection |= !config.encodingSelectionSeed.has_value();
  }
  EXPECT_TRUE(observedRandomSelection);
  EXPECT_TRUE(observedManualSelection);

  const auto withoutRandomSelection =
      randomizeWriterConfig(123, {.randomEncodingSelectionOneIn = 0});
  EXPECT_FALSE(withoutRandomSelection.encodingSelectionSeed.has_value());
}

TEST(WriterConfigRandomizerTest, fixedSeedIsStable) {
  const auto config =
      randomizeWriterConfig(123, {.randomEncodingSelectionOneIn = 2});
  EXPECT_EQ(config.version, 2);
  EXPECT_EQ(
      config.debugString(),
      "version=2,seed=123,compression={acceptRatio:0.80707252,internalLevel:3},chunking={enabled:true,min:2048,max:4096,wideMax:4096,index:true,dedup:false},encoding={blockSize:256,selectionSeed:manual},flush={mode:per_batch,rawSize:4096,seed:2122292633536747636}");
}

TEST(WriterConfigRandomizerTest, appliesScalarOptions) {
  const auto config = randomizeWriterConfig(456);
  WriterOptions options;
  config.applyTo(options);
  config.applyEncodingSelectionTo(options);

  EXPECT_FLOAT_EQ(
      options.compressionOptions.compressionAcceptRatio,
      config.compressionAcceptRatio);
  EXPECT_EQ(options.enableChunking, config.enableChunking);
  EXPECT_EQ(options.minStreamChunkRawSize, config.minStreamChunkRawSize);
  EXPECT_EQ(options.maxStreamChunkRawSize, config.maxStreamChunkRawSize);
  EXPECT_EQ(
      options.wideSchemaMaxStreamChunkRawSize,
      config.wideSchemaMaxStreamChunkRawSize);
  EXPECT_EQ(options.enableChunkIndex, config.enableChunkIndex);
  EXPECT_EQ(
      options.enableStreamDeduplication, config.enableStreamDeduplication);
  EXPECT_EQ(options.blockBitPackingBlockSize, config.blockBitPackingBlockSize);
  auto policy =
      options.encodingSelectionPolicyCreator(TypeTraits<int32_t>::dataType);
  EXPECT_NE(
      dynamic_cast<testing::RandomEncodingSelectionPolicy<int32_t>*>(
          policy.get()),
      nullptr);
}

TEST(WriterConfigRandomizerTest, serializesEveryOption) {
  const auto config = randomizeWriterConfig(789);
  const auto params = config.toSerdeParams();

  EXPECT_TRUE(params.contains(std::string(Config::ENABLE_CHUNKING.key)));
  EXPECT_TRUE(params.contains(std::string(Config::ENABLE_CHUNK_INDEX.key)));
  EXPECT_TRUE(
      params.contains(std::string(Config::ENABLE_STREAM_DEDUPLICATION.key)));
  EXPECT_TRUE(
      params.contains(std::string(Config::FLUSH_POLICY_CONFIG.key)) ||
      params.contains(std::string(Config::RAW_STRIPE_SIZE.key)));
  EXPECT_EQ(
      params.at(std::string(Config::ENCODING_SELECTION_CONFIG.key)),
      config.encodingSelectionConfig());
}

TEST(WriterConfigRandomizerTest, installsEveryFlushMode) {
  auto config = randomizeWriterConfig(999);
  WriterOptions options;

  config.flushMode = FuzzedFlushMode::kRawSize;
  config.stripeRawSize = 4'096;
  config.applyTo(options);
  auto policy = options.flushPolicyFactory();
  if (policy == nullptr) {
    FAIL() << "Raw-size flush factory returned null";
  }
  EXPECT_NE(dynamic_cast<StripeRawSizeFlushPolicy*>(policy.get()), nullptr);
  EXPECT_FALSE(policy->shouldFlush({4'095, 0, 0}));
  EXPECT_TRUE(policy->shouldFlush({4'096, 0, 0}));

  config.flushMode = FuzzedFlushMode::kPerBatch;
  config.applyTo(options);
  policy = options.flushPolicyFactory();
  if (policy == nullptr) {
    FAIL() << "Per-batch flush factory returned null";
  }
  EXPECT_NE(dynamic_cast<TestFlushPolicy*>(policy.get()), nullptr);
  EXPECT_TRUE(policy->shouldChunk({0, 0, 0}));

  config.flushMode = FuzzedFlushMode::kRandom;
  config.applyTo(options);
  policy = options.flushPolicyFactory();
  if (policy == nullptr) {
    FAIL() << "Random flush factory returned null";
  }
  EXPECT_NE(dynamic_cast<TestFlushPolicy*>(policy.get()), nullptr);
}

} // namespace
} // namespace facebook::nimble::fuzzer
