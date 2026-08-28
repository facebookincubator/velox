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
#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <array>
#include <memory>
#include <string>
#include <string_view>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/FlushPolicyFactory.h"

namespace facebook::nimble {

class ChunkFlushPolicyTest : public ::testing::Test {};

TEST_F(ChunkFlushPolicyTest, initialNoMemoryPressure) {
  ChunkFlushPolicy policy(
      ChunkFlushPolicyConfig{
          .writerMemoryHighThresholdBytes = 1000,
          .writerMemoryLowThresholdBytes = 500,
          .targetStripeSizeBytes = 800,
          .estimatedCompressionFactor = 1.5});

  StripeProgress progress{
      .stripeRawSize = 100,
      .stripeEncodedSize = 50,
      .stripeEncodedLogicalSize = 80};

  // Total Memory: 100 + 50 = 150 < 1000 (writerMemoryHighThresholdBytes)
  EXPECT_FALSE(policy.shouldChunk(progress));

  // shouldFlush checks two conditions:
  // 1. Memory pressure check: 100 + 50 = 150 < 1000 (passes, no flush)
  // 2. Stripe size check (only if condition 1 passes):
  //    compressionFactor = stripeEncodedLogicalSize / stripeEncodedSize
  //                      = 80 / 50 = 1.6
  //    expectedEncodedStripeSize = stripeEncodedSize + stripeRawSize /
  //                                max(compressionFactor, 1.0)
  //                              = 50 + 100 / max(1.6, 1.0)
  //                              = 50 + 100 / 1.6
  //                              = 50 + 62.5 = 112.5 < 800
  //                              (targetStripeSizeBytes)
  EXPECT_FALSE(policy.shouldFlush(progress));
  ;
}

TEST_F(ChunkFlushPolicyTest, memoryPressureChunkingLifecycle) {
  ChunkFlushPolicy policy(
      ChunkFlushPolicyConfig{
          .writerMemoryHighThresholdBytes = 1000,
          .writerMemoryLowThresholdBytes = 500,
          .targetStripeSizeBytes = 800,
          .estimatedCompressionFactor = 1.5});

  // Start chunking when memory exceeds high threshold
  StripeProgress progress1{
      .stripeRawSize = 600,
      .stripeEncodedSize = 500,
      .stripeEncodedLogicalSize = 400};
  // Total Memory: 600 + 500 = 1100 > 1000 (writerMemoryHighThresholdBytes)
  EXPECT_TRUE(policy.shouldChunk(progress1));

  // Continue chunking with decreasing raw size but still above low threshold
  // Raw size decreased (600 → 550) and memory still above low threshold
  StripeProgress progress2{
      .stripeRawSize = 550,
      .stripeEncodedSize = 520,
      .stripeEncodedLogicalSize = 400};
  // Total Memory: 550 + 520 = 1070 > 500 (writerMemoryLowThresholdBytes)
  EXPECT_TRUE(policy.shouldChunk(progress2));

  // Continue chunking as raw size decreases further
  StripeProgress progress3{
      .stripeRawSize = 300,
      .stripeEncodedSize = 300,
      .stripeEncodedLogicalSize = 400};
  // Total memory: 300 + 300 = 600 > 500
  EXPECT_TRUE(policy.shouldChunk(progress3));

  // Memory drops below low threshold - stop chunking
  StripeProgress progress4{
      .stripeRawSize = 200,
      .stripeEncodedSize = 200,
      .stripeEncodedLogicalSize = 400};
  // Total Memory: 200 + 200 = 400 < 500 (writerMemoryLowThresholdBytes)
  EXPECT_FALSE(policy.shouldChunk(progress4));
}

TEST_F(ChunkFlushPolicyTest, memoryPressureTriggersFlush) {
  ChunkFlushPolicy policy(
      ChunkFlushPolicyConfig{
          .writerMemoryHighThresholdBytes = 1000,
          .writerMemoryLowThresholdBytes = 500,
          .targetStripeSizeBytes = 800,
          .estimatedCompressionFactor = 1.5});

  StripeProgress progress{
      .stripeRawSize = 600,
      .stripeEncodedSize = 500,
      .stripeEncodedLogicalSize = 400};
  // Total Memory: 600 + 500 = 1100 > 1000 (writerMemoryHighThresholdBytes)
  EXPECT_TRUE(policy.shouldChunk(progress));

  // Total Memory: 600 + 500 = 1100 > 1000 (writerMemoryHighThresholdBytes)
  EXPECT_TRUE(policy.shouldFlush(progress));
}

TEST_F(ChunkFlushPolicyTest, stripeSizeTriggersFlush) {
  ChunkFlushPolicy policy(
      ChunkFlushPolicyConfig{
          .writerMemoryHighThresholdBytes = 2000,
          .writerMemoryLowThresholdBytes = 500,
          .targetStripeSizeBytes = 800,
          .estimatedCompressionFactor = 1.5});

  StripeProgress progress{
      .stripeRawSize = 1000,
      .stripeEncodedSize = 400,
      .stripeEncodedLogicalSize = 600};

  // Expected Encoded Memory calculation:
  // compressionFactor = stripeEncodedLogicalSize / stripeEncodedSize
  //                   = 600 / 400 = 1.5
  // expectedEncodedStripeSize = stripeEncodedSize + stripeRawSize /
  //                             max(compressionFactor, 1.0)
  //                           = 400 + 1000 / max(1.5, 1.0)
  //                           = 400 + 1000 / 1.5
  //                           = 400 + 666.67 = 1066.67 > 800
  EXPECT_TRUE(policy.shouldFlush(progress));
  ;
}

TEST_F(ChunkFlushPolicyTest, zeroEncodedSize) {
  ChunkFlushPolicy policy(
      ChunkFlushPolicyConfig{
          .writerMemoryHighThresholdBytes = 1000,
          .writerMemoryLowThresholdBytes = 500,
          .targetStripeSizeBytes = 800,
          .estimatedCompressionFactor = 2.0});

  StripeProgress progress{
      .stripeRawSize = 1200,
      .stripeEncodedSize = 0,
      .stripeEncodedLogicalSize = 0};

  // Memory pressure check:
  // tripeRawSize + stripeEncodedSize = 1200 + 0 = 1200 > 1000
  // This triggers chunking due to memory pressure
  EXPECT_TRUE(policy.shouldChunk(progress));
  ;

  // Flush check: Memory pressure (1200 > 1000) triggers stripe flush
  EXPECT_TRUE(policy.shouldFlush(progress));
  ;
}

TEST(FlushPolicyTest, stripeRawSizeFlushPolicyTest) {
  StripeRawSizeFlushPolicy policy(/*stripeRawSize=*/1000);

  StripeProgress progress{
      .stripeRawSize = 1200, // Exceeds threshold
      .stripeEncodedSize = 600,
      .stripeEncodedLogicalSize = 800};

  EXPECT_TRUE(policy.shouldFlush(progress));
  ;

  StripeProgress progress2{
      .stripeRawSize = 800, // Below threshold
      .stripeEncodedSize = 600,
      .stripeEncodedLogicalSize = 700};

  EXPECT_FALSE(policy.shouldFlush(progress2));
}

TEST(FlushPolicyTest, lambdaFlushPolicyTest) {
  LambdaFlushPolicy policy(
      /*flushLambda_=*/
      [](const StripeProgress& progress) {
        return progress.stripeRawSize > 1000;
      },
      /*chunkLambda_=*/
      [](const StripeProgress& progress) {
        return progress.stripeEncodedSize > 500;
      });

  // Test None case
  StripeProgress noneProgress{
      .stripeRawSize = 300,
      .stripeEncodedSize = 200,
      .stripeEncodedLogicalSize = 250};
  EXPECT_FALSE(policy.shouldFlush(noneProgress));
  EXPECT_FALSE(policy.shouldChunk(noneProgress));

  // Test Chunk case
  StripeProgress chunkProgress{
      .stripeRawSize = 500,
      .stripeEncodedSize = 600,
      .stripeEncodedLogicalSize = 450};
  EXPECT_TRUE(policy.shouldChunk(chunkProgress));

  // Test Stripe case
  StripeProgress stripeProgress{
      .stripeRawSize = 1200,
      .stripeEncodedSize = 400,
      .stripeEncodedLogicalSize = 800};
  EXPECT_TRUE(policy.shouldFlush(stripeProgress));
}

TEST(FlushPolicyTest, lambdaFlushPolicyDefaultLambdas) {
  LambdaFlushPolicy policy;

  StripeProgress progress{
      .stripeRawSize = 1200,
      .stripeEncodedSize = 600,
      .stripeEncodedLogicalSize = 800};

  EXPECT_FALSE(policy.shouldFlush(progress));
  ;
  EXPECT_FALSE(policy.shouldChunk(progress));
}

class TestFlushPolicyTest : public ::testing::Test {};

// Same seed reproduces the decision sequence (so a failing run replays from its
// logged seed); distinct seeds diverge (the seed actually drives the rng).
TEST_F(TestFlushPolicyTest, seedBasics) {
  // A non-trivial progress so shouldFlush exercises its full computation.
  const StripeProgress progress{
      .stripeRawSize = 1000,
      .stripeEncodedSize = 500,
      .stripeEncodedLogicalSize = 800};

  // Same seed -> identical decisions.
  TestFlushPolicyFactory factoryA{TestFlushPolicy::Options{.seed = 42}};
  TestFlushPolicyFactory factoryB{TestFlushPolicy::Options{.seed = 42}};
  auto policyA = factoryA();
  auto policyB = factoryB();
  for (int i = 0; i < 1000; ++i) {
    EXPECT_EQ(policyA->shouldChunk(progress), policyB->shouldChunk(progress));
    EXPECT_EQ(policyA->shouldFlush(progress), policyB->shouldFlush(progress));
  }

  // Distinct seeds -> at least one differing decision.
  TestFlushPolicyFactory factoryC{TestFlushPolicy::Options{.seed = 1}};
  TestFlushPolicyFactory factoryD{TestFlushPolicy::Options{.seed = 2}};
  auto policyC = factoryC();
  auto policyD = factoryD();
  bool anyDifferent = false;
  for (int i = 0; i < 1000; ++i) {
    if (policyC->shouldChunk(progress) != policyD->shouldChunk(progress)) {
      anyDifferent = true;
    }
  }
  EXPECT_TRUE(anyDifferent);
}

// A policy recreated per call from a factory keeps drawing from the factory's
// single shared State, so its decision stream matches one persistent policy
// from an equivalent same-seed factory. This is what makes the fuzzer's
// boundaries vary across the writer's per-write() policy recreation.
TEST_F(TestFlushPolicyTest, sharedStateContinuesSequence) {
  const StripeProgress progress{
      .stripeRawSize = 0,
      .stripeEncodedSize = 0,
      .stripeEncodedLogicalSize = 0};

  constexpr int kDecisions = 8;
  TestFlushPolicyFactory persistentFactory{
      TestFlushPolicy::Options{.seed = 99}};
  auto persistent = persistentFactory();
  std::array<bool, kDecisions> persistentDecisions{};
  for (int i = 0; i < kDecisions; ++i) {
    persistentDecisions[i] = persistent->shouldChunk(progress);
  }

  // Same seed, but a fresh policy recreated per call (mimics the writer's
  // per-write() recreation) -- must reproduce the persistent stream.
  TestFlushPolicyFactory recreatingFactory{
      TestFlushPolicy::Options{.seed = 99}};
  std::array<bool, kDecisions> recreatedDecisions{};
  for (int i = 0; i < kDecisions; ++i) {
    recreatedDecisions[i] = recreatingFactory()->shouldChunk(progress);
  }
  EXPECT_EQ(persistentDecisions, recreatedDecisions);
}

// The size guard must force a flush regardless of the random draw or seed when
// the predicted encoded stripe size already exceeds maxStripePhysicalSize.
TEST_F(TestFlushPolicyTest, flushSizeGuard) {
  // stripeEncodedSize alone is >= the default 128MB max, so
  // expectedEncodedStripeSize is too, independent of the compression heuristic.
  const StripeProgress oversized{
      .stripeRawSize = 0,
      .stripeEncodedSize = 128 * 1024L * 1024L,
      .stripeEncodedLogicalSize = 0};

  for (uint64_t seed = 0; seed < 50; ++seed) {
    TestFlushPolicyFactory factory{TestFlushPolicy::Options{.seed = seed}};
    auto policy = factory();
    for (int i = 0; i < 100; ++i) {
      EXPECT_TRUE(policy->shouldFlush(oversized));
    }
  }
}

// The default flushChunkProbability is 1/3; over many calls the observed rate
// should land near it.
TEST_F(TestFlushPolicyTest, defaultChunkRate) {
  TestFlushPolicyFactory factory{TestFlushPolicy::Options{.seed = 7}};
  auto policy = factory();

  const StripeProgress progress{
      .stripeRawSize = 0,
      .stripeEncodedSize = 0,
      .stripeEncodedLogicalSize = 0};

  constexpr int kCalls = 100000;
  int chunkCount = 0;
  for (int i = 0; i < kCalls; ++i) {
    if (policy->shouldChunk(progress)) {
      ++chunkCount;
    }
  }

  const double rate = static_cast<double>(chunkCount) / kCalls;
  // Band around the default 1/3 probability.
  EXPECT_GE(rate, 0.30);
  EXPECT_LE(rate, 0.36);
}

// The default flushStripeProbability is 1/20 when the size guard never trips;
// with a tiny progress the predicted size stays far below the physical max, so
// only the random draw can flush.
TEST_F(TestFlushPolicyTest, defaultFlushRate) {
  TestFlushPolicyFactory factory{TestFlushPolicy::Options{.seed = 7}};
  auto policy = factory();

  // All sizes tiny, so expectedEncodedStripeSize is far below 128MB.
  const StripeProgress progress{
      .stripeRawSize = 1,
      .stripeEncodedSize = 1,
      .stripeEncodedLogicalSize = 1};

  constexpr int kCalls = 100000;
  int flushCount = 0;
  for (int i = 0; i < kCalls; ++i) {
    if (policy->shouldFlush(progress)) {
      ++flushCount;
    }
  }

  const double rate = static_cast<double>(flushCount) / kCalls;
  // Band around the default 1/20 probability.
  EXPECT_GE(rate, 0.03);
  EXPECT_LE(rate, 0.07);
}

// A custom flushChunkProbability is honored: the observed chunk rate tracks it.
TEST_F(TestFlushPolicyTest, customChunkProbability) {
  TestFlushPolicyFactory factory{
      TestFlushPolicy::Options{.seed = 11, .flushChunkProbability = 0.8}};
  auto policy = factory();

  const StripeProgress progress{
      .stripeRawSize = 0,
      .stripeEncodedSize = 0,
      .stripeEncodedLogicalSize = 0};

  constexpr int kCalls = 100000;
  int chunkCount = 0;
  for (int i = 0; i < kCalls; ++i) {
    if (policy->shouldChunk(progress)) {
      ++chunkCount;
    }
  }

  const double rate = static_cast<double>(chunkCount) / kCalls;
  // Band around the configured 0.8 probability.
  EXPECT_GE(rate, 0.78);
  EXPECT_LE(rate, 0.82);
}

// flushChunkProbability 1.0 chunks on every write (the test_per_batch behavior
// the factory pins), while a configurable stripe probability still drives
// flushes.
TEST_F(TestFlushPolicyTest, alwaysChunksWhenProbabilityOne) {
  TestFlushPolicyFactory factory{TestFlushPolicy::Options{
      .seed = 5, .flushChunkProbability = 1.0, .flushStripeProbability = 0.5}};
  auto policy = factory();
  const StripeProgress progress{
      .stripeRawSize = 1,
      .stripeEncodedSize = 1,
      .stripeEncodedLogicalSize = 1};
  int flushCount = 0;
  for (int i = 0; i < 1000; ++i) {
    EXPECT_TRUE(policy->shouldChunk(progress));
    if (policy->shouldFlush(progress)) {
      ++flushCount;
    }
  }
  // The configurable stripe probability is honored (not forced off).
  EXPECT_GT(flushCount, 0);
}

// The constructor sanity-checks every tuning value as an internal error (a
// config coming through FlushPolicyFactoryFactory is already user-validated).
// The factory constructs the policy, so a bad Options surfaces when invoked.
TEST_F(TestFlushPolicyTest, ctorInvalidArgs) {
  // maxStripePhysicalSize must be > 0.
  EXPECT_THROW(
      TestFlushPolicyFactory{
          TestFlushPolicy::Options{.maxStripePhysicalSize = 0}}(),
      NimbleInternalError);
  // estimatedCompressionFactor must be >= 1.0.
  EXPECT_THROW(
      TestFlushPolicyFactory{
          TestFlushPolicy::Options{.estimatedCompressionFactor = 0.5}}(),
      NimbleInternalError);
  // flushChunkProbability must be within [0, 1].
  EXPECT_THROW(
      TestFlushPolicyFactory{
          TestFlushPolicy::Options{.flushChunkProbability = -0.1}}(),
      NimbleInternalError);
  EXPECT_THROW(
      TestFlushPolicyFactory{
          TestFlushPolicy::Options{.flushChunkProbability = 1.5}}(),
      NimbleInternalError);
  // flushStripeProbability must be within [0, 1].
  EXPECT_THROW(
      TestFlushPolicyFactory{
          TestFlushPolicy::Options{.flushStripeProbability = -0.1}}(),
      NimbleInternalError);
  EXPECT_THROW(
      TestFlushPolicyFactory{
          TestFlushPolicy::Options{.flushStripeProbability = 1.5}}(),
      NimbleInternalError);
  // The defaults and explicit valid values construct cleanly.
  EXPECT_NO_THROW(TestFlushPolicyFactory{TestFlushPolicy::Options{}}());
  EXPECT_NO_THROW((TestFlushPolicyFactory{TestFlushPolicy::Options{
      .flushChunkProbability = 0.0,
      .flushStripeProbability = 1.0,
      .maxStripePhysicalSize = 1024,
      .estimatedCompressionFactor = 1.0}}()));
}

// A custom maxStripePhysicalSize is honored: a small max makes the size guard
// trip (forcing a flush) regardless of the random draw.
TEST_F(TestFlushPolicyTest, customFlushSizeGuard) {
  TestFlushPolicyFactory factory{
      TestFlushPolicy::Options{.seed = 3, .maxStripePhysicalSize = 1024}};
  auto policy = factory();

  // Encoded size alone already exceeds the 1024-byte max.
  const StripeProgress overMax{
      .stripeRawSize = 0,
      .stripeEncodedSize = 2048,
      .stripeEncodedLogicalSize = 0};

  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(policy->shouldFlush(overMax));
  }
}

class FlushPolicyFactoryFactoryTest : public ::testing::Test {
 protected:
  // Returns the factory for a config string, asserting one was produced. The
  // factory owns any shared rng state, so callers must keep it alive for as
  // long as any policy it produces -- hence we hand back the factory, not a
  // policy (a returned policy would dangle once the factory dropped).
  FlushPolicyFactory buildFactory(std::string_view configStr) {
    auto factory = FlushPolicyFactoryFactory::create(configStr);
    EXPECT_TRUE(factory.has_value());
    return factory.value_or(FlushPolicyFactory{});
  }
};

// An absent type (empty string, or tuning with no "type" key) keeps the
// production policy.
TEST_F(FlushPolicyFactoryFactoryTest, absentTypeReturnsNullopt) {
  EXPECT_FALSE(FlushPolicyFactoryFactory::create("").has_value());
  // A spec with tuning but no type keeps the production policy.
  EXPECT_FALSE(FlushPolicyFactoryFactory::create("seed:5").has_value());
}

// An unknown type -- including the former "default" -- is a user error.
TEST_F(FlushPolicyFactoryFactoryTest, unknownTypeThrows) {
  EXPECT_THROW(
      FlushPolicyFactoryFactory::create("type:bogus"), NimbleUserError);
  EXPECT_THROW(
      FlushPolicyFactoryFactory::create("type:default"), NimbleUserError);
}

// "test_random" builds a TestFlushPolicy and honors the parsed tuning: chunk
// probability 1.0 always chunks, and the 1KB max trips the size guard while a
// small stripe stays under it.
TEST_F(FlushPolicyFactoryFactoryTest, testRandomBuildsPolicyAndParsesParams) {
  auto factory = buildFactory(
      "type:test_random,flush_chunk_probability:1.0,flush_stripe_probability:0.0,"
      "max_stripe_physical_size:1KB,seed:5");
  ASSERT_NE(factory, nullptr);
  auto policy = factory();
  ASSERT_NE(policy, nullptr);
  EXPECT_NE(dynamic_cast<TestFlushPolicy*>(policy.get()), nullptr);

  const StripeProgress small{
      .stripeRawSize = 0,
      .stripeEncodedSize = 0,
      .stripeEncodedLogicalSize = 0};
  EXPECT_TRUE(policy->shouldChunk(small));
  EXPECT_TRUE(policy->shouldChunk(small));

  const StripeProgress overMax{
      .stripeRawSize = 0,
      .stripeEncodedSize = 2048,
      .stripeEncodedLogicalSize = 0};
  EXPECT_TRUE(policy->shouldFlush(overMax));
  EXPECT_FALSE(policy->shouldFlush(small));
}

// "test_per_batch" pins flush_chunk_probability to 1.0 (always chunks),
// overriding any value in the spec.
TEST_F(FlushPolicyFactoryFactoryTest, testPerBatchPinsChunkProbability) {
  const StripeProgress small{
      .stripeRawSize = 1,
      .stripeEncodedSize = 1,
      .stripeEncodedLogicalSize = 1};

  auto factory = buildFactory("type:test_per_batch,seed:5");
  ASSERT_NE(factory, nullptr);
  auto policy = factory();
  ASSERT_NE(policy, nullptr);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(policy->shouldChunk(small));
  }

  // A flush_chunk_probability in the spec is overridden to 1.0, not honored.
  auto overriddenFactory =
      buildFactory("type:test_per_batch,flush_chunk_probability:0.5");
  ASSERT_NE(overriddenFactory, nullptr);
  auto overridden = overriddenFactory();
  ASSERT_NE(overridden, nullptr);
  for (int i = 0; i < 100; ++i) {
    EXPECT_TRUE(overridden->shouldChunk(small));
  }
}

// A malformed entry, unknown key, unparseable value, or out-of-range value is a
// NimbleUserError (not a folly::ConversionError or the ctor's internal error).
TEST_F(FlushPolicyFactoryFactoryTest, invalidConfigThrows) {
  const auto create = [](const std::string& spec) {
    return FlushPolicyFactoryFactory::create("type:test_random," + spec);
  };
  EXPECT_THROW(create("no_colon_here"), NimbleUserError);
  EXPECT_THROW(create("unknown_key:1"), NimbleUserError);
  EXPECT_THROW(create("seed:abc"), NimbleUserError);
  EXPECT_THROW(create("max_stripe_physical_size:notasize"), NimbleUserError);
  EXPECT_THROW(create("flush_chunk_probability:2.0"), NimbleUserError);
  EXPECT_THROW(create("max_stripe_physical_size:0B"), NimbleUserError);
}

// The factory hands every policy the same shared state, so recreating the
// policy on each call (as Writer does via flushPolicyFactory() per
// write()) keeps advancing one rng rather than restarting it. A
// recreate-per-call decision stream must therefore match a single persistent
// policy from an equivalent factory -- i.e. the factory neutralizes the
// writer's per-write() recreation.
TEST_F(FlushPolicyFactoryFactoryTest, recreatedPoliciesShareRng) {
  const StripeProgress progress{
      .stripeRawSize = 0,
      .stripeEncodedSize = 0,
      .stripeEncodedLogicalSize = 0};

  constexpr int kDecisions = 8;
  auto referenceFactory = buildFactory("type:test_random,seed:7");
  ASSERT_NE(referenceFactory, nullptr);
  auto reference = referenceFactory();
  std::array<bool, kDecisions> referenceDecisions{};
  for (int i = 0; i < kDecisions; ++i) {
    referenceDecisions[i] = reference->shouldChunk(progress);
  }

  // A separate factory with the same seed, but a fresh policy per call.
  auto recreatingFactory = buildFactory("type:test_random,seed:7");
  ASSERT_NE(recreatingFactory, nullptr);
  std::array<bool, kDecisions> recreatedDecisions{};
  for (int i = 0; i < kDecisions; ++i) {
    recreatedDecisions[i] = recreatingFactory()->shouldChunk(progress);
  }
  EXPECT_EQ(referenceDecisions, recreatedDecisions);
}

} // namespace facebook::nimble
