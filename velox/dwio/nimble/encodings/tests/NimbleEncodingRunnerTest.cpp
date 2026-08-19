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

#include "velox/dwio/nimble/encodings/benchmarks/NimbleEncodingRunner.h"

#include <array>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <folly/json/json.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"

namespace facebook::nimble::benchmarks {
namespace {

class NimbleEncodingRunnerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    velox::memory::MemoryManager::testingSetInstance({});
  }

  void SetUp() override {
    rootPool_ =
        velox::memory::memoryManager()->addRootPool("NimbleEncodingRunnerTest");
    pool_ = rootPool_->addLeafChild("runner");
  }

  EncodingRunnerConfig config(std::string taskId) const {
    return EncodingRunnerConfig{
        .taskId = std::move(taskId),
        .rowCount = 257,
        .seed = 0xC0FFEE,
        .warmups = 0,
        .samples = 5,
        .minSampleTimeMicros = 0,
        .innerIterations = 1,
    };
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> pool_;
};

TEST_F(NimbleEncodingRunnerTest, sameSeedProducesIdenticalArtifacts) {
  const auto runnerConfig = config("nimble.dictionary.encode.v1");

  const auto first = runEncodingBenchmark(runnerConfig, *pool_);
  const auto second = runEncodingBenchmark(runnerConfig, *pool_);

  EXPECT_EQ(first.encodedArtifact, second.encodedArtifact);
  EXPECT_EQ(first.inputDigest, second.inputDigest);
  EXPECT_EQ(first.outputDigest, second.outputDigest);
  EXPECT_EQ(first.artifactDigest, second.artifactDigest);
  EXPECT_EQ(first.inputDigest, first.outputDigest);
  EXPECT_EQ(64, first.inputDigest.size());
  EXPECT_EQ(64, first.artifactDigest.size());
  ASSERT_EQ(first.samplesSeconds.size(), runnerConfig.samples);
  for (const auto sample : first.samplesSeconds) {
    EXPECT_GT(sample, 0.0);
  }
}

TEST_F(
    NimbleEncodingRunnerTest,
    stageZeroArtifactsRoundTripAcrossRunnerBoundary) {
  const std::array<std::string, 4> taskIds{
      "nimble.dictionary.decode_dense.v1",
      "nimble.nullable.decode_dense.v1",
      "nimble.alp.decode_dense.v1",
      "nimble.delta_block.decode_dense.v1",
  };

  for (const auto& taskId : taskIds) {
    SCOPED_TRACE(taskId);
    const auto runnerConfig = config(taskId);
    const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);
    const auto verification = verifyEncodingArtifact(
        runnerConfig, measurement.encodedArtifact, *pool_);

    EXPECT_EQ(measurement.taskId, verification.taskId);
    EXPECT_EQ(measurement.encoding, verification.encoding);
    EXPECT_EQ(measurement.dataType, verification.dataType);
    EXPECT_EQ(measurement.inputDigest, verification.inputDigest);
    EXPECT_EQ(measurement.outputDigest, verification.outputDigest);
    EXPECT_EQ(measurement.artifactDigest, verification.artifactDigest);
    EXPECT_EQ(measurement.encodedBytes, verification.encodedBytes);
  }
}

TEST_F(NimbleEncodingRunnerTest, stageZeroTaskMatrixProducesRawTimingSamples) {
  const std::array<std::string, 10> lanes{
      "encode",
      "decode_construct",
      "decode_dense",
      "decode_range50",
      "decode_scatter10",
      "decode_scatter1",
      "skip_seek",
      "view_random",
      "slice",
      "selection_e2e",
  };
  struct EncodingCase {
    std::string_view slug;
    std::string_view dataType;
    bool supportsView;
  };
  const std::array<EncodingCase, 4> encodings{{
      {"dictionary", "int64", true},
      {"nullable", "int64", false},
      {"alp", "double", true},
      {"delta_block", "int64", true},
  }};

  for (const auto& encoding : encodings) {
    for (const auto& lane : lanes) {
      if (lane == "view_random" && !encoding.supportsView) {
        continue;
      }
      SCOPED_TRACE(std::string{encoding.slug} + "." + lane);
      auto runnerConfig =
          config("nimble." + std::string{encoding.slug} + "." + lane + ".v1");
      runnerConfig.rowCount = 100;
      const auto measurement = runEncodingBenchmark(runnerConfig, *pool_);

      EXPECT_EQ(lane, measurement.lane);
      EXPECT_EQ(encoding.slug, measurement.encoding);
      EXPECT_EQ(encoding.dataType, measurement.dataType);
      EXPECT_EQ(measurement.inputDigest, measurement.outputDigest);
      EXPECT_EQ(runnerConfig.samples, measurement.samplesSeconds.size());
    }
  }
}

TEST_F(NimbleEncodingRunnerTest, invalidContractRejectsConfiguration) {
  auto tooFewSamples = config("nimble.dictionary.decode_dense.v1");
  tooFewSamples.samples = 4;
  EXPECT_THROW(
      runEncodingBenchmark(tooFewSamples, *pool_), std::invalid_argument);

  auto tooFewRows = config("nimble.dictionary.decode_dense.v1");
  tooFewRows.rowCount = 99;
  EXPECT_THROW(runEncodingBenchmark(tooFewRows, *pool_), std::invalid_argument);

  auto tooManyRows = config("nimble.dictionary.decode_dense.v1");
  tooManyRows.rowCount = 1'048'577;
  EXPECT_THROW(
      runEncodingBenchmark(tooManyRows, *pool_), std::invalid_argument);

  auto tooManySamples = config("nimble.dictionary.decode_dense.v1");
  tooManySamples.samples = 101;
  EXPECT_THROW(
      runEncodingBenchmark(tooManySamples, *pool_), std::invalid_argument);

  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.unknown.decode_dense.v1"), *pool_),
      std::invalid_argument);
  EXPECT_THROW(
      runEncodingBenchmark(config("nimble.nullable.view_random.v1"), *pool_),
      std::invalid_argument);
}

TEST_F(NimbleEncodingRunnerTest, mismatchedTaskRejectsArtifact) {
  const auto dictionary =
      runEncodingBenchmark(config("nimble.dictionary.decode_dense.v1"), *pool_);

  EXPECT_THROW(
      verifyEncodingArtifact(
          config("nimble.delta_block.decode_dense.v1"),
          dictionary.encodedArtifact,
          *pool_),
      std::runtime_error);
}

TEST_F(
    NimbleEncodingRunnerTest,
    consumerLaneAcceptsCanonicalArtifactAndProducerLaneRejectsIt) {
  const auto consumerConfig = config("nimble.dictionary.decode_dense.v1");
  const auto reference = runEncodingBenchmark(consumerConfig, *pool_);

  const auto candidate = runEncodingBenchmark(
      consumerConfig,
      *pool_,
      std::optional<std::string_view>{reference.encodedArtifact});

  EXPECT_EQ(reference.encodedArtifact, candidate.encodedArtifact);
  EXPECT_EQ(reference.artifactDigest, candidate.artifactDigest);
  EXPECT_EQ(reference.encodedBytes, candidate.encodedBytes);
  EXPECT_THROW(
      runEncodingBenchmark(
          config("nimble.dictionary.encode.v1"),
          *pool_,
          std::optional<std::string_view>{reference.encodedArtifact}),
      std::invalid_argument);
}

TEST_F(NimbleEncodingRunnerTest, emptyArtifactIsRejected) {
  EXPECT_THROW(
      verifyEncodingArtifact(
          config("nimble.dictionary.decode_dense.v1"), "", *pool_),
      std::runtime_error);
}

TEST_F(NimbleEncodingRunnerTest, rawMeasurementDoesNotClaimFinalGraderScore) {
  const auto measurement =
      runEncodingBenchmark(config("nimble.alp.decode_range50.v1"), *pool_);

  const auto json = folly::parseJson(measurementToJson(measurement));

  EXPECT_EQ(1, json["schema_version"].asInt());
  EXPECT_EQ("nimble_encoding_measurement", json["kind"].asString());
  EXPECT_EQ(measurement.taskId, json["task_id"].asString());
  EXPECT_EQ(measurement.lane, json["lane"].asString());
  EXPECT_TRUE(json["correctness"].asBool());
  EXPECT_EQ(measurement.samplesSeconds.size(), json["samples_seconds"].size());
  EXPECT_EQ(0, json.count("reward"));
  EXPECT_EQ(0, json.count("baseline_seconds"));
  EXPECT_EQ(0, json.count("encoded_artifact"));
}

TEST_F(NimbleEncodingRunnerTest, invalidRawMeasurementIsNotSerialized) {
  auto measurement =
      runEncodingBenchmark(config("nimble.alp.decode_dense.v1"), *pool_);
  measurement.samplesSeconds.front() = 0.0;

  EXPECT_THROW(measurementToJson(measurement), std::runtime_error);
}

} // namespace
} // namespace facebook::nimble::benchmarks
