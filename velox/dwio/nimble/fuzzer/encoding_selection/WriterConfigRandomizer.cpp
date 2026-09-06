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

#include <algorithm>
#include <random>
#include <string_view>

#include <fmt/format.h>
#include <folly/Random.h>
#include <folly/hash/Hash.h>

#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/velox/NimbleConfig.h"
#include "velox/dwio/nimble/writer/EncodingSelectionPolicyFactory.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/FlushPolicyFactory.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"

namespace facebook::nimble::fuzzer {
namespace {

std::mt19937_64 makeGenerator(uint64_t seed, std::string_view domain) {
  return std::mt19937_64{folly::hash::hash_combine(seed, domain)};
}

std::string boolString(bool value) {
  return value ? "true" : "false";
}

std::string flushModeName(FuzzedFlushMode mode) {
  switch (mode) {
    case FuzzedFlushMode::kRawSize:
      return "raw_size";
    case FuzzedFlushMode::kPerBatch:
      return "per_batch";
    case FuzzedFlushMode::kRandom:
      return "random";
  }
  NIMBLE_UNREACHABLE("Unknown fuzzed flush mode.");
}

} // namespace

void RandomizedWriterConfig::applyTo(WriterOptions& options) const {
  options.compressionOptions.compressionAcceptRatio = compressionAcceptRatio;
  options.compressionOptions.zstdMinCompressionSize = zstdMinCompressionSize;
  options.compressionOptions.internalMinCompressionSize =
      internalMinCompressionSize;
  options.compressionOptions.internalCompressionLevel =
      internalCompressionLevel;

  options.enableChunking = enableChunking;
  options.minStreamChunkRawSize = minStreamChunkRawSize;
  options.maxStreamChunkRawSize = maxStreamChunkRawSize;
  options.wideSchemaMaxStreamChunkRawSize = wideSchemaMaxStreamChunkRawSize;
  options.enableChunkIndex = enableChunkIndex;
  options.enableStreamDeduplication = enableStreamDeduplication;

  options.blockBitPackingBlockSize = blockBitPackingBlockSize;

  switch (flushMode) {
    case FuzzedFlushMode::kRawSize:
      options.flushPolicyFactory = [rawSize = stripeRawSize]() {
        return std::make_unique<StripeRawSizeFlushPolicy>(rawSize);
      };
      break;
    case FuzzedFlushMode::kPerBatch: {
      TestFlushPolicy::Options flushOptions;
      flushOptions.seed = flushSeed;
      flushOptions.flushChunkProbability = 1.0;
      flushOptions.flushStripeProbability = 0.0;
      options.flushPolicyFactory = TestFlushPolicyFactory{flushOptions};
    } break;
    case FuzzedFlushMode::kRandom:
      options.flushPolicyFactory =
          TestFlushPolicyFactory{TestFlushPolicy::Options{.seed = flushSeed}};
      break;
  }
}

void RandomizedWriterConfig::applyEncodingSelectionTo(
    WriterOptions& options,
    std::optional<EncodingType> forcedEncoding) const {
  if (!encodingSelectionSeed.has_value()) {
    return;
  }
  auto creator = createEncodingSelectionPolicyFactory(
      encodingSelectionConfig(forcedEncoding), options.compressionOptions);
  NIMBLE_CHECK(
      creator.has_value(),
      "Encoding selection config produced no policy creator for writer config seed {}.",
      seed);
  options.encodingSelectionPolicyCreator = std::move(*creator);
}

std::unordered_map<std::string, std::string>
RandomizedWriterConfig::toSerdeParams() const {
  std::unordered_map<std::string, std::string> params{
      {std::string(Config::ENCODING_SELECTION_COMPRESSION_ACCEPT_RATIO.key),
       fmt::format("{:.9g}", compressionAcceptRatio)},
      {std::string(Config::ZSTD_COMPRESSION_MIN_SIZE.key),
       fmt::format("{}", zstdMinCompressionSize)},
      {std::string(Config::ZSTRONG_COMPRESSION_MIN_SIZE.key),
       fmt::format("{}", internalMinCompressionSize)},
      {std::string(Config::ZSTRONG_COMPRESSION_LEVEL.key),
       fmt::format("{}", internalCompressionLevel)},
      {std::string(Config::ENABLE_CHUNKING.key), boolString(enableChunking)},
      {std::string(Config::CHUNKING_WRITER_MIN_CHUNK_SIZE.key),
       fmt::format("{}", minStreamChunkRawSize)},
      {std::string(Config::CHUNKING_WRITER_MAX_CHUNK_SIZE.key),
       fmt::format("{}", maxStreamChunkRawSize)},
      {std::string(Config::CHUNKING_WRITER_WIDE_SCHEMA_MAX_CHUNK_SIZE.key),
       fmt::format("{}", wideSchemaMaxStreamChunkRawSize)},
      {std::string(Config::ENABLE_CHUNK_INDEX.key),
       boolString(enableChunkIndex)},
      {std::string(Config::ENABLE_STREAM_DEDUPLICATION.key),
       boolString(enableStreamDeduplication)},
      {std::string(Config::BLOCK_BIT_PACKING_BLOCK_SIZE.key),
       fmt::format("{}", blockBitPackingBlockSize)},
  };

  switch (flushMode) {
    case FuzzedFlushMode::kRawSize:
      params.emplace(
          std::string(Config::RAW_STRIPE_SIZE.key),
          fmt::format("{}", stripeRawSize));
      break;
    case FuzzedFlushMode::kPerBatch:
      params.emplace(
          std::string(Config::FLUSH_POLICY_CONFIG.key),
          fmt::format(
              "type:test_per_batch,seed:{},flush_stripe_probability:0",
              flushSeed));
      break;
    case FuzzedFlushMode::kRandom:
      params.emplace(
          std::string(Config::FLUSH_POLICY_CONFIG.key),
          fmt::format("type:test_random,seed:{}", flushSeed));
      break;
  }

  if (encodingSelectionSeed.has_value()) {
    params.emplace(
        std::string(Config::ENCODING_SELECTION_CONFIG.key),
        encodingSelectionConfig());
  }
  return params;
}

std::string RandomizedWriterConfig::encodingSelectionConfig(
    std::optional<EncodingType> forcedEncoding) const {
  NIMBLE_CHECK(
      encodingSelectionSeed.has_value(),
      "Random encoding selection is disabled for writer config seed {}.",
      seed);
  if (forcedEncoding.has_value()) {
    return fmt::format(
        "type:random,seed:{},encodings:{}",
        *encodingSelectionSeed,
        toString(*forcedEncoding));
  }
  return fmt::format("type:random,seed:{}", *encodingSelectionSeed);
}

std::string RandomizedWriterConfig::debugString() const {
  return fmt::format(
      "version={},seed={},compression={{acceptRatio:{:.9g},internalLevel:{}}},chunking={{enabled:{},min:{},max:{},wideMax:{},index:{},dedup:{}}},encoding={{blockSize:{},selectionSeed:{}}},flush={{mode:{},rawSize:{},seed:{}}}",
      version,
      seed,
      compressionAcceptRatio,
      internalCompressionLevel,
      enableChunking,
      minStreamChunkRawSize,
      maxStreamChunkRawSize,
      wideSchemaMaxStreamChunkRawSize,
      enableChunkIndex,
      enableStreamDeduplication,
      blockBitPackingBlockSize,
      encodingSelectionSeed.has_value()
          ? fmt::format("{}", *encodingSelectionSeed)
          : "manual",
      flushModeName(flushMode),
      stripeRawSize,
      flushSeed);
}

RandomizedWriterConfig randomizeWriterConfig(
    uint64_t seed,
    WriterConfigRandomizationOptions options) {
  RandomizedWriterConfig config;
  config.seed = seed;

  auto compressionGenerator = makeGenerator(seed, "compression");
  config.compressionAcceptRatio = 0.5f +
      0.5f *
          static_cast<float>(folly::Random::randDouble01(compressionGenerator));
  config.internalCompressionLevel =
      1 + folly::Random::rand32(9, compressionGenerator);

  auto chunkingGenerator = makeGenerator(seed, "chunking");
  config.flushMode =
      static_cast<FuzzedFlushMode>(folly::Random::rand32(3, chunkingGenerator));
  if (config.flushMode == FuzzedFlushMode::kRawSize) {
    config.enableChunking = false;
  } else if (config.flushMode == FuzzedFlushMode::kPerBatch) {
    config.enableChunking = true;
  } else {
    config.enableChunking = !folly::Random::oneIn(4, chunkingGenerator);
  }
  const uint32_t maxChunkExponent =
      10 + folly::Random::rand32(12, chunkingGenerator);
  config.maxStreamChunkRawSize = uint64_t{1} << maxChunkExponent;
  config.wideSchemaMaxStreamChunkRawSize = config.maxStreamChunkRawSize;
  config.minStreamChunkRawSize = folly::Random::oneIn(2, chunkingGenerator)
      ? 0
      : uint64_t{1} << folly::Random::rand32(
            std::min<uint32_t>(14, maxChunkExponent), chunkingGenerator);
  config.enableChunkIndex = folly::Random::oneIn(2, chunkingGenerator);
  if (!config.enableChunking) {
    config.enableChunkIndex = false;
  }
  config.enableStreamDeduplication = folly::Random::oneIn(2, chunkingGenerator);

  auto encodingGenerator = makeGenerator(seed, "encoding");
  config.blockBitPackingBlockSize = static_cast<uint16_t>(
      uint16_t{1} << (5 + folly::Random::rand32(6, encodingGenerator)));
  if (options.randomEncodingSelectionOneIn != 0 &&
      (options.randomEncodingSelectionOneIn == 1 ||
       folly::Random::oneIn(
           options.randomEncodingSelectionOneIn, encodingGenerator))) {
    config.encodingSelectionSeed = folly::Random::rand64(encodingGenerator);
  }

  auto flushGenerator = makeGenerator(seed, "flush");
  config.flushSeed = folly::Random::rand64(flushGenerator);
  if (config.flushMode == FuzzedFlushMode::kRawSize) {
    config.stripeRawSize = uint64_t{1}
        << (12 + folly::Random::rand32(8, flushGenerator));
  }

  return config;
}

} // namespace facebook::nimble::fuzzer
