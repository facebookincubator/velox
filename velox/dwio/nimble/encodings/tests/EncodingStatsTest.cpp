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

#include <folly/hash/Hash.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/compression/CompressionPolicy.h"
#include "velox/dwio/nimble/compression/EncodingStats.h"
#include "velox/dwio/nimble/encodings/FixedBitWidthEncoding.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"

namespace facebook::nimble {
namespace {

// A shadow re-encode that fails.
[[noreturn]] uint64_t failingReEncode() {
  throw std::runtime_error("shadow failed");
}

velox::memory::MemoryPool& leafPool() {
  static const auto pool = velox::memory::deprecatedAddDefaultLeafMemoryPool();
  return *pool;
}

std::map<std::string, uint64_t> counterMap(const EncodingStats& stats) {
  std::map<std::string, uint64_t> counters;
  for (const auto& counter : stats.counters()) {
    EXPECT_TRUE(counters.emplace(counter.name, counter.value).second)
        << counter.name;
  }
  return counters;
}

const std::vector<std::string> kAlwaysPresent{
    "encoding.typedLanes.full.chunks",
    "encoding.typedLanes.full.rawBytes",
    "encoding.typedLanes.full.storedBytes",
    "encoding.typedLanes.narrow.chunks",
    "encoding.typedLanes.narrow.rawBytes",
    "encoding.typedLanes.narrow.storedBytes",
    "encoding.replayedChunks",
    "encoding.replayFallbacks",
};

// Checks that every name in 'counters' has its family's number of
// '.'-separated segments; the family is the second segment.
void expectFixedSegments(const std::map<std::string, uint64_t>& counters) {
  const std::map<std::string, size_t> kSegments{
      {"payload", 8},
      {"typedLanes", 4},
      {"shadow", 5},
      {"replayedChunks", 2},
      {"replayFallbacks", 2},
      {"shadowSampleSeed", 2}};
  for (const auto& [name, value] : counters) {
    SCOPED_TRACE(name);
    const auto family = name.substr(9, name.find('.', 9) - 9);
    ASSERT_TRUE(name.starts_with("encoding.") && kSegments.contains(family));
    EXPECT_EQ(
        static_cast<size_t>(std::count(name.begin(), name.end(), '.')) + 1,
        kSegments.at(family));
  }
}

TEST(EncodingStatsTest, alwaysPresentCountersAtZero) {
  for (const uint32_t rate : {0u, 100u}) {
    SCOPED_TRACE(rate);
    EncodingStats stats{rate};
    const auto counters = counterMap(stats);
    for (const auto& name : kAlwaysPresent) {
      ASSERT_TRUE(counters.contains(name)) << name;
      EXPECT_EQ(counters.at(name), 0) << name;
    }
    // Shadow counters appear once a chunk is re-encoded; a nonzero rate adds
    // its seed.
    EXPECT_EQ(counters.size(), kAlwaysPresent.size() + (rate > 0 ? 1 : 0));
  }
}

// A nonzero sample rate publishes its seed, in its own family; a zero rate
// publishes none.
TEST(EncodingStatsTest, shadowSampleSeedCounterOnlyWithRate) {
  for (const uint64_t seed :
       {uint64_t{0}, uint64_t{42}, uint64_t{0x7FFFFFFFFFFFFFFF}}) {
    SCOPED_TRACE(seed);
    const auto counters = counterMap(EncodingStats{100, seed});
    expectFixedSegments(counters);
    ASSERT_TRUE(counters.contains("encoding.shadowSampleSeed"));
    EXPECT_EQ(counters.at("encoding.shadowSampleSeed"), seed);
    EXPECT_EQ(EncodingStats(100, seed).shadowSampleSeed(), seed);
  }
  EXPECT_FALSE(
      counterMap(EncodingStats{0, 42}).contains("encoding.shadowSampleSeed"));
  EXPECT_FALSE(
      counterMap(EncodingStats{}).contains("encoding.shadowSampleSeed"));
}

TEST(EncodingStatsTest, payloadCountersAndLaneTotals) {
  EncodingStats stats;
  // Narrow 32-bit lanes at level 1, accepted.
  stats.addPayload(
      {EncodingType::FixedBitWidth,
       DataType::Uint32,
       CompressionType::MetaInternal,
       1,
       false},
      4'008,
      1'000);
  stats.addPayload(
      {EncodingType::FixedBitWidth,
       DataType::Uint32,
       CompressionType::MetaInternal,
       1,
       false},
      4'008,
      1'200);
  // Full-width 64-bit lanes, compression declined at level 1.
  stats.addPayload(
      {EncodingType::FixedBitWidth,
       DataType::Int64,
       CompressionType::Uncompressed,
       1,
       true},
      8'008,
      8'008);
  // Bit-packed FixedBitWidth and Trivial: not lanes.
  stats.addPayload(
      {EncodingType::FixedBitWidth,
       DataType::Undefined,
       CompressionType::MetaInternal,
       4,
       false},
      1'007,
      500);
  stats.addPayload(
      {EncodingType::Trivial,
       DataType::String,
       CompressionType::Uncompressed,
       0,
       false},
      100,
      100);
  stats.addReplayedChunk();
  stats.addReplayFallback();
  stats.addReplayFallback();

  const auto counters = counterMap(stats);
  expectFixedSegments(counters);
  EXPECT_EQ(counters.at("encoding.typedLanes.narrow.chunks"), 2);
  EXPECT_EQ(counters.at("encoding.typedLanes.narrow.rawBytes"), 4'008 * 2);
  EXPECT_EQ(counters.at("encoding.typedLanes.narrow.storedBytes"), 2'200);
  EXPECT_EQ(counters.at("encoding.typedLanes.full.chunks"), 1);
  EXPECT_EQ(counters.at("encoding.typedLanes.full.rawBytes"), 8'008);
  EXPECT_EQ(counters.at("encoding.typedLanes.full.storedBytes"), 8'008);
  EXPECT_EQ(counters.at("encoding.replayedChunks"), 1);
  EXPECT_EQ(counters.at("encoding.replayFallbacks"), 2);
  EXPECT_EQ(
      counters.at(
          "encoding.payload.FixedBitWidth.Uint32.narrow.MetaInternal.L1.chunks"),
      2);
  EXPECT_EQ(
      counters.at(
          "encoding.payload.FixedBitWidth.Uint32.narrow.MetaInternal.L1.storedBytes"),
      2'200);
  EXPECT_EQ(
      counters.at(
          "encoding.payload.FixedBitWidth.Int64.full.Uncompressed.L1.chunks"),
      1);
  EXPECT_EQ(
      counters.at(
          "encoding.payload.FixedBitWidth.Undefined.none.MetaInternal.L4.rawBytes"),
      1'007);
  EXPECT_EQ(
      counters.at(
          "encoding.payload.Trivial.String.none.Uncompressed.none.chunks"),
      1);
}

// Every payload a CompressionEncoder can count gets a well-formed counter
// name: every encoding that compresses, every data type (Undefined included),
// every compression type, levels and the full-width flag, each with its
// family's number of segments. The full-width flag names a width only for
// FixedBitWidth typed lanes, so elsewhere it doesn't make a payload distinct.
TEST(EncodingStatsTest, payloadCounterNamesAreClean) {
  EncodingStats stats;
  const std::vector<EncodingType> encodings{
      EncodingType::Trivial,
      EncodingType::Fsst,
      EncodingType::FixedBitWidth,
      EncodingType::FOR,
      EncodingType::BlockBitPacking,
      EncodingType::BlockBitPackingV2};
  const std::vector<CompressionType> compressions{
      CompressionType::Uncompressed,
      CompressionType::Zstd,
      CompressionType::MetaInternal,
      CompressionType::Lz4,
      CompressionType::OpenZL};
  // Every DataType enumerator, in declaration order. DataType::String is the
  // last one (common/Types.h): extend this list when an enumerator is added.
  constexpr std::array kDataTypes{
      DataType::Undefined,
      DataType::Int8,
      DataType::Uint8,
      DataType::Int16,
      DataType::Uint16,
      DataType::Int32,
      DataType::Uint32,
      DataType::Int64,
      DataType::Uint64,
      DataType::Float,
      DataType::Double,
      DataType::Bool,
      DataType::String};
  static_assert(kDataTypes.back() == DataType::String);
  static_assert(
      kDataTypes.size() == static_cast<size_t>(DataType::String) + 1,
      "kDataTypes must list every DataType enumerator.");
  size_t payloads = 0;
  for (const auto encoding : encodings) {
    for (const auto dataType : kDataTypes) {
      for (const auto compression : compressions) {
        for (const int32_t level : {0, 1, 22}) {
          for (const bool fullWidth : {false, true}) {
            stats.addPayload(
                {encoding, dataType, compression, level, fullWidth}, 1, 1);
            const bool lanes = encoding == EncodingType::FixedBitWidth &&
                dataType != DataType::Undefined;
            payloads += !fullWidth || lanes ? 1 : 0;
          }
        }
      }
    }
  }
  const auto counters = counterMap(stats);
  // Three counters per distinct payload, plus the always-present ones.
  EXPECT_EQ(counters.size(), kAlwaysPresent.size() + 3 * payloads);
  expectFixedSegments(counters);
  for (const auto& [name, value] : counters) {
    SCOPED_TRACE(name);
    EXPECT_EQ(name.find("Unknown"), std::string::npos);
    EXPECT_EQ(name.find(' '), std::string::npos);
    EXPECT_EQ(name.find(':'), std::string::npos);
  }
  EXPECT_TRUE(counters.contains(
      "encoding.payload.FixedBitWidth.Undefined.none.MetaInternal.L1.chunks"));
  EXPECT_TRUE(counters.contains(
      "encoding.payload.FixedBitWidth.Int64.full.OpenZL.none.chunks"));
  EXPECT_TRUE(counters.contains(
      "encoding.payload.FixedBitWidth.Int64.narrow.OpenZL.none.chunks"));
  EXPECT_TRUE(counters.contains(
      "encoding.payload.Trivial.Bool.none.Uncompressed.none.chunks"));
}

TEST(EncodingStatsTest, chunkScopeKeysSampling) {
  const EncodingStats seeded{10, 0x5EED};
  const auto chunkKey = seeded.shadowSampleKey(7, 0);
  EXPECT_EQ(chunkKey, seeded.shadowSampleKey(7, 0));
  EXPECT_EQ(chunkKey, (EncodingStats{100, 0x5EED}.shadowSampleKey(7, 0)));
  EXPECT_NE(chunkKey, seeded.shadowSampleKey(8, 0));
  EXPECT_NE(chunkKey, seeded.shadowSampleKey(7, 1));
  EXPECT_NE(chunkKey, (EncodingStats{10, 0x5EEE}.shadowSampleKey(7, 0)));
  EXPECT_FALSE(EncodingChunkScope::current().has_value());
  {
    EncodingChunkScope scope{7, 0};
    const EncodingChunkScope::Chunk outer{.streamOffset = 7, .chunkIndex = 0};
    EXPECT_EQ(EncodingChunkScope::current(), outer);
    {
      EncodingChunkScope inner{7, 1};
      EXPECT_EQ(
          EncodingChunkScope::current(),
          (EncodingChunkScope::Chunk{.streamOffset = 7, .chunkIndex = 1}));
    }
    EXPECT_EQ(EncodingChunkScope::current(), outer);
  }
  EXPECT_FALSE(EncodingChunkScope::current().has_value());

  EncodingStats stats{1};
  // Rate 1 samples every chunk.
  for (uint64_t key = 0; key < 100; ++key) {
    EXPECT_TRUE(stats.sampleShadow(key));
  }
  // Rate 10 samples some, deterministically.
  EncodingStats sparse{10};
  int sampled = 0;
  for (uint64_t key = 0; key < 1'000; ++key) {
    sampled += sparse.sampleShadow(key) ? 1 : 0;
    EXPECT_EQ(sparse.sampleShadow(key), sparse.sampleShadow(key));
  }
  EXPECT_GT(sampled, 0);
  EXPECT_LT(sampled, 1'000);
}

// The key is hashCombine(hashCombine(streamOffset, chunkIndex), seed), with
// hashCombine folly::hash::hash_128_to_64, and a chunk is sampled when
// twang_mix64 of its key is 0 modulo the rate. The expected values were
// computed independently, in Python.
TEST(EncodingStatsTest, shadowSampleKeyMatchesGolden) {
  struct Golden {
    uint32_t streamOffset;
    uint64_t chunkIndex;
    uint64_t seed;
    uint64_t key;
    uint64_t mix;
    bool sampledAt100;
  };
  // One row per line, as in the golden TSV.
  // clang-format off
  constexpr std::array<Golden, 8> kGolden{{
      {0, 0, 0x0000000000000000, 0x0000000000000000, 0x77cfa1eef01bca90, false},
      {4096, 0, 0x0000000000000001, 0x0e60ee1013024747, 0x06dd49bbd513c93b, false},
      {123456, 7, 0x0123456789abcdef, 0x291287be0ae5fe01, 0x7e8dc4d22052822a, false},
      {4294967295, 3, 0x7fffffffffffffff, 0xf818ea67795d6d08, 0xf70a342b82ebdbfc, false},
      {987654, 12, 0x5eed5eed5eed5eed, 0xf160226e1b261948, 0xbf5d9ec44b96e963, false},
      {1048576, 28, 0xfedcba9876543210, 0x46c2b280f81b2feb, 0x3b98961e000e1779, false},
      {7, 194, 0x0000000000005eed, 0x3de76dea65474a48, 0x2296b6b38de22f24, true},
      {1048576, 45, 0x000000000000002a, 0xebfc41928fed185f, 0xb9b5386188705990, true},
  }};
  // clang-format on
  for (const auto& golden : kGolden) {
    SCOPED_TRACE(
        ::testing::Message() << golden.streamOffset << " " << golden.chunkIndex
                             << " " << golden.seed);
    const EncodingStats stats{100, golden.seed};
    const auto key =
        stats.shadowSampleKey(golden.streamOffset, golden.chunkIndex);
    EXPECT_EQ(key, golden.key);
    EXPECT_EQ(folly::hash::twang_mix64(key), golden.mix);
    EXPECT_EQ(stats.sampleShadow(key), golden.sampledAt100);
  }
}

// The sum of the payload chunk counters.
uint64_t payloadChunks(const std::map<std::string, uint64_t>& counters) {
  uint64_t chunks = 0;
  for (const auto& [name, value] : counters) {
    if (name.starts_with("encoding.payload.") && name.ends_with(".chunks")) {
      chunks += value;
    }
  }
  return chunks;
}

// OpenZL reads the payload's DataType, so FixedBitWidth chunks can take lanes.
CompressionOptions openzlOptions() {
  return CompressionOptions{
      .compressionType = CompressionType::OpenZL,
      .openzlMinCompressionSize = 0,
  };
}

// Encodes 'values' as FixedBitWidth, compressed with OpenZL.
template <typename T>
std::string encodeFixedBitWidth(
    const std::vector<T>& values,
    const Encoding::Options& options) {
  Buffer buffer{leafPool()};
  auto policy = std::make_unique<ManualEncodingSelectionPolicy<T>>(
      std::vector<std::pair<EncodingType, float>>{
          {EncodingType::FixedBitWidth, 1.0}},
      openzlOptions(),
      /*identifier=*/std::nullopt);
  return std::string{EncodingFactory::encode<T>(
      std::move(policy), std::span<const T>{values}, buffer, options)};
}

// FixedBitWidth counts its payloads with their lane width, and counting leaves
// the chunk's bytes alone. Within a sampled chunk it also re-encodes its lanes
// bit-packed, still leaving the chunk's bytes alone, and the re-encode is not
// counted as a payload.
TEST(EncodingStatsTest, fixedBitWidthCountsPayloads) {
  std::vector<uint32_t> values(4'096);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint32_t>((i * 37) % 64) << 18;
  }
  Encoding::Options lanes;
  lanes.fixedBitWidthTypedLanes = true;
  const auto withoutStats = encodeFixedBitWidth<uint32_t>(values, lanes);

  EncodingStats stats{1};
  auto counted = lanes;
  counted.encodingStats = &stats;
  {
    EncodingStats::ShadowChunk chunk;
    EXPECT_EQ(encodeFixedBitWidth<uint32_t>(values, counted), withoutStats);
    stats.shadowEncode(values.size(), chunk);
  }
  // The prefix, then the compression type, the bit width and the baseline.
  const auto payloadSize = [&](const std::string& encoded) -> uint64_t {
    return encoded.size() -
        EncodingPrefix::serializedSize(
               static_cast<uint32_t>(values.size()), false) -
        2 - sizeof(uint32_t);
  };
  const uint64_t lanePayload = payloadSize(withoutStats);
  const uint64_t packedPayload =
      payloadSize(encodeFixedBitWidth<uint32_t>(values, Encoding::Options{}));

  const auto counters = counterMap(stats);
  expectFixedSegments(counters);
  // 24 bits of a 32-bit type: full-width lanes.
  EXPECT_EQ(counters.at("encoding.typedLanes.full.chunks"), 1);
  EXPECT_EQ(counters.at("encoding.typedLanes.narrow.chunks"), 0);
  EXPECT_EQ(
      counters.at("encoding.typedLanes.full.rawBytes"), values.size() * 4 + 8);
  EXPECT_EQ(counters.at("encoding.typedLanes.full.storedBytes"), lanePayload);
  // OpenZL: no MetaInternal level, so the lanes alone changed the chunk.
  const std::string shadow{
      "encoding.shadow.fixedBitWidthTypedLanes.FixedBitWidth."};
  EXPECT_EQ(counters.at(shadow + "sampledChunks"), 1);
  EXPECT_EQ(counters.at(shadow + "sampledValues"), values.size());
  EXPECT_EQ(counters.at(shadow + "actualStoredBytes"), lanePayload);
  EXPECT_EQ(counters.at(shadow + "baselineStoredBytes"), packedPayload);
  EXPECT_EQ(counters.at(shadow + "failures"), 0);
  EXPECT_EQ(payloadChunks(counters), 1);
}

// Payloads counted in tallies and folded give the counters, names, values and
// order, that counting each with addPayload() gives, also when a tally fills
// up and counts the rest with addPayload(), and a folded tally starts over.
// An encode counts in its tally only with its EncodingStats set.
TEST(EncodingStatsTest, payloadTallyMatchesAddPayload) {
  // 66 distinct payload keys, more than a tally keeps.
  std::vector<EncodingStats::Payload> payloads;
  for (const auto encoding :
       {EncodingType::FixedBitWidth, EncodingType::Trivial}) {
    for (const auto dataType :
         {DataType::Undefined,
          DataType::Int32,
          DataType::Uint32,
          DataType::Int64}) {
      for (const auto compression :
           {CompressionType::Uncompressed, CompressionType::MetaInternal}) {
        for (const int32_t level : {0, 1, 4}) {
          for (const bool fullWidth : {false, true}) {
            payloads.push_back(
                {encoding, dataType, compression, level, fullWidth});
          }
        }
      }
    }
  }
  EncodingStats direct;
  EncodingStats tallied;
  PayloadTally first{tallied};
  PayloadTally second{tallied};
  first.add(payloads.front(), 10, 5);
  direct.addPayload(payloads.front(), 10, 5);
  EXPECT_EQ(counterMap(tallied).size(), kAlwaysPresent.size());
  uint64_t bytes = 100;
  for (const bool split : {false, true}) {
    for (size_t i = 0; i < payloads.size(); ++i, ++bytes) {
      direct.addPayload(payloads[i], bytes, bytes / 2);
      (split && i % 2 == 1 ? second : first).add(payloads[i], bytes, bytes / 2);
    }
    first.fold();
    second.fold();
  }
  EXPECT_GT(first.overflows(), 0);
  EXPECT_EQ(second.overflows(), 0);
  const auto expected = direct.counters();
  const auto actual = tallied.counters();
  ASSERT_EQ(actual.size(), expected.size());
  for (size_t i = 0; i < expected.size(); ++i) {
    SCOPED_TRACE(expected[i].name);
    EXPECT_EQ(actual[i].name, expected[i].name);
    EXPECT_EQ(actual[i].value, expected[i].value);
    EXPECT_EQ(actual[i].unit, expected[i].unit);
  }

  EXPECT_EQ(PayloadTally::current(), nullptr);
  {
    const PayloadTally::Scope scope{&first};
    EXPECT_EQ(PayloadTally::current(), &first);
    {
      const PayloadTally::Scope inner{nullptr};
      EXPECT_EQ(PayloadTally::current(), nullptr);
    }
    EXPECT_EQ(PayloadTally::current(), &first);
  }
  EXPECT_EQ(PayloadTally::current(), nullptr);

  std::vector<uint32_t> values(1'024);
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = static_cast<uint32_t>((i * 37) % 64) << 18;
  }
  EncodingStats stats;
  PayloadTally tally{stats};
  Encoding::Options options;
  options.payloadTally = &tally;
  encodeFixedBitWidth<uint32_t>(values, options);
  options.encodingStats = &stats;
  encodeFixedBitWidth<uint32_t>(values, options);
  EXPECT_EQ(payloadChunks(counterMap(stats)), 0);
  tally.fold();
  EXPECT_EQ(payloadChunks(counterMap(stats)), 1);
}

// A sampled chunk counts once, under the settings that changed it, with its
// lane payloads' stored bytes and both CPU times. A failed re-encode counts a
// failure and its CPU times, and a chunk without lane payloads isn't counted.
TEST(EncodingStatsTest, shadowChunkCountsTheChunkOnce) {
  using ShadowChunk = EncodingStats::ShadowChunk;
  EncodingStats stats{1};
  const EncodingStats::Payload lanes{
      .encodingType = EncodingType::FixedBitWidth,
      .dataType = DataType::Uint32,
      .compressionType = CompressionType::OpenZL,
      .level = 0,
      .fullWidth = false};
  auto level = lanes;
  level.compressionType = CompressionType::MetaInternal;
  level.level = 1;
  level.typedLevel = true;
  const auto start = ShadowChunk::threadCpuNanos();

  EXPECT_EQ(ShadowChunk::current(), nullptr);
  {
    ShadowChunk chunk;
    EXPECT_EQ(ShadowChunk::current(), &chunk);
    chunk.reEncode(lanes, 100, start, [] { return uint64_t{150}; });
    chunk.reEncode(lanes, 30, start, [] { return uint64_t{20}; });
    stats.shadowEncode(1'000, chunk);
  }
  EXPECT_EQ(ShadowChunk::current(), nullptr);
  {
    ShadowChunk chunk;
    chunk.reEncode(lanes, 10, start, [] { return uint64_t{10}; });
    chunk.reEncode(level, 200, start, [] { return uint64_t{300}; });
    stats.shadowEncode(500, chunk);
  }
  {
    ShadowChunk chunk;
    chunk.reEncode(lanes, 50, start, failingReEncode);
    stats.shadowEncode(400, chunk);
  }
  {
    ShadowChunk chunk;
    stats.shadowEncode(700, chunk);
  }

  const auto counters = counterMap(stats);
  expectFixedSegments(counters);
  const std::string lanesKey{
      "encoding.shadow.fixedBitWidthTypedLanes.FixedBitWidth."};
  EXPECT_EQ(counters.at(lanesKey + "sampledChunks"), 1);
  EXPECT_EQ(counters.at(lanesKey + "sampledValues"), 1'000);
  EXPECT_EQ(counters.at(lanesKey + "actualStoredBytes"), 130);
  EXPECT_EQ(counters.at(lanesKey + "baselineStoredBytes"), 170);
  EXPECT_EQ(counters.at(lanesKey + "failures"), 1);
  EXPECT_GT(counters.at(lanesKey + "actualCpuNs"), 0);
  EXPECT_TRUE(counters.contains(lanesKey + "baselineCpuNs"));
  const std::string pairKey{
      "encoding.shadow.fixedBitWidthTypedLanes+fixedBitWidthTypedLanesOpenzlLevel."
      "FixedBitWidth."};
  EXPECT_EQ(counters.at(pairKey + "sampledChunks"), 1);
  EXPECT_EQ(counters.at(pairKey + "sampledValues"), 500);
  EXPECT_EQ(counters.at(pairKey + "actualStoredBytes"), 210);
  EXPECT_EQ(counters.at(pairKey + "baselineStoredBytes"), 310);
  EXPECT_EQ(counters.at(pairKey + "failures"), 0);
  // Rate 1 also publishes the seed counter.
  EXPECT_TRUE(counters.contains("encoding.shadowSampleSeed"));
  EXPECT_EQ(counters.size(), kAlwaysPresent.size() + 1 + 2 * 7);
}

} // namespace
} // namespace facebook::nimble
