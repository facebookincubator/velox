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

#include "velox/dwio/nimble/compression/EncodingStats.h"

#include <fmt/format.h>
#include <folly/hash/Hash.h>

#include <string_view>
#include <utility>

#include "velox/common/process/ProcessBase.h"
#include "velox/common/testutil/TestValue.h"
#include "velox/dwio/nimble/common/Exceptions.h"

namespace facebook::nimble {
namespace {

thread_local std::optional<EncodingChunkScope::Chunk> currentChunk;
thread_local EncodingStats::ShadowChunk* currentShadowChunk{nullptr};
thread_local PayloadTally* currentPayloadTally{nullptr};

uint64_t hashCombine(uint64_t seed, uint64_t value) {
  return folly::hash::hash_128_to_64(seed, value);
}

uint64_t load(const std::atomic<uint64_t>& counter) {
  return counter.load(std::memory_order_relaxed);
}

// The name of 'dataType' in counter names. toString(DataType) has no name for
// Undefined, the data type of bit-packed payloads; every other data type a
// payload can carry has one.
std::string dataTypeName(DataType dataType) {
  return dataType == DataType::Undefined ? std::string{"Undefined"}
                                         : toString(dataType);
}

// The <width> segment of a payload's counter names: FixedBitWidth typed lanes
// as wide as their values' type ("full") or narrower ("narrow"), or a payload
// that isn't typed lanes ("none"). Typed lanes are the only typed FixedBitWidth
// payloads.
std::string_view laneWidthName(const EncodingStats::Payload& payload) {
  if (payload.encodingType != EncodingType::FixedBitWidth ||
      payload.dataType == DataType::Undefined) {
    return "none";
  }
  return payload.fullWidth ? "full" : "narrow";
}

} // namespace

EncodingStats::EncodingStats(
    uint32_t shadowSampleRate,
    uint64_t shadowSampleSeed)
    : shadowSampleRate_{shadowSampleRate},
      shadowSampleSeed_{shadowSampleSeed} {}

uint64_t EncodingStats::payloadKey(const Payload& payload) {
  // laneWidthName(payload) == "full", without comparing strings.
  const bool fullLanes = payload.encodingType == EncodingType::FixedBitWidth &&
      payload.dataType != DataType::Undefined && payload.fullWidth;
  return static_cast<uint64_t>(payload.encodingType) << 40 |
      static_cast<uint64_t>(payload.dataType) << 32 |
      static_cast<uint64_t>(payload.compressionType) << 24 |
      static_cast<uint64_t>(static_cast<uint32_t>(payload.level) & 0xFFFF)
      << 8 |
      static_cast<uint64_t>(fullLanes);
}

void EncodingStats::addPayload(
    const Payload& payload,
    uint64_t rawBytes,
    uint64_t storedBytes) {
  std::lock_guard<std::mutex> lock{mutex_};
  auto& entry =
      payloads_
          .try_emplace(payloadKey(payload), std::make_pair(payload, Slot{}))
          .first->second;
  ++entry.second.chunks;
  entry.second.rawBytes += rawBytes;
  entry.second.storedBytes += storedBytes;
}

void EncodingStats::addTally(const PayloadTally& tally) {
  std::lock_guard<std::mutex> lock{mutex_};
  for (const auto& entry : tally.entries_) {
    if (!entry.used) {
      continue;
    }
    auto& slot =
        payloads_.try_emplace(entry.key, std::make_pair(entry.payload, Slot{}))
            .first->second.second;
    slot.chunks += entry.chunks;
    slot.rawBytes += entry.rawBytes;
    slot.storedBytes += entry.storedBytes;
  }
}

uint64_t EncodingStats::shadowSampleKey(
    uint32_t streamOffset,
    uint64_t chunkIndex) const {
  return hashCombine(hashCombine(streamOffset, chunkIndex), shadowSampleSeed_);
}

bool EncodingStats::sampleShadow(uint64_t key) const {
  NIMBLE_DCHECK_GT(shadowSampleRate_, 0);
  return folly::hash::twang_mix64(key) % shadowSampleRate_ == 0;
}

EncodingStats::ShadowChunk::ShadowChunk() : previous_{currentShadowChunk} {
  currentShadowChunk = this;
}

EncodingStats::ShadowChunk::~ShadowChunk() {
  currentShadowChunk = previous_;
}

EncodingStats::ShadowChunk* EncodingStats::ShadowChunk::current() {
  return currentShadowChunk;
}

uint64_t EncodingStats::ShadowChunk::threadCpuNanos() {
  return velox::process::threadCpuNanos();
}

void EncodingStats::ShadowChunk::reEncode(
    const Payload& payload,
    uint64_t storedBytes,
    uint64_t startCpuNs,
    const std::function<uint64_t()>& encode) {
  const auto reEncodeStartCpuNs = threadCpuNanos();
  ++payloads_;
  typedLevel_ = typedLevel_ || payload.typedLevel;
  actualStoredBytes_ += storedBytes;
  actualCpuNs_ += reEncodeStartCpuNs - startCpuNs;
  try {
    velox::common::testutil::TestValue::adjust(
        "facebook::nimble::EncodingStats::shadowEncode", &storedBytes);
    baselineStoredBytes_ += encode();
  } catch (const std::exception&) {
    // The re-encode is a measurement; the real write goes on.
    failed_ = true;
  }
  baselineCpuNs_ += threadCpuNanos() - reEncodeStartCpuNs;
}

void EncodingStats::shadowEncode(uint64_t values, const ShadowChunk& chunk) {
  if (chunk.payloads_ == 0) {
    return;
  }
  // Names the settings that changed the chunk: typed lanes changed every
  // typed-lane payload, and the level those it compressed.
  const std::string settings = chunk.typedLevel_
      ? "fixedBitWidthTypedLanes+fixedBitWidthTypedLanesOpenzlLevel"
      : "fixedBitWidthTypedLanes";
  std::lock_guard<std::mutex> lock{mutex_};
  auto& slot = shadows_[{settings, EncodingType::FixedBitWidth}];
  slot.actualCpuNs += chunk.actualCpuNs_;
  slot.baselineCpuNs += chunk.baselineCpuNs_;
  if (chunk.failed_) {
    ++slot.failures;
    return;
  }
  ++slot.sampledChunks;
  slot.sampledValues += values;
  slot.actualStoredBytes += chunk.actualStoredBytes_;
  slot.baselineStoredBytes += chunk.baselineStoredBytes_;
}

std::vector<EncodingStats::Counter> EncodingStats::counters() const {
  Slot fullLanes;
  Slot narrowLanes;
  std::vector<Counter> payloadCounters;
  {
    std::lock_guard<std::mutex> lock{mutex_};
    for (const auto& [key, entry] : payloads_) {
      const auto& [payload, slot] = entry;
      const auto width = laneWidthName(payload);
      if (width != "none") {
        auto& total = width == "full" ? fullLanes : narrowLanes;
        total.chunks += slot.chunks;
        total.rawBytes += slot.rawBytes;
        total.storedBytes += slot.storedBytes;
      }
      const auto prefix = fmt::format(
          "encoding.payload.{}.{}.{}.{}.{}.",
          toString(payload.encodingType),
          dataTypeName(payload.dataType),
          width,
          toString(payload.compressionType),
          payload.level != 0 ? fmt::format("L{}", payload.level) : "none");
      payloadCounters.push_back({prefix + "chunks", slot.chunks, Unit::kNone});
      payloadCounters.push_back(
          {prefix + "rawBytes", slot.rawBytes, Unit::kBytes});
      payloadCounters.push_back(
          {prefix + "storedBytes", slot.storedBytes, Unit::kBytes});
    }
  }
  std::vector<Counter> counters;
  for (const auto& [width, total] :
       {std::pair{"full", fullLanes}, std::pair{"narrow", narrowLanes}}) {
    const auto prefix = fmt::format("encoding.typedLanes.{}.", width);
    counters.push_back({prefix + "chunks", total.chunks, Unit::kNone});
    counters.push_back({prefix + "rawBytes", total.rawBytes, Unit::kBytes});
    counters.push_back(
        {prefix + "storedBytes", total.storedBytes, Unit::kBytes});
  }
  counters.push_back(
      {"encoding.replayedChunks", load(replayedChunks_), Unit::kNone});
  counters.push_back(
      {"encoding.replayFallbacks", load(replayFallbacks_), Unit::kNone});
  if (shadowSampleRate_ > 0) {
    counters.push_back(
        {"encoding.shadowSampleSeed", shadowSampleSeed_, Unit::kNone});
  }
  counters.insert(
      counters.end(),
      std::make_move_iterator(payloadCounters.begin()),
      std::make_move_iterator(payloadCounters.end()));
  std::lock_guard<std::mutex> lock{mutex_};
  for (const auto& [key, slot] : shadows_) {
    const auto prefix =
        fmt::format("encoding.shadow.{}.{}.", key.first, toString(key.second));
    counters.push_back(
        {prefix + "sampledChunks", slot.sampledChunks, Unit::kNone});
    counters.push_back(
        {prefix + "sampledValues", slot.sampledValues, Unit::kNone});
    counters.push_back(
        {prefix + "actualStoredBytes", slot.actualStoredBytes, Unit::kBytes});
    counters.push_back(
        {prefix + "baselineStoredBytes",
         slot.baselineStoredBytes,
         Unit::kBytes});
    counters.push_back(
        {prefix + "actualCpuNs", slot.actualCpuNs, Unit::kNanos});
    counters.push_back(
        {prefix + "baselineCpuNs", slot.baselineCpuNs, Unit::kNanos});
    counters.push_back({prefix + "failures", slot.failures, Unit::kNone});
  }
  return counters;
}

void PayloadTally::add(
    const EncodingStats::Payload& payload,
    uint64_t rawBytes,
    uint64_t storedBytes) {
  const uint64_t key = EncodingStats::payloadKey(payload);
  // Linear probing. The table keeps fewer keys than slots, so a probe always
  // ends at the key's entry or at a free one.
  size_t slot = folly::hash::twang_mix64(key) & (kSlots - 1);
  while (entries_[slot].used && entries_[slot].key != key) {
    slot = (slot + 1) & (kSlots - 1);
  }
  auto& entry = entries_[slot];
  if (!entry.used) {
    if (keys_ == kMaxKeys) {
      ++overflows_;
      stats_->addPayload(payload, rawBytes, storedBytes);
      return;
    }
    entry.key = key;
    entry.payload = payload;
    entry.used = true;
    ++keys_;
  }
  ++entry.chunks;
  entry.rawBytes += rawBytes;
  entry.storedBytes += storedBytes;
}

void PayloadTally::fold() {
  if (keys_ == 0) {
    return;
  }
  stats_->addTally(*this);
  entries_.fill(Entry{});
  keys_ = 0;
}

PayloadTally::Scope::Scope(PayloadTally* tally)
    : previous_{currentPayloadTally} {
  currentPayloadTally = tally;
}

PayloadTally::Scope::~Scope() {
  currentPayloadTally = previous_;
}

PayloadTally* PayloadTally::current() {
  return currentPayloadTally;
}

EncodingChunkScope::EncodingChunkScope(
    uint32_t streamOffset,
    uint64_t chunkIndex)
    : previous_{currentChunk} {
  currentChunk = Chunk{.streamOffset = streamOffset, .chunkIndex = chunkIndex};
}

EncodingChunkScope::~EncodingChunkScope() {
  currentChunk = previous_;
}

std::optional<EncodingChunkScope::Chunk> EncodingChunkScope::current() {
  return currentChunk;
}

} // namespace facebook::nimble
