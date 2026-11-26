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
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <functional>
#include <map>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "velox/dwio/nimble/common/Types.h"

namespace facebook::nimble {

class PayloadTally;

/// Writer encoding counters: what each payload handed to compression was, the
/// chunks that replayed a captured layout, and an optional sampled re-encode
/// of chunks with settings turned off. In this diff only FixedBitWidth passes
/// the sink, and the re-encode covers FixedBitWidth typed lanes. A writer
/// creates one only when an option that needs it is set (typed lanes, a typed
/// compression level, or a shadow sample rate), so writers without those
/// options count nothing and publish nothing new. Encodes run in parallel, so
/// every method is thread safe.
///
/// Names: counter "encoding.X" is published as writer runtime stat
/// "nimble.encoding.X" and as FileCloseMetrics field "encoding.X". Every name
/// in a family has the same number of '.'-separated segments, so a reader can
/// split it by position.
/// - Always present:
///   - encoding.typedLanes.<width>.{chunks,rawBytes,storedBytes}: FixedBitWidth
///     typed-lane payloads, with <width> "full" for lanes as wide as their
///     values' type and "narrow" for narrower ones;
///   - encoding.replayedChunks, encoding.replayFallbacks.
/// - Present when the shadow sample rate is nonzero: encoding.shadowSampleSeed,
///   the writer's shadow sampling seed, so that its sampled chunks can be
///   reproduced (see shadowSampleKey()).
/// - Present when non-zero, one triple per payload key:
///   encoding.payload.<Encoding>.<DataType>.<width>.<Compression>.<level>
///   .{chunks,rawBytes,storedBytes}. <width> is "full" or "narrow" for typed
///   lanes as above, "none" for any other payload. <Compression> is the stored
///   compression. <level> is "L<n>", the MetaInternal level the compression
///   attempt used, typed-lanes level included, or "none" when no MetaInternal
///   attempt was made.
/// - Present once a chunk was re-encoded: encoding.shadow.<settings>
///   .FixedBitWidth.{sampledChunks,sampledValues,actualStoredBytes,
///   baselineStoredBytes,actualCpuNs,baselineCpuNs,failures}. <settings> names
///   the settings that changed the chunk: "fixedBitWidthTypedLanes", or
///   "fixedBitWidthTypedLanes+fixedBitWidthTypedLanesOpenzlLevel" when the
///   typed-lanes level compressed one of its lanes. actualStoredBytes and
///   baselineStoredBytes are the chunk's typed-lane payloads' stored bytes as
///   written and in the re-encode. actualCpuNs is the CPU time of those
///   payloads' encodes as written, and baselineCpuNs that of their re-encodes,
///   so their difference is what the settings cost and baselineCpuNs is what
///   sampling costs.
/// Raw bytes are the payload before compression; stored bytes are what was
/// written (compressed when accepted). Encoding headers are excluded.
class EncodingStats {
 public:
  /// A nonzero 'shadowSampleRate' N samples about 1 in N chunks for a
  /// re-encode into scratch memory (see ShadowChunk), keyed with
  /// 'shadowSampleSeed' (see shadowSampleKey()).
  explicit EncodingStats(
      uint32_t shadowSampleRate = 0,
      uint64_t shadowSampleSeed = 0);

  /// What a payload handed to compression was.
  struct Payload {
    EncodingType encodingType{EncodingType::Trivial};
    DataType dataType{DataType::Undefined};
    /// The compression stored.
    CompressionType compressionType{CompressionType::Uncompressed};
    /// The MetaInternal level of the compression attempt; 0 when none was
    /// made with MetaInternal.
    int32_t level{0};
    /// FixedBitWidth typed lanes as wide as their values' type.
    bool fullWidth{false};
    /// The encoding's typed compression level set 'level': a typed payload
    /// (DataType other than Undefined) compressed with MetaInternal, with a
    /// level override listed for its encoding. Not part of the counter key.
    bool typedLevel{false};
  };

  /// Counts one payload of 'rawBytes' bytes stored in 'storedBytes'.
  void
  addPayload(const Payload& payload, uint64_t rawBytes, uint64_t storedBytes);

  /// Adds the payloads counted in 'tally', as addPayload() would have.
  void addTally(const PayloadTally& tally);

  /// Counts a chunk encoded by replaying a captured layout.
  void addReplayedChunk() {
    replayedChunks_.fetch_add(1, std::memory_order_relaxed);
  }

  /// Counts a chunk whose replay failed and that was encoded again with a
  /// fresh selection.
  void addReplayFallback() {
    replayFallbacks_.fetch_add(1, std::memory_order_relaxed);
  }

  uint32_t shadowSampleRate() const {
    return shadowSampleRate_;
  }

  /// The seed of the shadow sampling key: one random value per writer, so
  /// that each file samples its own chunks.
  uint64_t shadowSampleSeed() const {
    return shadowSampleSeed_;
  }

  /// Returns the shadow sampling key of the chunk at index 'chunkIndex' in the
  /// stream at 'streamOffset': its position combined with the seed. It doesn't
  /// depend on which thread encodes the chunk, so the same chunks are sampled
  /// under serial and parallel encoding, and it doesn't read the values, so it
  /// costs the same for any chunk. The seed moves the sampled chunks from file
  /// to file. The sampling method is this function alone.
  uint64_t shadowSampleKey(uint32_t streamOffset, uint64_t chunkIndex) const;

  /// Returns whether the chunk with shadow key 'key' is sampled: when its hash
  /// is 0 modulo the sample rate. Only called with a nonzero rate.
  bool sampleShadow(uint64_t key) const;

  /// A sampled chunk's FixedBitWidth typed-lane payloads, as written and as
  /// re-encoded with typed lanes and the typed-lanes level off. The writer
  /// keeps one while it encodes a sampled chunk, as this thread's current
  /// one, so that lanes nested under another encoding are measured with their
  /// chunk; FixedBitWidth re-encodes each of its lane payloads into it, and
  /// the writer then counts it (shadowEncode()).
  class ShadowChunk {
   public:
    ShadowChunk();
    ~ShadowChunk();

    ShadowChunk(const ShadowChunk&) = delete;
    ShadowChunk& operator=(const ShadowChunk&) = delete;
    ShadowChunk(ShadowChunk&&) = delete;
    ShadowChunk& operator=(ShadowChunk&&) = delete;

    /// The sampled chunk this thread is encoding, if any.
    static ShadowChunk* current();

    /// This thread's CPU time, read by an encode before it starts, for
    /// reEncode().
    static uint64_t threadCpuNanos();

    /// Counts a typed-lane payload of the chunk, 'payload' stored in
    /// 'storedBytes' by an encode that started at 'startCpuNs', and runs
    /// 'encode', which returns the payload's stored bytes with typed lanes and
    /// the typed-lanes level off. Measurement only: an 'encode' that throws
    /// fails the chunk and never fails the write.
    void reEncode(
        const Payload& payload,
        uint64_t storedBytes,
        uint64_t startCpuNs,
        const std::function<uint64_t()>& encode);

   private:
    friend class EncodingStats;

    ShadowChunk* const previous_;
    uint64_t payloads_{0};
    bool typedLevel_{false};
    bool failed_{false};
    uint64_t actualStoredBytes_{0};
    uint64_t baselineStoredBytes_{0};
    uint64_t actualCpuNs_{0};
    uint64_t baselineCpuNs_{0};
  };

  /// Counts 'chunk', a sampled chunk of 'values' values, under
  /// encoding.shadow.<settings>.FixedBitWidth. A chunk without typed-lane
  /// payloads isn't counted; a failed one counts its CPU times and a failure.
  void shadowEncode(uint64_t values, const ShadowChunk& chunk);

  enum class Unit : uint8_t { kNone, kBytes, kNanos };

  struct Counter {
    std::string name;
    uint64_t value;
    Unit unit;
  };

  /// Returns all counters (see the class comment).
  std::vector<Counter> counters() const;

 private:
  friend class PayloadTally;

  struct Slot {
    uint64_t chunks{0};
    uint64_t rawBytes{0};
    uint64_t storedBytes{0};
  };

  struct ShadowSlot {
    uint64_t sampledChunks{0};
    uint64_t sampledValues{0};
    uint64_t actualStoredBytes{0};
    uint64_t baselineStoredBytes{0};
    uint64_t actualCpuNs{0};
    uint64_t baselineCpuNs{0};
    uint64_t failures{0};
  };

  static uint64_t payloadKey(const Payload& payload);

  const uint32_t shadowSampleRate_;
  const uint64_t shadowSampleSeed_;

  mutable std::mutex mutex_;
  // Keyed by payloadKey(); ordered, so counters come out in a stable order.
  std::map<uint64_t, std::pair<Payload, Slot>> payloads_;
  // Keyed by settings and encoding; ordered for the same reason.
  std::map<std::pair<std::string, EncodingType>, ShadowSlot> shadows_;

  std::atomic<uint64_t> replayedChunks_{0};
  std::atomic<uint64_t> replayFallbacks_{0};
};

/// Counts payloads for an EncodingStats without a lock, atomic or allocation:
/// the writer gives each encode task its own, makes it the task's thread's
/// current one (Scope), and folds it into the EncodingStats (fold()) once the
/// task's encodes are joined. A fixed table keyed by the payload key, so the
/// counters come out as if every payload were counted with addPayload(); a
/// payload that finds the table full is counted with addPayload().
class PayloadTally {
 public:
  explicit PayloadTally(EncodingStats& stats) : stats_{&stats} {}

  /// Counts one payload of 'rawBytes' bytes stored in 'storedBytes'.
  void add(
      const EncodingStats::Payload& payload,
      uint64_t rawBytes,
      uint64_t storedBytes);

  /// Adds the counted payloads to the EncodingStats and clears the tally.
  void fold();

  /// The payloads counted with addPayload() because the table was full.
  uint64_t overflows() const {
    return overflows_;
  }

  /// Makes 'tally' this thread's current one while it lives; a null 'tally'
  /// makes none current.
  class Scope {
   public:
    explicit Scope(PayloadTally* tally);
    ~Scope();

    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
    Scope(Scope&&) = delete;
    Scope& operator=(Scope&&) = delete;

   private:
    PayloadTally* const previous_;
  };

  /// The tally of the encode task running on this thread, if any.
  static PayloadTally* current();

 private:
  friend class EncodingStats;

  static constexpr size_t kSlots{64};
  // At most this many keys are kept, so that probes stay short.
  static constexpr size_t kMaxKeys{48};
  static_assert((kSlots & (kSlots - 1)) == 0 && kMaxKeys < kSlots);

  struct Entry {
    uint64_t key{0};
    EncodingStats::Payload payload;
    uint64_t chunks{0};
    uint64_t rawBytes{0};
    uint64_t storedBytes{0};
    bool used{false};
  };

  EncodingStats* stats_;
  std::array<Entry, kSlots> entries_{};
  size_t keys_{0};
  uint64_t overflows_{0};
};

/// Names the stream chunk this thread is encoding, by its stream offset and
/// its index in the stream, for the shadow sampling key
/// (EncodingStats::shadowSampleKey()). The writer opens one around each chunk
/// encode when shadow sampling is on.
class EncodingChunkScope {
 public:
  struct Chunk {
    uint32_t streamOffset;
    uint64_t chunkIndex;

    bool operator==(const Chunk&) const = default;
  };

  EncodingChunkScope(uint32_t streamOffset, uint64_t chunkIndex);
  ~EncodingChunkScope();

  EncodingChunkScope(const EncodingChunkScope&) = delete;
  EncodingChunkScope& operator=(const EncodingChunkScope&) = delete;
  EncodingChunkScope(EncodingChunkScope&&) = delete;
  EncodingChunkScope& operator=(EncodingChunkScope&&) = delete;

  /// Returns the chunk being encoded on this thread, if any.
  static std::optional<Chunk> current();

 private:
  const std::optional<Chunk> previous_;
};

} // namespace facebook::nimble
