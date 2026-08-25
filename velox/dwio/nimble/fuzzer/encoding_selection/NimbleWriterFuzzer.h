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
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/type/Type.h"
#include "velox/vector/BaseVector.h"

namespace facebook::nimble::fuzzer {

/// Tallies how often each candidate was offered and actually selected across
/// the unfiltered and forced phases. The forced policy silently falls back to
/// Trivial whenever EncodingSizeEstimation rejects a stream, so those
/// fallbacks are kept separate from an unfiltered policy simply choosing a
/// different compatible candidate.
///
/// Counts are per encoded chunk, not per stream: chunking splits one stream
/// into many chunks, each of which carries its own encoding tree.
struct EncodingCoverage {
  /// Files whose policy candidate set contained this encoding.
  uint64_t numFilesOffered{0};

  /// Files where at least one chunk used this encoding.
  uint64_t numFilesApplied{0};

  /// Unfiltered files where at least one chunk used this encoding.
  uint64_t numUnfilteredFilesApplied{0};

  /// Chunks whose top-level encoding is this one.
  uint64_t numChunksApplied{0};

  /// Files written in the repair phase with only this encoding requested.
  uint64_t numForcedFilesWritten{0};

  /// Chunks in forced files that fell back to another encoding.
  uint64_t numForcedChunksFellBack{0};
};

/// A requested encoding paired with the Nimble DataType of the stream it was
/// requested for. Encoding compatibility is defined on DataType, so this is the
/// granularity at which "was it exercised" can actually be answered.
using EncodingPair = std::pair<DataType, EncodingType>;

/// Chunk-level tally for one EncodingPair. EncodingCoverage cannot stand in for
/// this: it is keyed on the encoding alone, so an encoding applied to Int64
/// marks itself covered while every other type it accepts goes untested.
struct PairCoverage {
  /// Chunks of this DataType whose top-level encoding is the requested one.
  uint64_t numChunksApplied{0};

  /// Chunks in forced files that fell back to another encoding.
  uint64_t numForcedChunksFellBack{0};

  /// Files whose policy offered this encoding to a stream of this DataType.
  /// This, not the chunk counts, is how well sampled the pair is: chunk counts
  /// scale with
  /// maxStreamChunkRawSize and with how many nested streams a schema carries,
  /// so a single unlucky final-iteration schema can run up dozens of chunks
  /// for one DataType and make a one-draw pair look well established.
  uint64_t numFilesSeen{0};
};

/// Reader paths a written file is verified through. Nimble has two independent
/// encoding-deserialization tables and the selective reader picks between them
/// at runtime, so an encoding can round-trip on one path and fail on another.
/// Adding ALP and Fsst to the writer's candidate set previously required
/// repairing both legacy tables by hand (D114295784); covering all four here
/// turns that class of gap into a fuzzer failure.
enum class ReaderPath {
  /// nimble::VeloxReader with the default legacy::EncodingFactory.
  kLegacyFactory,

  /// nimble::VeloxReader with the non-legacy EncodingFactory.
  kDefaultFactory,

  /// Selective reader, legacy::LegacyEncodingTrait visitor dispatch.
  kSelectiveLegacyDispatch,

  /// Selective reader, DefaultEncodingTrait visitor dispatch.
  kSelectiveDefaultDispatch,
};

/// Every reader path a file is checked through, in verification order.
inline constexpr std::array<ReaderPath, 4> kAllReaderPaths = {
    ReaderPath::kLegacyFactory,
    ReaderPath::kDefaultFactory,
    ReaderPath::kSelectiveLegacyDispatch,
    ReaderPath::kSelectiveDefaultDispatch,
};

std::string_view toString(ReaderPath readerPath);

/// Result of one runFixed round.
enum class WriteOutcome {
  /// The requested encoding was applied to at least one chunk, and every
  /// reader path decoded the file back to the input.
  kApplied,

  /// The file was written and verified, but no chunk used the requested
  /// encoding.
  kNotApplied,
};

/// Writable encodings the fuzzer requests. The list is derived from
/// ManualEncodingSelectionPolicyFactory::possibleEncodings(), which is also the
/// source of truth for the 'encodings:' key of
/// nimble.encoding_selection_config.
std::vector<EncodingType> allCandidateEncodings();

/// Number of unfiltered random-policy files written before coverage repair.
inline constexpr uint32_t kNumUnfilteredRounds = 10;

/// Whether an encoding may only be requested for a stream that is not REAL or
/// DOUBLE. SimdForBitpack and Huffman can be selected for a floating-point
/// stream because their write-side gate tests physicalType (uint32_t/uint64_t,
/// hence integral). DeltaBlock's direct gate tests the logical type, but a
/// Nullable float's nested policy sees the physical type and can select it
/// there. Every read-side gate rejects all three for float and double,
/// producing a file no selective reader can decode. See T283330065.
///
/// The restriction is per stream, not per schema: a schema holding one REAL
/// column alongside integer columns still exercises these encodings on the
/// integer columns.
bool isIntegralOnlyEncoding(EncodingType encodingType);

/// Whether EncodingSizeEstimation's *type* gate admits 'encodingType' for a
/// stream of 'dataType', independent of the data itself.
///
/// The estimator collapses two very different rejections into one
/// std::nullopt: "this encoding can never encode this type" (permanent) and
/// "this encoding does not suit these particular values" (data-dependent).
/// Only the first is knowable from the type alone, and it is what separates a
/// coverage hole from a combination that was never applicable. This predicate
/// recovers it by mirroring the `if constexpr` gates and the
/// numeric/bool/string dispatch in EncodingSizeEstimation.
///
/// It mirrors the *write* side only: SimdForBitpack and Huffman are reported
/// compatible with Float and Double because that is what the write-side gate
/// does -- it tests physicalType, which is integral for floats -- even though
/// no reader can decode the result (T283330065). Callers that also care about
/// readability must exclude those separately. Nothing pins this mirror to
/// EncodingSizeEstimation; it has drifted once (T283801877).
bool isTypeCompatible(EncodingType encodingType, DataType dataType);

/// Knobs for a fuzzer run. Everything else is derived from the seed so a
/// failing run reproduces from the logged seed alone.
struct NimbleWriterFuzzerOptions {
  uint64_t seed{0};

  /// Maximum nesting depth of the generated schema.
  int32_t maxSchemaDepth{3};

  /// Rows per written batch.
  uint32_t batchSize{200};

  /// Batches written per file.
  uint32_t numBatches{3};

  /// Whether to randomize compression, chunking and flush configuration per
  /// file. Deterministic callers turn this off so a run depends only on the
  /// data and the requested encoding, not on the writer-config draw order.
  bool randomizeWriterConfig{true};
};

/// Differential writer fuzzer. Each run generates one random schema and
/// dataset, first writes kNumUnfilteredRounds files whose streams and nested
/// nodes choose naturally from the default random-policy candidates, then
/// writes one forced file for each candidate not observed in that phase. It
/// decodes every file and asserts it reproduces the input exactly. Because all
/// files in a run hold the same data, agreement across encodings is checked
/// implicitly by comparing each against the common input.
///
/// The question is whether the bytes a given encoding produced are correct, so
/// the loop decodes each file once. runFixed() takes every ReaderPath instead,
/// which is what EncodingDispatchConsistencyTest uses to pin the dispatch
/// tables.
class NimbleWriterFuzzer {
 public:
  NimbleWriterFuzzer(
      NimbleWriterFuzzerOptions options,
      velox::memory::MemoryPool& rootPool);

  ~NimbleWriterFuzzer();

  /// Runs kNumUnfilteredRounds random-selection rounds, records which candidate
  /// encodings they used, then runs one forced repair round for each missing
  /// candidate. One reader path is selected deterministically from the seed,
  /// so repeated iterations cover all paths without decoding each file more
  /// than once. Throws NimbleInternalError on the first mismatch, with the
  /// seed, selection mode and reader path in the message.
  void run();

  /// Runs one write/read/verify round over caller-supplied data instead of
  /// fuzzed input, requesting a single encoding, and reads the result back
  /// through every ReaderPath. Lets a deterministic test drive the write path
  /// the fuzzer uses while also pinning every reader dispatch table.
  WriteOutcome runFixed(
      const std::vector<velox::VectorPtr>& batches,
      EncodingType encodingType);

  /// Decodes every chunk of 'file' at the Encoding level and compares the
  /// values against 'column' directly, with no reader constructed. This is the
  /// write-path check: a mismatch is the encoder's, and it names the chunk and
  /// index rather than a row of a reassembled RowVector. It also pins the
  /// DataType and row count each chunk declares, which a vector comparison
  /// structurally cannot see.
  ///
  /// Requires a single flat scalar column with no nulls, because that is the
  /// only shape whose stream-to-column mapping is unambiguous without
  /// reimplementing FieldWriter's decomposition. Returns false when the input
  /// does not qualify, so callers can fall back to reader-based verification.
  bool verifyEncodedColumn(
      const std::string& file,
      const velox::VectorPtr& batch,
      EncodingType encodingType);

  /// Advances the seed and reader-path rotation for the next run().
  void reSeed();

  uint64_t seed() const {
    return options_.seed;
  }

  /// Candidate encoding to outcome tally, accumulated across every run().
  const std::map<EncodingType, EncodingCoverage>& coverage() const {
    return coverage_;
  }

  /// Per (DataType, encoding) chunk tally, accumulated across every run().
  const std::map<EncodingPair, PairCoverage>& pairCoverage() const {
    return pairCoverage_;
  }

  uint64_t numUnfilteredFilesWritten() const {
    return numUnfilteredFilesWritten_;
  }

  /// Logs the coverage tally with never-applied encodings first, so a candidate
  /// that exercises nothing is the first thing visible in a CI log.
  void logCoverage() const;

  /// Logs, per DataType, which encodings were applied to it, which were
  /// requested but never landed, and which its type can never accept. This is
  /// the view that shows a hole inside an encoding that logCoverage() reports
  /// as covered.
  void logPairCoverage() const;

  /// Logs the physical chunk and metadata shapes validated by the run.
  void logChunkStatsCoverage() const;

  /// Candidate encodings that were never applied to a single chunk, either
  /// because no iteration drew a schema they accept or because every stream
  /// fell back to Trivial. A non-empty result means the run verified fewer
  /// encodings than its candidate list claims, which no round-trip assertion
  /// can detect on its own.
  std::vector<EncodingType> unappliedEncodings() const;

  /// EncodingPairs the run should have covered but did not. The expected set is
  /// built from the DataTypes the run actually produced, so it can never demand
  /// a type no schema carried, crossed with the encodings whose acceptance
  /// depends only on that type.
  ///
  /// Encodings with a data precondition -- Constant needs a single-valued
  /// stream, DeltaBlock a non-decreasing one, Huffman an alphabet of at least
  /// two values -- are reported by logPairCoverage() but excluded here.
  /// Whether they land is a property of the data draw, so demanding them would
  /// gate CI on the draw rather than on a regression.
  ///
  /// Pairs offered in too few independently seeded files are excluded for the
  /// same reason; a handful of opportunities cannot support a conclusion
  /// either way.
  std::vector<EncodingPair> unappliedPairs() const;

  /// Names of required chunk-stats metadata shapes not observed by the run.
  std::vector<std::string_view> uncoveredChunkStatsShapes() const;

 private:
  // Counts physical stream and metadata shapes validated during the run.
  struct ChunkStatsVerificationCoverage {
    // Counts files whose chunk stats were enabled or disabled.
    uint64_t numIndexedFiles{0};
    uint64_t numUnindexedFiles{0};
    // Counts indexed groups and stripes.
    uint64_t numStripeGroups{0};
    uint64_t numStripes{0};
    // Counts present scalar-value streams, structural streams emitted for
    // complex types, and streams split into multiple chunks.
    uint64_t numScalarStreams{0};
    uint64_t numStructuralStreams{0};
    uint64_t numMultiChunkStreams{0};
    // Counts chunk positions within their streams.
    uint64_t numFirstChunks{0};
    uint64_t numMiddleChunks{0};
    uint64_t numFinalChunks{0};
    // Counts decoded null distributions.
    uint64_t numZeroNullChunks{0};
    uint64_t numPartiallyNullChunks{0};
    uint64_t numFullyNullChunks{0};
    // Counts physical chunk-wrapper compression states.
    uint64_t numCompressedChunks{0};
    uint64_t numUncompressedChunks{0};
  };

  // Writes 'batches', returning the file bytes. When 'encodingType' is set,
  // that is the only candidate offered to the random policy; otherwise the
  // policy chooses from its full default set. The write type is taken from
  // batches[0].
  std::string writeFile(
      const std::vector<velox::VectorPtr>& batches,
      std::optional<EncodingType> encodingType,
      uint64_t iterationSeed);

  // Validates the chunk-stats section against independently parsed and decoded
  // physical chunks, and records which metadata shapes were exercised.
  void verifyChunkStatsMetadata(
      const std::string& file,
      bool chunkStatsEnabled);

  // Verifies file-level column statistics (value count, null count, min, max)
  // against the data that was actually written.
  void verifyColumnStatistics(
      const std::string& file,
      const velox::RowTypePtr& schema,
      const std::vector<velox::VectorPtr>& batches);

  // Reads 'file' through 'readerPath' and compares every row against
  // 'batches'. Throws on the first difference.
  void readAndVerify(
      const std::string& file,
      const velox::RowTypePtr& schema,
      const std::vector<velox::VectorPtr>& batches,
      std::string_view selectionContext,
      ReaderPath readerPath);

  // Records the encodings an unfiltered random-policy file actually used and
  // returns the candidate encodings observed in that file.
  std::set<EncodingType> recordUnfilteredCoverage(const std::string& file);

  // Records whether a file forced to 'encodingType' applied it or fell back.
  void recordForcedCoverage(const std::string& file, EncodingType encodingType);

  // Runs 'readerPaths' over a freshly written file and records coverage.
  // Returns whether the requested encoding was applied to at least one chunk.
  bool verifyReaderPaths(
      const std::string& file,
      const velox::RowTypePtr& schema,
      const std::vector<velox::VectorPtr>& batches,
      EncodingType encodingType,
      std::span<const ReaderPath> readerPaths);

  // Non-const because reSeed() advances the seed between runs.
  NimbleWriterFuzzerOptions options_;
  velox::memory::MemoryPool& rootPool_;
  const std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::map<EncodingType, EncodingCoverage> coverage_;
  std::map<EncodingPair, PairCoverage> pairCoverage_;
  uint64_t numUnfilteredFilesWritten_{0};
  ChunkStatsVerificationCoverage chunkStatsCoverage_;
  // DataTypes some stream actually carried. Bounds what unappliedPairs() may
  // demand, and is deliberately observed rather than derived from the schema:
  // the writer emits Uint32 length streams, Bool null streams and the Uint16
  // sub-microsecond stream of a TIMESTAMP column, none of which a leaf column
  // type predicts.
  std::set<DataType> observedDataTypes_;
};

} // namespace facebook::nimble::fuzzer
