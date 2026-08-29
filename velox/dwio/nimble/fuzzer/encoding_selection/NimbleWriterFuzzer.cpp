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
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <span>
#include <string>
#include <utility>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/Random.h>
#include <folly/container/F14Set.h>
#include <folly/hash/Hash.h>
#include <glog/logging.h>

#include "velox/common/file/File.h"
#include "velox/common/fuzzer/Utils.h"
#include "velox/common/io/IoStatistics.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Options.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/common/Statistics.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/ChunkHeader.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FixedBitArray.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/tests/RandomEncodingSelectionPolicy.h"
#include "velox/dwio/nimble/index/tests/ClusterIndexTestUtils.h"
#include "velox/dwio/nimble/tablet/TabletReader.h"
#include "velox/dwio/nimble/tablet/tests/TabletTestUtils.h"
#include "velox/dwio/nimble/velox/ChunkedStream.h"
#include "velox/dwio/nimble/velox/SchemaSerialization.h"
#include "velox/dwio/nimble/velox/VeloxReader.h"
#include "velox/dwio/nimble/velox/stats/ColumnStatistics.h"
#include "velox/dwio/nimble/velox/stats/VectorizedStatistics.h"
#include "velox/dwio/nimble/writer/EncodingSelectionPolicyFactory.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::nimble::fuzzer {

namespace {

using ::facebook::velox::RowTypePtr;
using ::facebook::velox::TypePtr;
using ::facebook::velox::VectorPtr;
using ::facebook::velox::fuzzer::FuzzerGenerator;

// Writable encodings that this fuzzer target intentionally does not force.
// The unfiltered random policy also omits these; this list keeps the repair
// phase and coverage gate from adding them back. This is not a global
// unsupported-encoding list.
constexpr auto kExcludedFuzzerCandidateEncodings =
    std::to_array({EncodingType::Huffman, EncodingType::SubIntSplit});

// Scalar types the Nimble writer round-trips with type identity.
// FieldWriter::create dispatches on the physical TypeKind, so DATE, TIME,
// INTERVAL and short DECIMAL write fine but read back as INTEGER/BIGINT, which
// would fail the schema assertion for reasons unrelated to encoding
// correctness. HUGEINT, UNKNOWN and OPAQUE are rejected by the writer outright.
std::vector<TypePtr> supportedScalarTypes() {
  return {
      velox::BOOLEAN(),
      velox::TINYINT(),
      velox::SMALLINT(),
      velox::INTEGER(),
      velox::BIGINT(),
      velox::REAL(),
      velox::DOUBLE(),
      velox::VARCHAR(),
      velox::VARBINARY(),
      velox::TIMESTAMP(),
  };
}

// createFlatMapFieldWriter rejects BOOLEAN, REAL, DOUBLE and TIMESTAMP keys.
// Regular maps accept more, but keeping keys inside the flat-map set lets these
// schemas be reused unchanged if flat-map coverage is added later.
std::vector<TypePtr> supportedMapKeyTypes() {
  return {
      velox::TINYINT(),
      velox::SMALLINT(),
      velox::INTEGER(),
      velox::BIGINT(),
      velox::VARCHAR(),
      velox::VARBINARY(),
  };
}

// Encodings EncodingSizeEstimation::estimateBoolSize can return a size for.
bool isBoolCompatible(EncodingType encodingType) {
  switch (encodingType) {
    case EncodingType::Constant:
    case EncodingType::SparseBool:
    case EncodingType::Trivial:
    case EncodingType::RLE:
      return true;
    default:
      return false;
  }
}

// Encodings EncodingSizeEstimation::estimateStringSize can return a size for.
bool isStringCompatible(EncodingType encodingType) {
  switch (encodingType) {
    case EncodingType::Constant:
    case EncodingType::MainlyConstant:
    case EncodingType::Trivial:
    case EncodingType::Dictionary:
    case EncodingType::RLE:
    case EncodingType::Fsst:
      return true;
    default:
      return false;
  }
}

// Encodings EncodingSizeEstimation::estimateNumericSize can return a size for,
// excluding the ones whose `if constexpr` gate further narrows by numeric kind.
bool isNumericCompatible(EncodingType encodingType) {
  switch (encodingType) {
    case EncodingType::Constant:
    case EncodingType::MainlyConstant:
    case EncodingType::Trivial:
    case EncodingType::FixedBitWidth:
    case EncodingType::Dictionary:
    case EncodingType::RLE:
    case EncodingType::BlockBitPacking:
      return true;
    // Gated on isIntegralType<physicalType>(), which holds for float and
    // double as well since their physical types are uint32_t and uint64_t.
    case EncodingType::PFOR:
    case EncodingType::SimdForBitpack:
    case EncodingType::Huffman:
      return true;
    default:
      return false;
  }
}

bool isFloatingPointDataType(DataType dataType) {
  return dataType == DataType::Float || dataType == DataType::Double;
}

// sizeof(T) for the logical type behind a DataType, which some gates test
// directly.
size_t logicalTypeSize(DataType dataType) {
  switch (dataType) {
    case DataType::Int8:
    case DataType::Uint8:
      return 1;
    case DataType::Int16:
    case DataType::Uint16:
      return 2;
    case DataType::Int32:
    case DataType::Uint32:
    case DataType::Float:
      return 4;
    case DataType::Int64:
    case DataType::Uint64:
    case DataType::Double:
      return 8;
    default:
      // Bool, String and Undefined are handled by isTypeCompatible before it
      // reaches here, so a new DataType landing in this branch is a gap in the
      // mirror rather than a width of zero.
      NIMBLE_UNREACHABLE(
          fmt::format(
              "No logical width for data type {}.", toString(dataType)));
  }
}

// Selection files that must offer a (data type, encoding) pair before its zero
// count is treated as a coverage hole rather than as under-sampling. The ten
// unfiltered rounds use independent policy seeds and deliberately meet this
// threshold; candidates available only during repair accumulate one
// opportunity per outer iteration.
constexpr uint64_t kMinPairFiles = 10;

// Encodings EncodingSizeEstimation can decline on the values rather than on
// the stream type alone: Constant needs a single-valued stream, DeltaBlock a
// per-block non-decreasing one, Huffman an alphabet of at least two values.
// Whether they land is therefore a property of the data draw rather than of
// the schema, so unappliedPairs() reports them but does not demand them.
bool hasDataPrecondition(EncodingType encodingType) {
  return encodingType == EncodingType::Constant ||
      encodingType == EncodingType::DeltaBlock ||
      encodingType == EncodingType::Huffman;
}

// Wraps 'creator' so an integral-only encoding is skipped on floating-point
// streams alone, instead of being dropped from every schema that holds one.
// Excluding it schema-wide would also exclude it from the integer columns
// sharing that schema, which are exactly the streams it is meant to cover, and
// the per-encoding coverage tally cannot show that those pairs went untested.
EncodingSelectionPolicyCreator gateFloatingPointStreams(
    EncodingSelectionPolicyCreator creator,
    EncodingType encodingType,
    std::optional<CompressionOptions> compressionOptions) {
  if (!isIntegralOnlyEncoding(encodingType)) {
    return creator;
  }
  return
      [creator = std::move(creator),
       compressionOptions = std::move(compressionOptions)](DataType dataType) {
        if (dataType != DataType::Float && dataType != DataType::Double) {
          return creator(dataType);
        }
        // Ordinary selection rather than Trivial: falling back to Trivial
        // would leave float streams encoded differently depending on which
        // encoding the iteration happened to request.
        return ManualEncodingSelectionPolicyFactory{
            ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
            compressionOptions}
            .createPolicy(dataType);
      };
}

CompressionType randomCompressionType(FuzzerGenerator& rng) {
  static constexpr std::array<CompressionType, 5> kCompressionTypes = {
      CompressionType::Uncompressed,
      CompressionType::Zstd,
      CompressionType::MetaInternal,
      CompressionType::Lz4,
      CompressionType::OpenZL,
  };
  return kCompressionTypes[folly::Random::rand32(
      kCompressionTypes.size(), rng)];
}

// Applies the layout and experimental knobs the fuzzer is meant to cover. Kept
// separate from encoding selection so a failure can be attributed to either the
// requested encoding or the surrounding configuration.
void randomizeWriterOptions(WriterOptions& options, FuzzerGenerator& rng) {
  options.compressionOptions.compressionType = randomCompressionType(rng);
  // Accept ratio in [0.5, 1.0]. A low floor keeps compression from being
  // rejected so often that the compressed paths go unexercised.
  options.compressionOptions.compressionAcceptRatio =
      0.5f + 0.5f * static_cast<float>(folly::Random::randDouble01(rng));
  options.compressionOptions.zstdCompressionLevel =
      1 + folly::Random::rand32(9, rng);
  options.compressionOptions.internalCompressionLevel =
      1 + folly::Random::rand32(9, rng);
  // Compress even tiny streams; the production minimums would skip nearly
  // everything the fuzzer writes.
  options.compressionOptions.zstdMinCompressionSize = 0;
  options.compressionOptions.lz4MinCompressionSize = 0;
  options.compressionOptions.internalMinCompressionSize = 0;
  options.compressionOptions.openzlMinCompressionSize = 0;

  options.enableChunking = !folly::Random::oneIn(4, rng);
  options.minStreamChunkRawSize = folly::Random::oneIn(2, rng)
      ? 0
      : (uint64_t{1} << folly::Random::rand32(14, rng));
  options.maxStreamChunkRawSize = uint64_t{1}
      << (10 + folly::Random::rand32(12, rng));
  options.enableChunkIndex =
      options.enableChunking && folly::Random::oneIn(2, rng);
  options.enableStreamDeduplication = folly::Random::oneIn(2, rng);
  options.fixedBitWidthUseExactBits = folly::Random::oneIn(2, rng);
  options.allowNestedAlpSelection = folly::Random::oneIn(2, rng);
  // Ratios above 1.0 keep FSST even when it does not shrink the data, so the
  // encoding is actually exercised instead of rejected on size.
  options.fsstCompressionTargetRatio =
      0.2 + 1.0 * folly::Random::randDouble01(rng);
  options.blockBitPackingBlockSize =
      static_cast<uint16_t>(1 << (5 + folly::Random::rand32(6, rng)));

  // Three flush regimes: a small size-based threshold, a chunk on every batch,
  // and seeded probabilistic chunking plus stripe cutting. These only take
  // effect because writeFile passes flushAfterWrite=false; with the helper's
  // default the writer would cut a stripe per batch and shouldFlush would
  // never be consulted. The lambda policies need an rng that outlives them,
  // because VeloxWriter rebuilds the policy on every write().
  switch (folly::Random::rand32(3, rng)) {
    case 1:
      options.flushPolicyFactory = []() {
        return std::make_unique<LambdaFlushPolicy>(
            [](const StripeProgress&) { return false; },
            [](const StripeProgress&) { return true; });
      };
      break;
    case 2: {
      auto policyRng =
          std::make_shared<std::mt19937_64>(folly::Random::rand64(rng));
      options.flushPolicyFactory = [policyRng]() {
        return std::make_unique<LambdaFlushPolicy>(
            [policyRng](const StripeProgress&) {
              return folly::Random::oneIn(20, *policyRng);
            },
            [policyRng](const StripeProgress&) {
              return folly::Random::oneIn(3, *policyRng);
            });
      };
      break;
    }
    default: {
      // A threshold small enough that the fuzzer's modest batches actually
      // cross it, so this regime produces multi-stripe files.
      const uint64_t stripeRawSize = uint64_t{1}
          << (12 + folly::Random::rand32(8, rng));
      options.flushPolicyFactory = [stripeRawSize]() {
        return std::make_unique<StripeRawSizeFlushPolicy>(stripeRawSize);
      };
      break;
    }
  }
}

uint64_t totalRows(const std::vector<VectorPtr>& batches) {
  uint64_t rows = 0;
  for (const auto& batch : batches) {
    rows += batch->size();
  }
  return rows;
}

// Identifies stream IDs whose values have special meaning when validating
// chunk null counts and recording scalar-versus-structural coverage.
struct PhysicalStreamRoles {
  // Streams that store parent validity as Boolean values.
  folly::F14FastSet<uint32_t> nullStreams;
  // Streams that directly store scalar column values.
  folly::F14FastSet<uint32_t> scalarStreams;
};

// Collects the physical stream roles represented by a serialized schema.
void collectPhysicalStreamRoles(const Type& type, PhysicalStreamRoles& roles) {
  switch (type.kind()) {
    case Kind::Scalar:
      roles.scalarStreams.insert(type.asScalar().scalarDescriptor().offset());
      return;
    case Kind::TimestampMicroNano: {
      const auto& timestamp = type.asTimestampMicroNano();
      roles.scalarStreams.insert(timestamp.microsDescriptor().offset());
      roles.scalarStreams.insert(timestamp.nanosDescriptor().offset());
      return;
    }
    case Kind::Row: {
      const auto& row = type.asRow();
      roles.nullStreams.insert(row.nullsDescriptor().offset());
      for (size_t i = 0; i < row.childrenCount(); ++i) {
        collectPhysicalStreamRoles(*row.childAt(i), roles);
      }
      return;
    }
    case Kind::Array:
      collectPhysicalStreamRoles(*type.asArray().elements(), roles);
      return;
    case Kind::ArrayWithOffsets:
      collectPhysicalStreamRoles(*type.asArrayWithOffsets().elements(), roles);
      return;
    case Kind::Map: {
      const auto& map = type.asMap();
      collectPhysicalStreamRoles(*map.keys(), roles);
      collectPhysicalStreamRoles(*map.values(), roles);
      return;
    }
    case Kind::SlidingWindowMap: {
      const auto& map = type.asSlidingWindowMap();
      collectPhysicalStreamRoles(*map.keys(), roles);
      collectPhysicalStreamRoles(*map.values(), roles);
      return;
    }
    case Kind::FlatMap: {
      const auto& flatMap = type.asFlatMap();
      roles.nullStreams.insert(flatMap.nullsDescriptor().offset());
      for (size_t i = 0; i < flatMap.childrenCount(); ++i) {
        collectPhysicalStreamRoles(*flatMap.childAt(i), roles);
      }
      return;
    }
  }
  NIMBLE_UNREACHABLE("Unsupported schema kind: {}.", type.kind());
}

/// Materializes a nullable encoding and counts null entries in the bitmap.
template <typename T>
uint32_t materializedNullCount(
    Encoding& encoding,
    velox::memory::MemoryPool* pool) {
  const auto rowCount = encoding.rowCount();
  Vector<T> values{pool, rowCount};
  Vector<char> nonNulls{
      pool, static_cast<uint32_t>(FixedBitArray::bufferSize(rowCount, 1))};
  nonNulls.zero_out();
  const auto returnedNonNullCount = encoding.materializeNullable(
      rowCount, values.data(), [&]() { return nonNulls.data(); });

  uint32_t bitmapNonNullCount{0};
  for (uint32_t row = 0; row < rowCount; ++row) {
    bitmapNonNullCount += velox::bits::isBitSet(
        reinterpret_cast<const uint8_t*>(nonNulls.data()), row);
  }
  NIMBLE_CHECK_EQ(
      returnedNonNullCount,
      bitmapNonNullCount,
      "Nullable encoding returned a non-null count inconsistent with its bitmap.");
  return rowCount - bitmapNonNullCount;
}

/// Returns the null count for a chunk by decoding its content independently.
/// Null-only streams count false Boolean values; nullable typed streams
/// materialize the bitmap; non-nullable streams return 0.
uint32_t decodedNullCount(
    Encoding& encoding,
    bool isNullStream,
    velox::memory::MemoryPool* pool) {
  if (isNullStream) {
    NIMBLE_CHECK_EQ(
        encoding.dataType(),
        DataType::Bool,
        "Null-only stream must contain Boolean validity values.");
    Vector<bool> nonNulls{pool, encoding.rowCount()};
    encoding.materialize(encoding.rowCount(), nonNulls.data());
    return static_cast<uint32_t>(
        std::count(nonNulls.begin(), nonNulls.end(), false));
  }

  if (!encoding.isNullable()) {
    return 0;
  }

  switch (encoding.dataType()) {
    case DataType::Bool:
      return materializedNullCount<bool>(encoding, pool);
    case DataType::Int8:
      return materializedNullCount<int8_t>(encoding, pool);
    case DataType::Uint8:
      return materializedNullCount<uint8_t>(encoding, pool);
    case DataType::Int16:
      return materializedNullCount<int16_t>(encoding, pool);
    case DataType::Uint16:
      return materializedNullCount<uint16_t>(encoding, pool);
    case DataType::Int32:
      return materializedNullCount<int32_t>(encoding, pool);
    case DataType::Uint32:
      return materializedNullCount<uint32_t>(encoding, pool);
    case DataType::Int64:
      return materializedNullCount<int64_t>(encoding, pool);
    case DataType::Uint64:
      return materializedNullCount<uint64_t>(encoding, pool);
    case DataType::Float:
      return materializedNullCount<float>(encoding, pool);
    case DataType::Double:
      return materializedNullCount<double>(encoding, pool);
    case DataType::String:
      return materializedNullCount<std::string_view>(encoding, pool);
    case DataType::Undefined:
      NIMBLE_UNREACHABLE("Nullable encoding has undefined data type.");
  }
  NIMBLE_UNREACHABLE("Unknown data type: {}.", encoding.dataType());
}

const std::vector<EncodingType>& unfilteredCandidateEncodings() {
  static const auto kCandidates =
      testing::RandomEncodingSelectionPolicyFactory::defaultEncodingChoices();
  return kCandidates;
}

struct FileEncodingObservations {
  std::map<EncodingPair, uint64_t> chunkCounts;
  std::set<DataType> dataTypes;
  std::set<EncodingType> encodingTypes;
};

FileEncodingObservations inspectFileEncodings(
    const std::string& file,
    velox::memory::MemoryPool* pool) {
  FileEncodingObservations observations;
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet =
      TabletReader::create(readFile, pool, test::makeTestTabletOptions(pool));

  for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    const auto stripeIdentifier = tablet->stripeIdentifier(stripe);
    std::vector<uint32_t> streamIdentifiers(
        tablet->streamCount(stripeIdentifier));
    std::iota(streamIdentifiers.begin(), streamIdentifiers.end(), 0);

    auto streams = tablet->load(stripeIdentifier, streamIdentifiers);
    for (auto& stream : streams) {
      if (stream == nullptr) {
        continue;
      }
      InMemoryChunkedStream chunkedStream{*pool, std::move(stream)};
      while (chunkedStream.hasNext()) {
        const auto chunk = chunkedStream.nextChunk();
        // Encoding::Options{} matches the writer because the fuzzer never
        // enables experimentalCompactRowCountEncoding, the only option that
        // affects how an encoding header is parsed.
        const auto layout =
            EncodingLayoutCapture::capture(chunk, Encoding::Options{});
        // NullableEncoding stamps the value DataType into its prefix, and
        // capture() looks through the Nullable wrapper to the data node.
        const auto dataType = EncodingPrefix::dataType(chunk);
        const auto encodingType = layout.encodingType();
        ++observations.chunkCounts[{dataType, encodingType}];
        observations.dataTypes.insert(dataType);
        observations.encodingTypes.insert(encodingType);
      }
    }
  }
  return observations;
}

// Rewrites part of a floating-point column to decimal-like values, the shape
// ALP is built for. ALPEncoding::estimateSize declines only empty input, so
// this is not what gets ALP selected -- it is what gets ALP's normal encoding
// path exercised instead of its exception path, which is where every value
// lands on unshaped VectorFuzzer output. A tail of rows is left untouched so
// the fuzzed NaN and infinity values still reach the encoder.
template <typename T>
void makeDecimalLike(velox::FlatVector<T>* vector, FuzzerGenerator& rng) {
  static constexpr std::array<double, 5> kDivisors = {
      1.0, 10.0, 100.0, 1'000.0, 10'000.0};
  const auto divisor = kDivisors[folly::Random::rand32(kDivisors.size(), rng)];
  const auto rewriteUpTo = static_cast<velox::vector_size_t>(
      vector->size() * (0.8 + 0.2 * folly::Random::randDouble01(rng)));
  for (velox::vector_size_t i = 0; i < rewriteUpTo; ++i) {
    if (vector->isNullAt(i)) {
      continue;
    }
    const auto scaled =
        static_cast<double>(folly::Random::rand32(20'001, rng)) - 10'000.0;
    vector->set(i, static_cast<T>(scaled / divisor));
  }
}

// Rewrites a string column to values built from a small token alphabet, so
// substrings repeat across rows. FsstEncoding::estimateSize cannot decline --
// it returns a plain size -- so selection is never the obstacle. The obstacle
// is inside the encoder: it compresses, then falls back to Trivial when the
// result misses fsstCompressionTargetRatio, which high-entropy VectorFuzzer
// strings always do.
void makeSymbolRich(
    velox::FlatVector<velox::StringView>* vector,
    FuzzerGenerator& rng) {
  static constexpr std::string_view kAlphabet = "abcdefghijklmnopqrstuvwxyz";
  constexpr size_t kNumTokens = 6;
  constexpr size_t kTokenLength = 4;

  std::vector<std::string> tokens;
  tokens.reserve(kNumTokens);
  for (size_t token = 0; token < kNumTokens; ++token) {
    std::string value;
    value.reserve(kTokenLength);
    for (size_t i = 0; i < kTokenLength; ++i) {
      value.push_back(kAlphabet[folly::Random::rand32(kAlphabet.size(), rng)]);
    }
    tokens.push_back(std::move(value));
  }

  for (velox::vector_size_t i = 0; i < vector->size(); ++i) {
    if (vector->isNullAt(i)) {
      continue;
    }
    std::string value;
    const auto numTokens = 3 + folly::Random::rand32(6, rng);
    for (uint32_t token = 0; token < numTokens; ++token) {
      value += tokens[folly::Random::rand32(kNumTokens, rng)];
    }
    vector->set(i, velox::StringView(value));
  }
}

// Collapses the non-null rows of a column to a single repeated value.
// ConstantEncoding requires exactly one distinct value, which fuzzed data never
// produces, so without this the encoding would report zero coverage on every
// run and train readers to ignore the warning. Nulls are left in place: the
// writer encodes them in a separate stream under a Nullable wrapper, so the
// values stream is still constant.
template <typename T>
void makeConstant(velox::FlatVector<T>* vector) {
  velox::vector_size_t sourceRow = 0;
  while (sourceRow < vector->size() && vector->isNullAt(sourceRow)) {
    ++sourceRow;
  }
  if (sourceRow == vector->size()) {
    return;
  }
  // Copy out before writing back: reading through the vector while calling
  // set() on it would depend on FlatVector's buffer-growth behavior.
  if constexpr (std::is_same_v<T, velox::StringView>) {
    const std::string value(vector->valueAt(sourceRow));
    for (velox::vector_size_t i = 0; i < vector->size(); ++i) {
      if (!vector->isNullAt(i)) {
        vector->set(i, velox::StringView(value));
      }
    }
  } else {
    const T value = vector->valueAt(sourceRow);
    for (velox::vector_size_t i = 0; i < vector->size(); ++i) {
      if (!vector->isNullAt(i)) {
        vector->set(i, value);
      }
    }
  }
}

// Rewrites an integer column to a non-decreasing sequence, the one shape
// DeltaBlock accepts. Repeats are drawn in so the result is not a pure ramp,
// which would make the delta stream uniformly one. 'cursor' carries the last
// value emitted for this column across batches: the estimator checks ordering
// per block and chunk boundaries do not follow batch boundaries, so a
// per-batch restart would leave most blocks out of order and the shape
// unusable.
template <typename T>
void makeNonDecreasing(
    velox::FlatVector<T>* vector,
    int64_t& cursor,
    uint64_t totalRows,
    FuzzerGenerator& rng) {
  constexpr int64_t kMinValue = std::numeric_limits<T>::min();
  constexpr int64_t kMaxValue = std::numeric_limits<T>::max();
  // Spend at most half the type's range over the whole file. A fixed step
  // probability exhausts a narrow type long before the rows run out -- TINYINT
  // has 255 values against 600 rows -- and the saturated tail is a run of one
  // repeated value, which is a shape several encoders refuse outright. Capped
  // below 1 so wide types still draw repeats.
  const double stepProbability = std::min(
      0.67,
      (static_cast<double>(kMaxValue) - static_cast<double>(kMinValue)) /
          (2.0 * static_cast<double>(std::max<uint64_t>(totalRows, 1))));
  for (velox::vector_size_t i = 0; i < vector->size(); ++i) {
    if (vector->isNullAt(i)) {
      continue;
    }
    // The range guard is a backstop for a caller passing too few rows;
    // saturating is still non-decreasing, just degenerate.
    if (cursor < kMaxValue &&
        folly::Random::randDouble01(rng) < stepProbability) {
      ++cursor;
    }
    vector->set(i, static_cast<T>(cursor));
  }
}

// The signed integer kinds makeNonDecreasing can rewrite. TIMESTAMP also
// reaches the writer as Int64 but is left alone: its Velox representation is
// not a plain integer.
bool isIntegerKind(velox::TypeKind kind) {
  return kind == velox::TypeKind::TINYINT ||
      kind == velox::TypeKind::SMALLINT || kind == velox::TypeKind::INTEGER ||
      kind == velox::TypeKind::BIGINT;
}

// Lowest value the kind can hold, so a non-decreasing column starts at the
// bottom of its range instead of saturating a few hundred rows in.
int64_t minValueOf(velox::TypeKind kind) {
  if (kind == velox::TypeKind::TINYINT) {
    return std::numeric_limits<int8_t>::min();
  }
  if (kind == velox::TypeKind::SMALLINT) {
    return std::numeric_limits<int16_t>::min();
  }
  if (kind == velox::TypeKind::INTEGER) {
    return std::numeric_limits<int32_t>::min();
  }
  NIMBLE_CHECK(
      kind == velox::TypeKind::BIGINT,
      "Not an integer kind: {}.",
      static_cast<int>(kind));
  return std::numeric_limits<int64_t>::min();
}

// Which layout a column is pushed toward for a whole iteration.
enum class ColumnShape {
  // Left as VectorFuzzer produced it, apart from the type-specific default
  // shaping (decimal-like floats, symbol-rich strings) the encodings need.
  kDefault,
  kConstant,
  kNonDecreasing,
};

// Draws one shape per top-level column. Drawn once per iteration rather than
// per batch because every batch of a file feeds the same streams: a column
// shaped constant in one batch and random in the next produces neither a
// constant stream nor a useful random one.
std::vector<ColumnShape> drawColumnShapes(
    const RowTypePtr& schema,
    FuzzerGenerator& rng) {
  std::vector<ColumnShape> shapes;
  shapes.reserve(schema->size());
  for (auto i = 0; i < schema->size(); ++i) {
    if (folly::Random::oneIn(5, rng)) {
      shapes.push_back(ColumnShape::kConstant);
    } else if (
        isIntegerKind(schema->childAt(i)->kind()) &&
        folly::Random::oneIn(4, rng)) {
      shapes.push_back(ColumnShape::kNonDecreasing);
    } else {
      shapes.push_back(ColumnShape::kDefault);
    }
  }
  return shapes;
}

// Reshapes the top-level scalar columns of a fuzzed batch toward the data
// layouts the encodings target. Nested columns keep their random content, and
// nulls are preserved so the null-handling paths stay covered. Applied to only
// some iterations so purely random data is still exercised.
//
// 'cursors' holds the running value of each non-decreasing column and is
// updated in place, so successive batches continue the sequence.
void applyEncodingFriendlyShapes(
    velox::RowVector* batch,
    const std::vector<ColumnShape>& shapes,
    std::vector<int64_t>& cursors,
    uint64_t totalRows,
    FuzzerGenerator& rng) {
  NIMBLE_CHECK_EQ(
      batch->childrenSize(),
      shapes.size(),
      "Column shapes were drawn for a different schema than this batch.");
  NIMBLE_CHECK_EQ(
      cursors.size(),
      shapes.size(),
      "Every column needs a cursor slot, drawn shape or not.");
  for (auto childIndex = 0; childIndex < batch->childrenSize(); ++childIndex) {
    const auto& child = batch->childAt(childIndex);
    if (child == nullptr || !child->isFlatEncoding()) {
      continue;
    }
    const bool collapseToConstant =
        shapes.at(childIndex) == ColumnShape::kConstant;
    const bool makeMonotonic =
        shapes.at(childIndex) == ColumnShape::kNonDecreasing;
    auto& cursor = cursors.at(childIndex);
    switch (child->typeKind()) {
      case velox::TypeKind::BOOLEAN:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<bool>());
        }
        break;
      case velox::TypeKind::TINYINT:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<int8_t>());
        } else if (makeMonotonic) {
          makeNonDecreasing(
              child->asFlatVector<int8_t>(), cursor, totalRows, rng);
        }
        break;
      case velox::TypeKind::SMALLINT:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<int16_t>());
        } else if (makeMonotonic) {
          makeNonDecreasing(
              child->asFlatVector<int16_t>(), cursor, totalRows, rng);
        }
        break;
      case velox::TypeKind::INTEGER:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<int32_t>());
        } else if (makeMonotonic) {
          makeNonDecreasing(
              child->asFlatVector<int32_t>(), cursor, totalRows, rng);
        }
        break;
      case velox::TypeKind::BIGINT:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<int64_t>());
        } else if (makeMonotonic) {
          makeNonDecreasing(
              child->asFlatVector<int64_t>(), cursor, totalRows, rng);
        }
        break;
      case velox::TypeKind::REAL:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<float>());
        } else {
          makeDecimalLike(child->asFlatVector<float>(), rng);
        }
        break;
      case velox::TypeKind::DOUBLE:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<double>());
        } else {
          makeDecimalLike(child->asFlatVector<double>(), rng);
        }
        break;
      case velox::TypeKind::VARCHAR:
      case velox::TypeKind::VARBINARY:
        if (collapseToConstant) {
          makeConstant(child->asFlatVector<velox::StringView>());
        } else {
          makeSymbolRich(child->asFlatVector<velox::StringView>(), rng);
        }
        break;
      default:
        break;
    }
  }
}

// Value equality for decoded chunk data. Mirrors what the vector comparison
// does for floating point: NaN equals NaN, so a column carrying fuzzed NaNs is
// not reported as a round-trip failure.
template <typename T>
bool decodedValueEquals(const T& lhs, const T& rhs) {
  if constexpr (std::is_floating_point_v<T>) {
    if (std::isnan(lhs) && std::isnan(rhs)) {
      return true;
    }
  }
  return lhs == rhs;
}

// Decodes one chunk and compares it against 'expected' starting at 'cursor'.
template <typename T, typename ExpectedAt>
void compareDecodedChunk(
    Encoding& encoding,
    velox::memory::MemoryPool* pool,
    velox::vector_size_t numExpected,
    const ExpectedAt& expectedAt,
    velox::vector_size_t& cursor,
    EncodingType encodingType,
    uint64_t seed) {
  const auto rowCount = encoding.rowCount();
  // Nimble's Vector, not std::vector: the latter's bool specialization has no
  // usable data() and yields proxy references.
  Vector<T> decoded{pool, rowCount};
  encoding.materialize(rowCount, decoded.data());
  for (uint32_t i = 0; i < rowCount; ++i) {
    NIMBLE_CHECK_LT(
        cursor,
        numExpected,
        "Encoded stream holds more rows than were written (seed {}, encoding {}).",
        seed,
        toString(encodingType));
    NIMBLE_CHECK(
        decodedValueEquals(decoded[i], expectedAt(cursor)),
        "Encoded value mismatch at index {} (seed {}, encoding {}, chunk encoding {}).",
        cursor,
        seed,
        toString(encodingType),
        toString(encoding.encodingType()));
    ++cursor;
  }
}

} // namespace

std::string_view toString(ReaderPath readerPath) {
  switch (readerPath) {
    case ReaderPath::kLegacyFactory:
      return "VeloxReader/legacyFactory";
    case ReaderPath::kDefaultFactory:
      return "VeloxReader/defaultFactory";
    case ReaderPath::kSelectiveLegacyDispatch:
      return "selective/legacyDispatch";
    case ReaderPath::kSelectiveDefaultDispatch:
      return "selective/defaultDispatch";
  }
  NIMBLE_UNREACHABLE("Unknown reader path: {}.", static_cast<int>(readerPath));
}

std::vector<EncodingType> allCandidateEncodings() {
  // The fuzzer's repair phase forces each candidate through
  // `nimble.encoding_selection_config` using the `encodings:` key. Keep this
  // list derived from the same writable-encoding source as that parser, so
  // deprecating an encoding for new writes does not leave the fuzzer forcing a
  // config string production code now rejects.
  auto encodings = ManualEncodingSelectionPolicyFactory::possibleEncodings();
  for (const auto encodingType : kExcludedFuzzerCandidateEncodings) {
    std::erase(encodings, encodingType);
  }
  return encodings;
}

bool isTypeCompatible(EncodingType encodingType, DataType dataType) {
  if (dataType == DataType::Undefined) {
    return false;
  }
  if (dataType == DataType::Bool) {
    return isBoolCompatible(encodingType);
  }
  if (dataType == DataType::String) {
    return isStringCompatible(encodingType);
  }

  // Gated on isFloatingPointType<T>() / isIntegralType<T>(), which test the
  // logical type, so these two split cleanly.
  if (encodingType == EncodingType::ALP) {
    return isFloatingPointDataType(dataType);
  }
  if (encodingType == EncodingType::DeltaBlock) {
    return !isFloatingPointDataType(dataType);
  }
  // Varint's gate is isIntegralType<physicalType>() && sizeof(T) >= 4, so the
  // narrow integers are excluded while float and double are not -- their
  // physical types are integral and already 4 or 8 bytes wide.
  if (encodingType == EncodingType::Varint) {
    return logicalTypeSize(dataType) >= 4;
  }
  return isNumericCompatible(encodingType);
}

bool isIntegralOnlyEncoding(EncodingType encodingType) {
  return encodingType == EncodingType::DeltaBlock ||
      encodingType == EncodingType::PFOR ||
      encodingType == EncodingType::SimdForBitpack ||
      encodingType == EncodingType::Huffman;
}

NimbleWriterFuzzer::NimbleWriterFuzzer(
    NimbleWriterFuzzerOptions options,
    velox::memory::MemoryPool& rootPool)
    : options_{std::move(options)},
      rootPool_{rootPool},
      leafPool_{rootPool.addLeafChild("nimble_writer_fuzzer")} {}

NimbleWriterFuzzer::~NimbleWriterFuzzer() = default;

void NimbleWriterFuzzer::reSeed() {
  const auto nextReaderPath = (options_.seed + 1) % kAllReaderPaths.size();
  const auto randomizedSeed =
      folly::hash::hash_combine(options_.seed, uint64_t{1});
  options_.seed =
      randomizedSeed - randomizedSeed % kAllReaderPaths.size() + nextReaderPath;
}

std::string NimbleWriterFuzzer::writeFile(
    const std::vector<VectorPtr>& batches,
    std::optional<EncodingType> encodingType,
    uint64_t iterationSeed) {
  FuzzerGenerator rng(iterationSeed);
  WriterOptions writerOptions;
  if (options_.randomizeWriterConfig) {
    randomizeWriterOptions(writerOptions, rng);
  }

  // Build the policy through the config-string parser rather than constructing
  // the factory directly, so nimble.encoding_selection_config parsing is
  // covered too. Every candidate encoding is accepted by that parser, which
  // validates against
  // ManualEncodingSelectionPolicyFactory::possibleEncodings().
  const auto selectionConfig = encodingType.has_value()
      ? fmt::format(
            "type:random,seed:{},encodings:{}",
            iterationSeed,
            toString(*encodingType))
      : fmt::format("type:random,seed:{}", iterationSeed);
  auto parsedCreator = createEncodingSelectionPolicyFactory(
      selectionConfig, writerOptions.compressionOptions);
  NIMBLE_CHECK(
      parsedCreator.has_value(),
      "Encoding selection config produced no policy creator for '{}'.",
      selectionConfig);
  EncodingSelectionPolicyCreator policyCreator = std::move(*parsedCreator);

  // The write-side gate admits floating point directly for SimdForBitpack and
  // Huffman. A Nullable float's nested policy can also admit DeltaBlock after
  // the logical type becomes its Uint32/Uint64 physical type.
  writerOptions.encodingSelectionPolicyCreator = encodingType.has_value()
      ? gateFloatingPointStreams(
            std::move(policyCreator),
            *encodingType,
            writerOptions.compressionOptions)
      : std::move(policyCreator);

  // flushAfterWrite=false leaves stripe boundaries to the flush policy; the
  // helper's default would cut a stripe after every batch and make every flush
  // regime identical.
  const bool chunkStatsEnabled = writerOptions.enableChunkIndex;
  auto file = test::createNimbleFile(
      rootPool_, batches, std::move(writerOptions), /*flushAfterWrite=*/false);
  verifyChunkStatsMetadata(file, chunkStatsEnabled);
  auto schema =
      std::dynamic_pointer_cast<const velox::RowType>(batches[0]->type());
  verifyColumnStatistics(file, schema, batches);
  verifySchemaAndStripeGroupConsistency(file, schema);
  return file;
}

void NimbleWriterFuzzer::verifyChunkStatsMetadata(
    const std::string& file,
    bool chunkStatsEnabled) {
  struct PhysicalChunk {
    uint32_t offset;
    uint32_t size;
    CompressionType compressionType;
  };

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = TabletReader::create(
      readFile, leafPool_.get(), test::makeTestTabletOptions(leafPool_.get()));

  auto schemaSection = tablet->loadOptionalSection(std::string(kSchemaSection));
  NIMBLE_CHECK(schemaSection.has_value(), "Schema not found.");
  const auto nimbleSchema =
      SchemaDeserializer::deserialize(schemaSection->content());
  PhysicalStreamRoles roles;
  collectPhysicalStreamRoles(*nimbleSchema, roles);

  if (chunkStatsEnabled) {
    ++chunkStatsCoverage_.numIndexedFiles;
  } else {
    ++chunkStatsCoverage_.numUnindexedFiles;
  }

  folly::F14FastSet<uint32_t> stripeGroups;
  for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    const auto stripeIdentifier = tablet->stripeIdentifier(stripe);
    const auto& chunkStats = stripeIdentifier.chunkStats();
    if (!chunkStatsEnabled) {
      NIMBLE_CHECK_NULL(
          chunkStats,
          "Chunk stats present in stripe {} but chunk index is disabled (seed {}).",
          stripe,
          options_.seed);
    }
    if (!chunkStats) {
      continue;
    }
    ++chunkStatsCoverage_.numStripes;

    index::test::ChunkStatsTestHelper chunkHelper(chunkStats.get());
    stripeGroups.insert(chunkHelper.firstStripe());
    const uint32_t stripeOffset = stripe - chunkHelper.firstStripe();
    NIMBLE_CHECK_LT(stripeOffset, chunkHelper.stripeCount());

    const uint32_t streamCount = tablet->streamCount(stripeIdentifier);
    NIMBLE_CHECK_EQ(streamCount, chunkHelper.streamCount());
    std::vector<uint32_t> streamIdentifiers(streamCount);
    std::iota(streamIdentifiers.begin(), streamIdentifiers.end(), 0);
    auto streams = tablet->load(stripeIdentifier, streamIdentifiers);

    for (uint32_t streamId = 0; streamId < streamCount; ++streamId) {
      auto streamStats = chunkHelper.streamStats(streamId);
      NIMBLE_CHECK_EQ(
          streamStats.chunkCounts.size(), chunkHelper.stripeCount());
      NIMBLE_CHECK_EQ(
          streamStats.chunkRows.size(), streamStats.chunkOffsets.size());
      NIMBLE_CHECK_EQ(
          streamStats.chunkRows.size(), streamStats.chunkNullCounts.size());
      const uint32_t beginChunk =
          stripeOffset == 0 ? 0 : streamStats.chunkCounts[stripeOffset - 1];
      const uint32_t endChunk = streamStats.chunkCounts[stripeOffset];
      NIMBLE_CHECK_LE(beginChunk, endChunk);
      NIMBLE_CHECK_LE(endChunk, streamStats.chunkRows.size());
      const uint32_t chunkCount = endChunk - beginChunk;

      auto& stream = streams[streamId];
      std::string_view rawStream;
      if (stream != nullptr) {
        rawStream = stream->getStream();
      }
      NIMBLE_CHECK_EQ(
          rawStream.size(), tablet->streamSize(stripeIdentifier, streamId));
      NIMBLE_CHECK_LE(rawStream.size(), std::numeric_limits<uint32_t>::max());

      std::vector<PhysicalChunk> physicalChunks;
      uint32_t offset{0};
      while (offset < rawStream.size()) {
        NIMBLE_CHECK_LE(
            kChunkHeaderSize,
            rawStream.size() - offset,
            "Truncated chunk header in stripe {}, stream {}.",
            stripe,
            streamId);
        const char* cursor = rawStream.data() + offset;
        const auto header = readChunkHeader(cursor);
        const uint64_t size = uint64_t{kChunkHeaderSize} + header.length;
        NIMBLE_CHECK_LE(
            size,
            rawStream.size() - offset,
            "Chunk exceeds stripe {}, stream {} boundary.",
            stripe,
            streamId);
        physicalChunks.push_back(
            {offset, static_cast<uint32_t>(size), header.compressionType});
        offset += size;
      }
      NIMBLE_CHECK_EQ(offset, rawStream.size());
      NIMBLE_CHECK_EQ(
          chunkCount,
          physicalChunks.size(),
          "Chunk count mismatch in stripe {}, stream {} (seed {}).",
          stripe,
          streamId,
          options_.seed);

      if (chunkCount == 0) {
        NIMBLE_CHECK_NULL(stream);
        continue;
      }
      NIMBLE_CHECK_NOT_NULL(stream);

      if (roles.scalarStreams.contains(streamId)) {
        ++chunkStatsCoverage_.numScalarStreams;
      } else {
        ++chunkStatsCoverage_.numStructuralStreams;
      }
      if (chunkCount > 1) {
        ++chunkStatsCoverage_.numMultiChunkStreams;
      }

      auto streamIndex = chunkStats->createStreamIndex(
          stripe, streamId, static_cast<uint32_t>(rawStream.size()));
      NIMBLE_CHECK_EQ(streamIndex != nullptr, chunkCount > 1);

      InMemoryChunkedStream chunkedStream{*leafPool_, std::move(stream)};
      Buffer stringBuffer{*leafPool_};
      auto stringBufferFactory = [&stringBuffer](uint32_t size) -> void* {
        return stringBuffer.reserve(size);
      };
      uint32_t previousRowBoundary{0};
      for (uint32_t chunk = 0; chunk < chunkCount; ++chunk) {
        const uint32_t metadataChunk = beginChunk + chunk;
        const auto& physicalChunk = physicalChunks.at(chunk);
        NIMBLE_CHECK_EQ(
            streamStats.chunkOffsets[metadataChunk], physicalChunk.offset);

        const uint32_t nextOffset = chunk + 1 < chunkCount
            ? streamStats.chunkOffsets[metadataChunk + 1]
            : static_cast<uint32_t>(rawStream.size());
        NIMBLE_CHECK_EQ(nextOffset - physicalChunk.offset, physicalChunk.size);

        const uint32_t rowBoundary = streamStats.chunkRows[metadataChunk];
        NIMBLE_CHECK_GT(rowBoundary, previousRowBoundary);
        const uint32_t chunkRows = rowBoundary - previousRowBoundary;
        NIMBLE_CHECK(chunkedStream.hasNext());
        auto encoding = EncodingFactory().create(
            *leafPool_, chunkedStream.nextChunk(), stringBufferFactory);
        NIMBLE_CHECK_NOT_NULL(encoding);
        NIMBLE_CHECK_EQ(
            encoding->rowCount(),
            chunkRows,
            "Chunk row count mismatch in stripe {}, stream {}, chunk {}.",
            stripe,
            streamId,
            chunk);

        const auto expectedNullCount = decodedNullCount(
            *encoding, roles.nullStreams.contains(streamId), leafPool_.get());
        const auto indexedNullCount =
            streamStats.chunkNullCounts[metadataChunk];
        NIMBLE_CHECK_EQ(
            indexedNullCount,
            expectedNullCount,
            "Chunk null count mismatch in stripe {}, stream {}, chunk {} (seed {}).",
            stripe,
            streamId,
            chunk,
            options_.seed);
        NIMBLE_CHECK_EQ(indexedNullCount > 0, expectedNullCount > 0);

        if (indexedNullCount == 0) {
          ++chunkStatsCoverage_.numZeroNullChunks;
        } else if (indexedNullCount == chunkRows) {
          ++chunkStatsCoverage_.numFullyNullChunks;
        } else {
          ++chunkStatsCoverage_.numPartiallyNullChunks;
        }
        if (physicalChunk.compressionType == CompressionType::Uncompressed) {
          ++chunkStatsCoverage_.numUncompressedChunks;
        } else {
          ++chunkStatsCoverage_.numCompressedChunks;
        }
        if (chunk == 0) {
          ++chunkStatsCoverage_.numFirstChunks;
        }
        if (chunk + 1 == chunkCount) {
          ++chunkStatsCoverage_.numFinalChunks;
        }
        if (chunk > 0 && chunk + 1 < chunkCount) {
          ++chunkStatsCoverage_.numMiddleChunks;
        }

        if (streamIndex) {
          const auto first = streamIndex->lookupChunk(previousRowBoundary);
          const auto last = streamIndex->lookupChunk(rowBoundary - 1);
          NIMBLE_CHECK_EQ(first.chunkOffset, physicalChunk.offset);
          NIMBLE_CHECK_EQ(first.chunkSize, physicalChunk.size);
          NIMBLE_CHECK_EQ(first.rowOffset, previousRowBoundary);
          NIMBLE_CHECK_EQ(last.chunkIndex, first.chunkIndex);
          NIMBLE_CHECK_EQ(last.chunkOffset, first.chunkOffset);
          const auto lookupNullCount =
              streamIndex->chunkNullCount(first.chunkIndex);
          NIMBLE_CHECK(
              lookupNullCount.has_value(),
              "Chunk {} of stripe {}, stream {} has no indexed null count.",
              chunk,
              stripe,
              streamId);
          NIMBLE_CHECK_EQ(*lookupNullCount, indexedNullCount);
          if (chunk + 1 < chunkCount) {
            const auto next = streamIndex->lookupChunk(rowBoundary);
            NIMBLE_CHECK_NE(next.chunkIndex, first.chunkIndex);
            NIMBLE_CHECK_EQ(next.chunkOffset, physicalChunks[chunk + 1].offset);
          }
        }
        previousRowBoundary = rowBoundary;
      }
      NIMBLE_CHECK(!chunkedStream.hasNext());
      if (streamIndex) {
        NIMBLE_CHECK_EQ(streamIndex->rowCount(), previousRowBoundary);
      }
    }
  }
  chunkStatsCoverage_.numStripeGroups += stripeGroups.size();
}

namespace {

/// Accumulated expected stats per schema node, built across all batches.
/// Converts to dwio::common format so the comparison exercises the same
/// conversion path that the filtering logic consumes.
struct ExpectedNodeStats {
  uint64_t valueCount{0};
  uint64_t nullCount{0};
  std::optional<int64_t> integralMin;
  std::optional<int64_t> integralMax;
  std::optional<double> floatingMin;
  std::optional<double> floatingMax;
  std::optional<std::string> stringMin;
  std::optional<std::string> stringMax;

  std::unique_ptr<velox::dwio::common::ColumnStatistics> toCommonStatistics()
      const {
    bool hasNull = nullCount > 0;
    if (integralMin.has_value()) {
      return std::make_unique<velox::dwio::common::IntegerColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          integralMin,
          integralMax,
          std::nullopt);
    }
    if (floatingMin.has_value()) {
      return std::make_unique<velox::dwio::common::DoubleColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          floatingMin,
          floatingMax,
          std::nullopt);
    }
    if (stringMin.has_value()) {
      return std::make_unique<velox::dwio::common::StringColumnStatistics>(
          valueCount,
          hasNull,
          std::nullopt,
          std::nullopt,
          stringMin,
          stringMax,
          std::nullopt);
    }
    return std::make_unique<velox::dwio::common::ColumnStatistics>(
        valueCount, hasNull, std::nullopt, std::nullopt);
  }
};

/// Accumulates expected stats for one batch into the per-node vector.
/// 'validRows' are row indices where all ancestors are non-null.
void accumulateNodeStats(
    const VectorPtr& vector,
    const std::vector<velox::vector_size_t>& validRows,
    std::vector<ExpectedNodeStats>& nodeStats,
    uint32_t& nodeId) {
  if (nodeId >= nodeStats.size()) {
    return;
  }
  auto& stats = nodeStats[nodeId];
  ++nodeId;

  std::vector<velox::vector_size_t> nonNullRows;
  nonNullRows.reserve(validRows.size());
  for (const auto row : validRows) {
    if (vector->isNullAt(row)) {
      ++stats.nullCount;
    } else {
      nonNullRows.push_back(row);
    }
  }
  stats.valueCount += nonNullRows.size();

  if (vector->typeKind() == velox::TypeKind::ROW) {
    auto rowVector = vector->as<velox::RowVector>();
    for (velox::column_index_t col = 0; col < rowVector->childrenSize();
         ++col) {
      accumulateNodeStats(
          rowVector->childAt(col), nonNullRows, nodeStats, nodeId);
    }
    return;
  }

  if (vector->typeKind() == velox::TypeKind::ARRAY) {
    auto arrayVector = vector->as<velox::ArrayVector>();
    std::vector<velox::vector_size_t> elementRows;
    for (const auto row : nonNullRows) {
      auto offset = arrayVector->offsetAt(row);
      auto size = arrayVector->sizeAt(row);
      for (velox::vector_size_t i = 0; i < size; ++i) {
        elementRows.push_back(offset + i);
      }
    }
    accumulateNodeStats(
        arrayVector->elements(), elementRows, nodeStats, nodeId);
    return;
  }

  if (vector->typeKind() == velox::TypeKind::MAP) {
    auto mapVector = vector->as<velox::MapVector>();
    std::vector<velox::vector_size_t> entryRows;
    for (const auto row : nonNullRows) {
      auto offset = mapVector->offsetAt(row);
      auto size = mapVector->sizeAt(row);
      for (velox::vector_size_t i = 0; i < size; ++i) {
        entryRows.push_back(offset + i);
      }
    }
    accumulateNodeStats(mapVector->mapKeys(), entryRows, nodeStats, nodeId);
    accumulateNodeStats(mapVector->mapValues(), entryRows, nodeStats, nodeId);
    return;
  }

  if (!vector->type()->isPrimitiveType()) {
    return;
  }

  for (const auto row : nonNullRows) {
    switch (vector->typeKind()) {
      case velox::TypeKind::BOOLEAN:
      case velox::TypeKind::TINYINT:
      case velox::TypeKind::SMALLINT:
      case velox::TypeKind::INTEGER:
      case velox::TypeKind::BIGINT:
      case velox::TypeKind::TIMESTAMP: {
        int64_t value;
        switch (vector->typeKind()) {
          case velox::TypeKind::BOOLEAN:
            value =
                vector->as<velox::SimpleVector<bool>>()->valueAt(row) ? 1 : 0;
            break;
          case velox::TypeKind::TINYINT:
            value = vector->as<velox::SimpleVector<int8_t>>()->valueAt(row);
            break;
          case velox::TypeKind::SMALLINT:
            value = vector->as<velox::SimpleVector<int16_t>>()->valueAt(row);
            break;
          case velox::TypeKind::INTEGER:
            value = vector->as<velox::SimpleVector<int32_t>>()->valueAt(row);
            break;
          case velox::TypeKind::BIGINT:
            value = vector->as<velox::SimpleVector<int64_t>>()->valueAt(row);
            break;
          case velox::TypeKind::TIMESTAMP:
            value = vector->as<velox::SimpleVector<velox::Timestamp>>()
                        ->valueAt(row)
                        .toMicros();
            break;
          default:
            NIMBLE_UNREACHABLE("Unexpected integral type.");
        }
        stats.integralMin = stats.integralMin.has_value()
            ? std::min(*stats.integralMin, value)
            : value;
        stats.integralMax = stats.integralMax.has_value()
            ? std::max(*stats.integralMax, value)
            : value;
        break;
      }
      case velox::TypeKind::REAL: {
        double value = static_cast<double>(
            vector->as<velox::SimpleVector<float>>()->valueAt(row));
        stats.floatingMin = stats.floatingMin.has_value()
            ? std::min(*stats.floatingMin, value)
            : value;
        stats.floatingMax = stats.floatingMax.has_value()
            ? std::max(*stats.floatingMax, value)
            : value;
        break;
      }
      case velox::TypeKind::DOUBLE: {
        double value = vector->as<velox::SimpleVector<double>>()->valueAt(row);
        stats.floatingMin = stats.floatingMin.has_value()
            ? std::min(*stats.floatingMin, value)
            : value;
        stats.floatingMax = stats.floatingMax.has_value()
            ? std::max(*stats.floatingMax, value)
            : value;
        break;
      }
      case velox::TypeKind::VARCHAR:
      case velox::TypeKind::VARBINARY: {
        auto sv =
            vector->as<velox::SimpleVector<velox::StringView>>()->valueAt(row);
        std::string value(sv.data(), sv.size());
        stats.stringMin = stats.stringMin.has_value()
            ? std::min(*stats.stringMin, value)
            : value;
        stats.stringMax = stats.stringMax.has_value()
            ? std::max(*stats.stringMax, value)
            : value;
        break;
      }
      default:
        break;
    }
  }
}

} // namespace

void NimbleWriterFuzzer::verifyColumnStatistics(
    const std::string& file,
    const RowTypePtr& schema,
    const std::vector<VectorPtr>& batches) {
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = TabletReader::create(
      readFile, leafPool_.get(), test::makeTestTabletOptions(leafPool_.get()));

  auto statsSection =
      tablet->loadOptionalSection(std::string(kVectorizedStatsSection));
  if (!statsSection.has_value()) {
    return;
  }

  auto vectorizedStats =
      VectorizedFileStats::deserialize(statsSection->content(), *leafPool_);

  auto schemaSection = tablet->loadOptionalSection(std::string(kSchemaSection));
  NIMBLE_CHECK(schemaSection.has_value(), "Schema not found.");
  auto nimbleSchema = SchemaDeserializer::deserialize(schemaSection->content());

  auto columnStats = vectorizedStats->toColumnStatistics(schema, nimbleSchema);

  NIMBLE_CHECK_GT(
      columnStats.size(),
      0,
      "Column stats vector is empty (seed {}).",
      options_.seed);

  std::vector<ExpectedNodeStats> expected(columnStats.size());
  for (const auto& batch : batches) {
    std::vector<velox::vector_size_t> allRows(batch->size());
    std::iota(allRows.begin(), allRows.end(), 0);
    uint32_t nodeId = 0;
    accumulateNodeStats(batch, allRows, expected, nodeId);
  }

  for (uint32_t node = 0; node < columnStats.size(); ++node) {
    const auto& nimbleActual = columnStats[node];
    if (nimbleActual == nullptr) {
      continue;
    }
    auto actual = nimbleActual->toCommonStatistics();
    auto expectedCommon = expected[node].toCommonStatistics();

    NIMBLE_CHECK_EQ(
        actual->getNumberOfValues().value_or(0),
        expectedCommon->getNumberOfValues().value_or(0),
        "Node {} value count mismatch (seed {}).",
        node,
        options_.seed);
    NIMBLE_CHECK_EQ(
        actual->hasNull().value_or(false),
        expectedCommon->hasNull().value_or(false),
        "Node {} hasNull mismatch (seed {}).",
        node,
        options_.seed);

    auto* actualInt =
        dynamic_cast<velox::dwio::common::IntegerColumnStatistics*>(
            actual.get());
    auto* expectedInt =
        dynamic_cast<velox::dwio::common::IntegerColumnStatistics*>(
            expectedCommon.get());
    if (actualInt != nullptr && expectedInt != nullptr &&
        expectedInt->getMinimum().has_value()) {
      NIMBLE_CHECK_EQ(
          *actualInt->getMinimum(),
          *expectedInt->getMinimum(),
          "Node {} integer min mismatch (seed {}).",
          node,
          options_.seed);
      NIMBLE_CHECK_EQ(
          *actualInt->getMaximum(),
          *expectedInt->getMaximum(),
          "Node {} integer max mismatch (seed {}).",
          node,
          options_.seed);
    }

    auto* actualDbl =
        dynamic_cast<velox::dwio::common::DoubleColumnStatistics*>(
            actual.get());
    auto* expectedDbl =
        dynamic_cast<velox::dwio::common::DoubleColumnStatistics*>(
            expectedCommon.get());
    if (actualDbl != nullptr && expectedDbl != nullptr &&
        expectedDbl->getMinimum().has_value()) {
      // Both bounds are NaN whenever the column carries one, because the
      // fuzzer generates NaN on purpose. Comparing with == would report those
      // agreeing statistics as a mismatch, so reuse the round-trip
      // comparison's NaN handling.
      const auto checkBound = [&](std::string_view bound,
                                  double actualValue,
                                  double expectedValue) {
        NIMBLE_CHECK(
            decodedValueEquals(actualValue, expectedValue),
            "Node {} double {} mismatch ({} vs. {}, seed {}).",
            node,
            bound,
            actualValue,
            expectedValue,
            options_.seed);
      };
      checkBound("min", *actualDbl->getMinimum(), *expectedDbl->getMinimum());
      checkBound("max", *actualDbl->getMaximum(), *expectedDbl->getMaximum());
    }

    auto* actualStr =
        dynamic_cast<velox::dwio::common::StringColumnStatistics*>(
            actual.get());
    auto* expectedStr =
        dynamic_cast<velox::dwio::common::StringColumnStatistics*>(
            expectedCommon.get());
    if (actualStr != nullptr && expectedStr != nullptr &&
        expectedStr->getMinimum().has_value()) {
      NIMBLE_CHECK_EQ(
          *actualStr->getMinimum(),
          *expectedStr->getMinimum(),
          "Node {} string min mismatch (seed {}).",
          node,
          options_.seed);
      NIMBLE_CHECK_EQ(
          *actualStr->getMaximum(),
          *expectedStr->getMaximum(),
          "Node {} string max mismatch (seed {}).",
          node,
          options_.seed);
    }
  }
}

void NimbleWriterFuzzer::verifySchemaAndStripeGroupConsistency(
    const std::string& file,
    const RowTypePtr& schema) {
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = TabletReader::create(
      readFile, leafPool_.get(), test::makeTestTabletOptions(leafPool_.get()));

  // Schema roundtrip: the Velox type reconstructed from the file must match
  // the type that was written.
  VeloxReader reader(
      std::make_shared<velox::InMemoryReadFile>(file), *leafPool_);
  NIMBLE_CHECK(
      schema->equivalent(*reader.type()),
      "Schema roundtrip mismatch (seed {}): written {} but read {}.",
      options_.seed,
      schema->toString(),
      reader.type()->toString());

  // StripeGroup consistency: per-stream byte ranges must not overlap and must
  // end before the next stripe or the end of the file.
  for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    const auto stripeIdentifier = tablet->stripeIdentifier(stripe);
    const uint32_t streamCount = tablet->streamCount(stripeIdentifier);
    if (streamCount == 0) {
      continue;
    }

    std::vector<TabletReader::StreamLocation> locations(streamCount);
    tablet->streamLocations(stripeIdentifier, locations);
    const uint64_t stripeOffset = tablet->stripeOffset(stripe);
    const uint64_t stripeEnd = stripe + 1 < tablet->stripeCount()
        ? tablet->stripeOffset(stripe + 1)
        : tablet->fileSize();
    NIMBLE_CHECK_LE(
        stripeOffset,
        stripeEnd,
        "Stripe {} has invalid bounds (seed {}).",
        stripe,
        options_.seed);

    // Collect non-empty stream ranges and sort by offset.
    struct StreamRange {
      uint32_t streamId;
      uint32_t offset;
      uint32_t size;
    };
    std::vector<StreamRange> ranges;
    for (uint32_t streamId = 0; streamId < streamCount; ++streamId) {
      const auto& location = locations[streamId];
      if (location.size > 0) {
        NIMBLE_CHECK_LE(
            uint64_t{location.offset} + location.size,
            stripeEnd - stripeOffset,
            "Stream {} extends past stripe {} bounds (seed {}).",
            streamId,
            stripe,
            options_.seed);
        ranges.push_back({streamId, location.offset, location.size});
      }
    }
    std::sort(
        ranges.begin(), ranges.end(), [](const auto& lhs, const auto& rhs) {
          return lhs.offset < rhs.offset;
        });

    // Verify no overlaps, allowing dedup aliases with identical ranges.
    for (size_t i = 1; i < ranges.size(); ++i) {
      const auto& prev = ranges[i - 1];
      const auto& curr = ranges[i];
      if (prev.offset == curr.offset && prev.size == curr.size) {
        continue;
      }
      NIMBLE_CHECK_LE(
          uint64_t{prev.offset} + prev.size,
          curr.offset,
          "Streams {} and {} overlap in stripe {} (seed {}).",
          prev.streamId,
          curr.streamId,
          stripe,
          options_.seed);
    }
  }

  // Stripe row counts must sum to the tablet's total row count.
  uint64_t totalRows = 0;
  for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    totalRows += tablet->stripeRowCount(stripe);
  }
  NIMBLE_CHECK_EQ(
      totalRows,
      tablet->tabletRowCount(),
      "Stripe row count sum mismatch (seed {}).",
      options_.seed);
}

void NimbleWriterFuzzer::readAndVerify(
    const std::string& file,
    const RowTypePtr& schema,
    const std::vector<VectorPtr>& batches,
    std::string_view selectionContext,
    ReaderPath readerPath) {
  // A read size that divides neither the batch size nor any chunk size keeps
  // read boundaries from lining up with chunk and stripe boundaries.
  constexpr uint64_t kReadSize = 97;

  // Maps a row of the flattened read sequence back to its source batch. The
  // cursor only ever moves forward, so it is advanced in step with the reads
  // rather than re-scanned per row.
  size_t batchIndex{0};
  velox::vector_size_t batchOffset{0};
  auto verifyBatch = [&](const VectorPtr& actual) {
    for (velox::vector_size_t i = 0; i < actual->size(); ++i) {
      while (batchIndex < batches.size() &&
             batchOffset == batches[batchIndex]->size()) {
        ++batchIndex;
        batchOffset = 0;
      }
      NIMBLE_CHECK_LT(
          batchIndex,
          batches.size(),
          "Reader returned more rows than were written (seed {}, {}, reader {}).",
          options_.seed,
          selectionContext,
          toString(readerPath));
      const auto& expected = batches[batchIndex];
      if (!expected->equalValueAt(actual.get(), batchOffset, i)) {
        NIMBLE_FAIL(
            "Round-trip mismatch (seed {}, {}, reader {}, batch {}, row {}). Expected: {} Actual: {}",
            options_.seed,
            selectionContext,
            toString(readerPath),
            batchIndex,
            batchOffset,
            expected->toString(batchOffset),
            actual->toString(i));
      }
      ++batchOffset;
    }
  };

  auto checkSchema = [&](const velox::RowType& actual) {
    // Compares field names as well as types; equivalent() ignores names, so a
    // name regression would slip through.
    NIMBLE_CHECK(
        *schema == actual,
        "Schema changed across the round trip (seed {}, {}, reader {}). Expected {}, got {}.",
        options_.seed,
        selectionContext,
        toString(readerPath),
        schema->toString(),
        actual.toString());
  };

  uint64_t rowsRead{0};
  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);

  if (readerPath == ReaderPath::kLegacyFactory ||
      readerPath == ReaderPath::kDefaultFactory) {
    VeloxReadParams params;
    if (readerPath == ReaderPath::kDefaultFactory) {
      params.encodingFactory =
          [](velox::memory::MemoryPool& pool,
             std::string_view data,
             std::function<void*(uint32_t)> stringBufferFactory) {
            return EncodingFactory().create(pool, data, stringBufferFactory);
          };
    }
    VeloxReader reader(readFile, *leafPool_, /*selector=*/nullptr, params);
    checkSchema(*reader.type());

    VectorPtr result;
    while (reader.next(kReadSize, result)) {
      verifyBatch(result);
      rowsRead += result->size();
    }
  } else {
    auto factory = velox::dwio::common::getReaderFactory(
        velox::dwio::common::FileFormat::NIMBLE);
    auto scanSpec = std::make_shared<velox::common::ScanSpec>("root");
    scanSpec->addAllChildFields(*schema);

    velox::dwio::common::ReaderOptions readerOptions(leafPool_.get());
    readerOptions.setDataIoStats(std::make_shared<velox::io::IoStatistics>());
    readerOptions.setMetadataIoStats(
        std::make_shared<velox::io::IoStatistics>());
    readerOptions.setScanSpec(scanSpec);
    auto reader = factory->createReader(
        std::make_unique<velox::dwio::common::BufferedInput>(
            readFile, *leafPool_),
        readerOptions);
    checkSchema(*reader->rowType());

    velox::dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(schema);
    rowOptions.setStringDecoderZeroCopy(
        readerPath == ReaderPath::kSelectiveDefaultDispatch);
    auto rowReader = reader->createRowReader(rowOptions);

    auto result = velox::BaseVector::create(schema, 0, leafPool_.get());
    while (rowReader->next(kReadSize, result) > 0) {
      verifyBatch(result);
      rowsRead += result->size();
    }
  }

  NIMBLE_CHECK_EQ(
      rowsRead,
      totalRows(batches),
      "Row count changed across the round trip (seed {}, {}, reader {}).",
      options_.seed,
      selectionContext,
      toString(readerPath));
}

std::set<EncodingType> NimbleWriterFuzzer::recordUnfilteredCoverage(
    const std::string& file) {
  const auto& candidates = unfilteredCandidateEncodings();
  for (const auto encodingType : candidates) {
    ++coverage_[encodingType].numFilesOffered;
  }
  ++numUnfilteredFilesWritten_;

  const auto observations = inspectFileEncodings(file, leafPool_.get());
  for (const auto encodingType : observations.encodingTypes) {
    NIMBLE_CHECK(
        std::find(candidates.begin(), candidates.end(), encodingType) !=
            candidates.end(),
        "Unfiltered policy produced encoding {} outside its default candidate set.",
        toString(encodingType));
    ++coverage_[encodingType].numFilesApplied;
    ++coverage_[encodingType].numUnfilteredFilesApplied;
  }
  for (const auto& [pair, chunkCount] : observations.chunkCounts) {
    coverage_[pair.second].numChunksApplied += chunkCount;
    pairCoverage_[pair].numChunksApplied += chunkCount;
  }
  for (const auto dataType : observations.dataTypes) {
    observedDataTypes_.insert(dataType);
    for (const auto encodingType : candidates) {
      if (isTypeCompatible(encodingType, dataType)) {
        ++pairCoverage_[{dataType, encodingType}].numFilesSeen;
      }
    }
  }
  return observations.encodingTypes;
}

void NimbleWriterFuzzer::recordForcedCoverage(
    const std::string& file,
    EncodingType encodingType) {
  auto& stats = coverage_[encodingType];
  ++stats.numFilesOffered;
  ++stats.numForcedFilesWritten;

  const auto observations = inspectFileEncodings(file, leafPool_.get());
  bool appliedInFile = false;
  for (const auto& [pair, chunkCount] : observations.chunkCounts) {
    auto& pairStats = pairCoverage_[{pair.first, encodingType}];
    if (pair.second == encodingType) {
      stats.numChunksApplied += chunkCount;
      pairStats.numChunksApplied += chunkCount;
      appliedInFile = true;
    } else {
      stats.numForcedChunksFellBack += chunkCount;
      pairStats.numForcedChunksFellBack += chunkCount;
    }
  }
  for (const auto dataType : observations.dataTypes) {
    observedDataTypes_.insert(dataType);
    ++pairCoverage_[{dataType, encodingType}].numFilesSeen;
  }
  if (appliedInFile) {
    ++stats.numFilesApplied;
  }
}

bool NimbleWriterFuzzer::verifyEncodedColumn(
    const std::string& file,
    const VectorPtr& batch,
    EncodingType encodingType) {
  const auto* row = batch->as<velox::RowVector>();
  if (row == nullptr || row->childrenSize() != 1 || row->mayHaveNulls()) {
    return false;
  }
  const auto& column = row->childAt(0);
  // Nulls would put the values under a Nullable wrapper and split the stream,
  // and a non-flat child has no direct value sequence to compare against.
  if (column == nullptr || !column->isFlatEncoding() ||
      column->mayHaveNulls()) {
    return false;
  }

  const auto dataType = velox::TypeKind::VARCHAR == column->typeKind() ||
          velox::TypeKind::VARBINARY == column->typeKind()
      ? DataType::String
      : DataType::Undefined;
  (void)dataType;

  auto readFile = std::make_shared<velox::InMemoryReadFile>(file);
  auto tablet = TabletReader::create(
      readFile, leafPool_.get(), test::makeTestTabletOptions(leafPool_.get()));

  velox::vector_size_t cursor{0};
  // String encodings materialize into caller-supplied storage. Buffer owns the
  // pool-backed arena and releases it here, rather than leaking one allocation
  // per chunk for the life of the fuzzer.
  Buffer stringBuffer{*leafPool_};
  auto stringBufferFactory = [&stringBuffer](uint32_t size) -> void* {
    return stringBuffer.reserve(size);
  };

  for (uint32_t stripe = 0; stripe < tablet->stripeCount(); ++stripe) {
    const auto stripeIdentifier = tablet->stripeIdentifier(stripe);
    std::vector<uint32_t> streamIdentifiers(
        tablet->streamCount(stripeIdentifier));
    std::iota(streamIdentifiers.begin(), streamIdentifiers.end(), 0);
    auto streams = tablet->load(stripeIdentifier, streamIdentifiers);
    for (auto& stream : streams) {
      if (stream == nullptr) {
        continue;
      }
      InMemoryChunkedStream chunkedStream{*leafPool_, std::move(stream)};
      while (chunkedStream.hasNext()) {
        const auto chunk = chunkedStream.nextChunk();
        auto encoding =
            EncodingFactory().create(*leafPool_, chunk, stringBufferFactory);
        switch (encoding->dataType()) {
          case DataType::Bool:
            compareDecodedChunk<bool>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<bool>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::Int8:
            compareDecodedChunk<int8_t>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<int8_t>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::Int16:
            compareDecodedChunk<int16_t>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<int16_t>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::Int32:
            compareDecodedChunk<int32_t>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<int32_t>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::Int64:
            compareDecodedChunk<int64_t>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<int64_t>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::Float:
            compareDecodedChunk<float>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<float>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::Double:
            compareDecodedChunk<double>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  return column->asUnchecked<velox::FlatVector<double>>()
                      ->valueAt(i);
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          case DataType::String:
            compareDecodedChunk<std::string_view>(
                *encoding,
                leafPool_.get(),
                column->size(),
                [&](auto i) {
                  const auto value =
                      column
                          ->asUnchecked<velox::FlatVector<velox::StringView>>()
                          ->valueAt(i);
                  return std::string_view(value.data(), value.size());
                },
                cursor,
                encodingType,
                options_.seed);
            break;
          default:
            // A stream type the column cannot have produced: the shape check
            // above admits only single scalar columns, so this is a stream the
            // writer added (a length or offset stream) and there is nothing to
            // compare it against.
            break;
        }
      }
    }
  }

  NIMBLE_CHECK_EQ(
      cursor,
      column->size(),
      "Encoded stream holds fewer rows than were written (seed {}, encoding {}).",
      options_.seed,
      toString(encodingType));
  return true;
}

bool NimbleWriterFuzzer::verifyReaderPaths(
    const std::string& file,
    const RowTypePtr& schema,
    const std::vector<VectorPtr>& batches,
    EncodingType encodingType,
    std::span<const ReaderPath> readerPaths) {
  const auto appliedBefore = coverage_[encodingType].numChunksApplied;
  recordForcedCoverage(file, encodingType);
  const auto selectionContext =
      fmt::format("encoding {}", toString(encodingType));
  for (const auto readerPath : readerPaths) {
    readAndVerify(file, schema, batches, selectionContext, readerPath);
  }
  return coverage_[encodingType].numChunksApplied > appliedBefore;
}

WriteOutcome NimbleWriterFuzzer::runFixed(
    const std::vector<VectorPtr>& batches,
    EncodingType encodingType) {
  NIMBLE_CHECK(!batches.empty(), "Expecting at least one input batch.");
  const auto schema = velox::asRowType(batches[0]->type());

  const auto file = writeFile(batches, encodingType, options_.seed);

  // Establish value correctness at the encoding level first, where a mismatch
  // is unambiguously the encoder's. The reader paths below then answer the
  // separate question of whether every dispatch table can decode what was
  // written. Single-batch flat columns only; anything else falls through to
  // the reader comparison alone.
  if (batches.size() == 1) {
    verifyEncodedColumn(file, batches.front(), encodingType);
  }

  return verifyReaderPaths(file, schema, batches, encodingType, kAllReaderPaths)
      ? WriteOutcome::kApplied
      : WriteOutcome::kNotApplied;
}

void NimbleWriterFuzzer::run() {
  FuzzerGenerator rng(options_.seed);

  const auto schema = velox::fuzzer::randRowType(
      rng,
      supportedScalarTypes(),
      options_.maxSchemaDepth,
      supportedMapKeyTypes(),
      /*mapValueTypes=*/{});

  velox::VectorFuzzer::Options fuzzerOptions;
  fuzzerOptions.vectorSize = options_.batchSize;
  fuzzerOptions.nullRatio = 0.2 * folly::Random::randDouble01(rng);
  // NaN and infinity are off by default and are exactly the values the
  // floating-point encodings are most likely to mishandle.
  fuzzerOptions.dataSpec = {/*includeNaN=*/true, /*includeInfinity=*/true};
  // Flat-map writers reject duplicate keys; normalization is what guarantees
  // the fuzzed maps are writable.
  fuzzerOptions.normalizeMapKeys = true;
  fuzzerOptions.containerHasNulls = folly::Random::oneIn(2, rng);
  fuzzerOptions.fuzzNonContiguousElements = folly::Random::oneIn(2, rng);
  // VectorFuzzer bounds timestamps so the writer's micros conversion cannot
  // overflow, and sub-microsecond nanos survive the round trip.
  fuzzerOptions.timestampPrecision =
      velox::fuzzer::FuzzerTimestampPrecision::kNanoSeconds;

  velox::VectorFuzzer vectorFuzzer(
      fuzzerOptions, leafPool_.get(), folly::Random::rand32(rng));

  // Two thirds of iterations bias the scalar columns toward the shapes ALP and
  // FSST are built for; the rest stay purely random so the generic encodings
  // keep seeing adversarial input.
  const bool shapeForEncodings = !folly::Random::oneIn(3, rng);
  const auto columnShapes = drawColumnShapes(schema, rng);
  // Seeded from the type minimum so a non-decreasing column spans the widest
  // range its type allows before saturating.
  std::vector<int64_t> monotonicCursors(
      schema->size(), std::numeric_limits<int64_t>::min());
  for (auto i = 0; i < schema->size(); ++i) {
    if (columnShapes.at(i) == ColumnShape::kNonDecreasing) {
      monotonicCursors.at(i) = minValueOf(schema->childAt(i)->kind());
    }
  }

  std::vector<VectorPtr> batches;
  batches.reserve(options_.numBatches);
  for (uint32_t batch = 0; batch < options_.numBatches; ++batch) {
    auto vector = vectorFuzzer.fuzzInputFlatRow(schema);
    if (shapeForEncodings) {
      applyEncodingFriendlyShapes(
          vector->asUnchecked<velox::RowVector>(),
          columnShapes,
          monotonicCursors,
          uint64_t{options_.batchSize} * options_.numBatches,
          rng);
    }
    batches.push_back(std::move(vector));
  }

  const auto candidates = allCandidateEncodings();
  const auto readerPath =
      kAllReaderPaths[options_.seed % kAllReaderPaths.size()];
  const std::array<ReaderPath, 1> readerPaths = {readerPath};
  VLOG(1) << "Fuzzing " << schema->toString() << " with " << candidates.size()
          << " candidates across " << kNumUnfilteredRounds
          << " unfiltered rounds, followed by missing-encoding repair through "
          << toString(readerPath) << ", seed " << options_.seed;

  std::set<EncodingType> observedEncodings;
  for (uint32_t round = 0; round < kNumUnfilteredRounds; ++round) {
    const auto roundSeed = folly::hash::hash_combine(
        options_.seed, std::string_view{"unfiltered"}, round);
    const auto file = writeFile(batches, std::nullopt, roundSeed);
    const auto observedInFile = recordUnfilteredCoverage(file);
    observedEncodings.insert(observedInFile.begin(), observedInFile.end());
    readAndVerify(
        file,
        schema,
        batches,
        fmt::format("unfiltered random selection round {}", round),
        readerPath);
  }

  std::vector<EncodingType> missingEncodings;
  std::vector<std::string> missingEncodingNames;
  for (const auto encodingType : candidates) {
    if (!observedEncodings.contains(encodingType)) {
      missingEncodings.push_back(encodingType);
      missingEncodingNames.push_back(toString(encodingType));
    }
  }
  VLOG(1) << fmt::format(
      "Unfiltered rounds observed {}/{} candidates; forcing {} missing encodings [{}]",
      observedEncodings.size(),
      candidates.size(),
      missingEncodings.size(),
      fmt::join(missingEncodingNames, ", "));

  // The default random policy omits the integral-only candidates because its
  // write-side and read-side floating-point gates disagree (T283330065), so
  // they always reach this repair phase. gateFloatingPointStreams holds them
  // off float streams alone while still exercising integer streams in a mixed
  // schema.
  for (const auto encodingType : missingEncodings) {
    const auto roundSeed = folly::hash::hash_combine(
        options_.seed,
        std::string_view{"forced"},
        static_cast<std::underlying_type_t<EncodingType>>(encodingType));
    const auto file = writeFile(batches, encodingType, roundSeed);
    verifyReaderPaths(file, schema, batches, encodingType, readerPaths);
  }
}

std::vector<EncodingType> NimbleWriterFuzzer::unappliedEncodings() const {
  std::vector<EncodingType> unapplied;
  for (const auto encodingType : allCandidateEncodings()) {
    const auto entry = coverage_.find(encodingType);
    // Absent means neither the random policy offered the candidate nor the
    // repair phase reached it. This is distinguished from a zero count because
    // the two point at very different causes.
    if (entry == coverage_.end() || entry->second.numChunksApplied == 0) {
      unapplied.push_back(encodingType);
    }
  }
  return unapplied;
}

std::vector<EncodingPair> NimbleWriterFuzzer::unappliedPairs() const {
  std::vector<EncodingPair> unapplied;
  for (const auto dataType : observedDataTypes_) {
    const bool isFloatingPoint =
        dataType == DataType::Float || dataType == DataType::Double;
    for (const auto encodingType : allCandidateEncodings()) {
      if (!isTypeCompatible(encodingType, dataType) ||
          (isFloatingPoint && isIntegralOnlyEncoding(encodingType)) ||
          hasDataPrecondition(encodingType)) {
        continue;
      }
      const auto entry = pairCoverage_.find({dataType, encodingType});
      if (entry == pairCoverage_.end()) {
        continue;
      }
      // Demand the pair only once enough independently seeded files have
      // offered it for a zero to mean something. The default candidates get
      // ten opportunities in one random sweep; repair-only candidates need ten
      // outer iterations that carry the DataType.
      if (entry->second.numFilesSeen >= kMinPairFiles &&
          entry->second.numChunksApplied == 0) {
        unapplied.emplace_back(dataType, encodingType);
      }
    }
  }
  return unapplied;
}

void NimbleWriterFuzzer::logPairCoverage() const {
  // Three buckets per data type. "incompatible" is the pairs
  // EncodingSizeEstimation's type gate can never admit. Separating them keeps
  // a real coverage hole distinct from a combination that was never possible,
  // whether the opportunity came from natural selection or a forced repair.
  LOG(WARNING) << "Encoding coverage by data type:";
  bool anyPrecondition = false;
  for (const auto dataType : observedDataTypes_) {
    const bool isFloatingPoint =
        dataType == DataType::Float || dataType == DataType::Double;
    std::vector<std::string> applied;
    std::vector<std::string> unapplied;
    std::vector<std::string> incompatible;
    for (const auto encodingType : allCandidateEncodings()) {
      // Integral-only encodings are withheld from floating point on purpose
      // (T283330065), so for those streams they are unusable rather than
      // merely unapplied.
      if (!isTypeCompatible(encodingType, dataType) ||
          (isFloatingPoint && isIntegralOnlyEncoding(encodingType))) {
        incompatible.emplace_back(toString(encodingType));
        continue;
      }
      const auto entry = pairCoverage_.find({dataType, encodingType});
      const bool wasApplied =
          entry != pairCoverage_.end() && entry->second.numChunksApplied > 0;
      // Flag the ones unappliedPairs() deliberately does not demand, so a zero
      // next to them does not read as a regression.
      anyPrecondition |= hasDataPrecondition(encodingType);
      auto name = hasDataPrecondition(encodingType)
          ? fmt::format("{}*", toString(encodingType))
          : std::string{toString(encodingType)};
      (wasApplied ? applied : unapplied).push_back(std::move(name));
    }
    LOG(WARNING) << fmt::format(
        "  {:<8} applied {}/{}  [{}]",
        toString(dataType),
        applied.size(),
        applied.size() + unapplied.size(),
        fmt::join(applied, " "));
    if (!unapplied.empty()) {
      LOG(WARNING) << fmt::format(
          "           never applied [{}]", fmt::join(unapplied, " "));
    }
    if (!incompatible.empty()) {
      LOG(WARNING) << fmt::format(
          "           incompatible  [{}]", fmt::join(incompatible, " "));
    }
  }
  if (anyPrecondition) {
    LOG(WARNING) << "  (* = applies only to data that meets a precondition, so "
                    "not demanded by the coverage gate)";
  }
}

void NimbleWriterFuzzer::logCoverage() const {
  std::vector<std::pair<EncodingType, EncodingCoverage>> entries(
      coverage_.begin(), coverage_.end());
  // Never-applied encodings first: a candidate that was never actually used is
  // the finding worth seeing at the top of a CI log. Stable so ties, which are
  // common at zero, print in a reproducible order.
  std::stable_sort(
      entries.begin(), entries.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.second.numChunksApplied < rhs.second.numChunksApplied;
      });

  // WARNING, not INFO: this is the run's headline result, and the Cogwheel job
  // passes --minloglevel=1, which would otherwise drop the whole table exactly
  // when a failure makes it most useful.
  LOG(WARNING)
      << "Encoding coverage (candidate -> actual top-level applications):";
  for (const auto& [encodingType, stats] : entries) {
    LOG(WARNING) << fmt::format(
        "  {:<16} offered={:<6} randomFilesApplied={:<6} filesApplied={:<6} chunksApplied={:<8} forcedFiles={:<6} forcedFallbacks={:<8}{}",
        toString(encodingType),
        stats.numFilesOffered,
        stats.numUnfilteredFilesApplied,
        stats.numFilesApplied,
        stats.numChunksApplied,
        stats.numForcedFilesWritten,
        stats.numForcedChunksFellBack,
        stats.numChunksApplied == 0 ? "   <-- NEVER APPLIED" : "");
  }
}

void NimbleWriterFuzzer::logChunkStatsCoverage() const {
  LOG(WARNING) << fmt::format(
      "Chunk-index coverage: files(indexed={},unindexed={}) groups={} stripes={} streams(scalar={},structural={},multiChunk={}) chunks(first={},middle={},final={},zeroNull={},partialNull={},fullNull={},compressed={},uncompressed={})",
      chunkStatsCoverage_.numIndexedFiles,
      chunkStatsCoverage_.numUnindexedFiles,
      chunkStatsCoverage_.numStripeGroups,
      chunkStatsCoverage_.numStripes,
      chunkStatsCoverage_.numScalarStreams,
      chunkStatsCoverage_.numStructuralStreams,
      chunkStatsCoverage_.numMultiChunkStreams,
      chunkStatsCoverage_.numFirstChunks,
      chunkStatsCoverage_.numMiddleChunks,
      chunkStatsCoverage_.numFinalChunks,
      chunkStatsCoverage_.numZeroNullChunks,
      chunkStatsCoverage_.numPartiallyNullChunks,
      chunkStatsCoverage_.numFullyNullChunks,
      chunkStatsCoverage_.numCompressedChunks,
      chunkStatsCoverage_.numUncompressedChunks);
}

std::vector<std::string_view> NimbleWriterFuzzer::uncoveredChunkStatsShapes()
    const {
  std::vector<std::string_view> uncovered;
  auto require = [&](bool covered, std::string_view name) {
    if (!covered) {
      uncovered.push_back(name);
    }
  };
  require(chunkStatsCoverage_.numIndexedFiles > 0, "indexed file");
  require(chunkStatsCoverage_.numUnindexedFiles > 0, "unindexed file");
  // TabletWithIndexTest.multipleGroups deterministically covers multiple
  // stripe groups. WriterOptions does not expose the tablet metadata flush
  // threshold, so requiring that layout here would be unreachable.
  require(
      chunkStatsCoverage_.numStripes > chunkStatsCoverage_.numStripeGroups,
      "multiple stripes per group");
  require(chunkStatsCoverage_.numScalarStreams > 0, "scalar stream");
  require(chunkStatsCoverage_.numStructuralStreams > 0, "structural stream");
  require(chunkStatsCoverage_.numMultiChunkStreams > 0, "multi-chunk stream");
  require(chunkStatsCoverage_.numFirstChunks > 0, "first chunk");
  require(chunkStatsCoverage_.numMiddleChunks > 0, "middle chunk");
  require(chunkStatsCoverage_.numFinalChunks > 0, "final chunk");
  require(chunkStatsCoverage_.numZeroNullChunks > 0, "zero-null chunk");
  require(
      chunkStatsCoverage_.numPartiallyNullChunks > 0, "partially-null chunk");
  // Fully-null chunks are generated deterministically by
  // WriterTest.chunkNullCountsForStructNullStream. Requiring a random schema
  // and chunk boundary to align around one here would make CI probabilistic.
  require(chunkStatsCoverage_.numUncompressedChunks > 0, "uncompressed chunk");
  return uncovered;
}

} // namespace facebook::nimble::fuzzer
