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
#include "velox/dwio/nimble/tests/OptionsOffFixtures.h"

#include <functional>
#include <limits>
#include <memory>
#include <utility>

#include "velox/common/file/File.h"
#include "velox/dwio/nimble/common/Checksum.h"
#include "velox/dwio/nimble/common/SharedDictionaryConfig.h"
#include "velox/dwio/nimble/common/Types.h"
#include "velox/dwio/nimble/encodings/common/EncodingLayout.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/writer/EncodingLayoutTree.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::test::options_off {
namespace {

// 'count' values from 64 distinct ones over [base, base + 2^bits), both ends
// included. The minimum is 'base' and the values need exactly 'bits' bits
// above it, so a chunk's stored width is 'bits' (rounded up to whole bytes
// unless exact bits are on).
template <typename T>
std::vector<T> spreadValues(
    velox::vector_size_t count,
    int bits,
    int64_t base,
    uint64_t seed) {
  const uint64_t range = bits >= 64 ? ~uint64_t{0} : (uint64_t{1} << bits);
  const uint64_t step = (range - 1) / 63;
  std::vector<T> values(count);
  for (velox::vector_size_t i = 0; i < count; ++i) {
    values[i] = static_cast<T>(
        static_cast<uint64_t>(base) + ((i * 37 + seed) % 64) * step);
  }
  values[0] = static_cast<T>(base);
  values[1] = static_cast<T>(static_cast<uint64_t>(base) + 63 * step);
  return values;
}

// Selects FixedBitWidth for every numeric stream (Trivial for strings and
// booleans), or Dictionary over FixedBitWidth for shared-dictionary types when
// 'dictionaryForIntegers'.
EncodingSelectionPolicyCreator fixedBitWidthCreator(
    const CompressionOptions& compression,
    bool dictionaryForIntegers) {
  return
      [compression, dictionaryForIntegers](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        const std::vector<std::pair<EncodingType, float>> fixedBitWidth{
            {EncodingType::FixedBitWidth, 1.0},
        };
        if (dictionaryForIntegers && isSharedDictionaryType(dataType)) {
          return ManualEncodingSelectionPolicyFactory{
              {{EncodingType::Dictionary, 1.0}}, compression, fixedBitWidth}
              .createPolicy(dataType);
        }
        if (dataType == DataType::String || dataType == DataType::Bool) {
          return ManualEncodingSelectionPolicyFactory{
              {{EncodingType::Trivial, 1.0}}, compression, fixedBitWidth}
              .createPolicy(dataType);
        }
        return ManualEncodingSelectionPolicyFactory{fixedBitWidth, compression}
            .createPolicy(dataType);
      };
}

// Selects with the default read factors.
EncodingSelectionPolicyCreator defaultFactorsCreator(
    CompressionOptions compression) {
  ManualEncodingSelectionPolicyFactory factory{
      ManualEncodingSelectionPolicyFactory::defaultEncodingReadFactors(),
      compression};
  return
      [factory = std::move(factory)](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
        return factory.createPolicy(dataType);
      };
}

// A layout tree with 'layout' for the top-level scalar column 'column' only.
EncodingLayoutTree columnLayout(
    const std::string& column,
    EncodingLayout layout) {
  return EncodingLayoutTree{
      Kind::Row,
      {},
      "",
      {EncodingLayoutTree{
          Kind::Scalar,
          {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
            std::move(layout)}},
          column}}};
}

// A layout tree with 'layout' for the column 'fsst'. A tree matches columns by
// position, so the columns before it get placeholder nodes of their kinds.
EncodingLayoutTree fsstColumnLayout(EncodingLayout layout) {
  std::vector<EncodingLayoutTree> columns;
  const auto addPlaceholders = [&columns](Kind kind, int count) {
    for (int i = 0; i < count; ++i) {
      columns.push_back(EncodingLayoutTree{kind, {}, ""});
    }
  };
  addPlaceholders(Kind::Scalar, /*count=*/12);
  addPlaceholders(Kind::Array, /*count=*/1);
  addPlaceholders(Kind::Map, /*count=*/1);
  addPlaceholders(Kind::Scalar, /*count=*/4);
  columns.push_back(
      EncodingLayoutTree{
          Kind::Scalar,
          {{EncodingLayoutTree::StreamIdentifiers::Scalar::ScalarStream,
            std::move(layout)}},
          "fsst"});
  return EncodingLayoutTree{Kind::Row, {}, "", std::move(columns)};
}

} // namespace

std::vector<Fixture> allFixtures() {
  const std::vector<std::pair<FixturePath, std::string>> paths{
      {FixturePath::kFreshDefaultFactors, "freshDefaultFactors"},
      {FixturePath::kFreshFixedBitWidth, "freshFixedBitWidth"},
      {FixturePath::kSelectionCache, "selectionCache"},
      {FixturePath::kLayoutTree, "layoutTree"},
      {FixturePath::kReplayFallback, "replayFallback"},
      {FixturePath::kSharedDictionary, "sharedDictionary"},
      {FixturePath::kPinnedEncodings, "pinnedEncodings"},
  };
  std::vector<Fixture> fixtures;
  for (const auto compression :
       {FixtureCompression::kUncompressed,
        FixtureCompression::kZstdBelowMinSize}) {
    const std::string compressionName =
        compression == FixtureCompression::kUncompressed ? "uncompressed"
                                                         : "zstdBelowMinSize";
    for (const bool exactBits : {false, true}) {
      for (const auto& [path, pathName] : paths) {
        std::string name = pathName;
        name += '.';
        name += compressionName;
        if (exactBits) {
          name += ".exactBits";
        }
        fixtures.push_back(
            Fixture{
                .name = std::move(name),
                .compression = compression,
                .path = path,
                .exactBits = exactBits,
            });
      }
    }
  }
  return fixtures;
}

// Columns, and what typed lanes would do with each if they were on:
// - i8: TINYINT, 7 bits. Never lanes: below 16 stored bits.
// - i16: SMALLINT, 12 bits from 100. A 16-bit type: full-width lanes.
// - i32_16: INTEGER, 16 bits from 5. 16-bit lanes in a 32-bit type.
// - i32_24: INTEGER, 24 bits from 1,000,000. Full-width 32-bit lanes.
// - i32_full: INTEGER over all 32 bits from 0. Stored at full width with a
//   zero baseline even without lanes: the reader's full-width case.
// - i64_09, i64_17, i64_24, i64_32: BIGINT needing 9, 17, 24 and 32 bits
//   (16-, 32-, 32- and 32-bit lanes), from 1,000,000, 0, 1,000,000 and
//   -5,000,000,000.
// - i64_40: BIGINT, 40 bits from 0. Full-width 64-bit lanes.
// - i64_full: BIGINT over all 64 bits from 0. Full width, zero baseline.
// - sparse: BIGINT, 16 bits, 90% null. Its non-null values are a nested
//   stream far shorter than the row count.
// - arr: ARRAY<INTEGER>, lengths 0-9, elements needing 20 bits.
// - map: MAP<BIGINT, REAL>, keys needing 18 bits.
// - lowcard: INTEGER from 50 distinct values: Dictionary indices, and the
//   shared-dictionary column.
// - runs: BIGINT in runs: RLE run lengths and values.
// - split: BIGINT with a high and a low part: the SubIntSplit-pinned column.
// - str: VARCHAR (Trivial lengths and data); fsst: VARCHAR, the FSST-pinned
//   column; flt: REAL; dbl: DOUBLE; flag: BOOLEAN.
std::vector<velox::RowVectorPtr> makeBatches(velox::memory::MemoryPool& pool) {
  velox::test::VectorMaker maker{&pool};
  std::vector<velox::RowVectorPtr> batches;
  uint64_t seed{0};
  for (const velox::vector_size_t rows : {4'096, 257, 256, 255, 100}) {
    ++seed;
    const auto sparseValues = spreadValues<int64_t>(rows, 16, 0, seed);
    std::vector<std::string> strings(rows);
    std::vector<std::string> fsstStrings(rows);
    std::vector<float> floats(rows);
    std::vector<double> doubles(rows);
    std::vector<int32_t> lowcard(rows);
    std::vector<int64_t> runs(rows);
    std::vector<int64_t> split(rows);
    for (velox::vector_size_t i = 0; i < rows; ++i) {
      strings[i] = std::string(
          static_cast<size_t>((i * 7 + seed) % 40),
          static_cast<char>('a' + i % 26));
      fsstStrings[i] = "user_" + std::to_string((i * 131 + seed) % 997) +
          "_session_" + std::to_string(i % 13);
      floats[i] = static_cast<float>((i * 3 + seed) % 1'000) * 1.5f;
      doubles[i] = static_cast<double>((i * 11 + seed) % 10'000) * 0.25;
      lowcard[i] = static_cast<int32_t>(((i * 7 + seed) % 50) * 1'003);
      runs[i] = static_cast<int64_t>((i / 16 + seed) % 100) * 17;
      split[i] = (static_cast<int64_t>((i + seed) % 7) << 40) |
          static_cast<int64_t>((i * 7'919 + seed) % 65'521);
    }
    batches.push_back(maker.rowVector(
        {"i8",     "i16",    "i32_16",  "i32_24", "i32_full", "i64_09",
         "i64_17", "i64_24", "i64_32",  "i64_40", "i64_full", "sparse",
         "arr",    "map",    "lowcard", "runs",   "split",    "str",
         "fsst",   "flt",    "dbl",     "flag"},
        {
            maker.flatVector(spreadValues<int8_t>(rows, 7, 0, seed)),
            maker.flatVector(spreadValues<int16_t>(rows, 12, 100, seed)),
            maker.flatVector(spreadValues<int32_t>(rows, 16, 5, seed)),
            maker.flatVector(spreadValues<int32_t>(rows, 24, 1'000'000, seed)),
            maker.flatVector(spreadValues<int32_t>(rows, 32, 0, seed)),
            maker.flatVector(spreadValues<int64_t>(rows, 9, 1'000'000, seed)),
            maker.flatVector(spreadValues<int64_t>(rows, 17, 0, seed)),
            maker.flatVector(spreadValues<int64_t>(rows, 24, 1'000'000, seed)),
            maker.flatVector(
                spreadValues<int64_t>(rows, 32, -5'000'000'000, seed)),
            maker.flatVector(spreadValues<int64_t>(rows, 40, 0, seed)),
            maker.flatVector(spreadValues<int64_t>(rows, 64, 0, seed)),
            maker.flatVector<int64_t>(
                rows,
                [&](velox::vector_size_t row) { return sparseValues[row]; },
                [](velox::vector_size_t row) { return row % 10 != 3; }),
            maker.arrayVector<int32_t>(
                rows,
                [](velox::vector_size_t row) { return row % 10; },
                [seed](velox::vector_size_t index) {
                  return static_cast<int32_t>(
                      ((index * 37 + seed) % 64) * (((1 << 20) - 1) / 63));
                }),
            maker.mapVector<int64_t, float>(
                rows,
                [](velox::vector_size_t row) { return row % 4; },
                [seed](velox::vector_size_t index) {
                  return static_cast<int64_t>(
                      ((index * 53 + seed) % 64) * (((1 << 18) - 1) / 63));
                },
                [](velox::vector_size_t index) {
                  return static_cast<float>(index % 100) * 0.5f;
                }),
            maker.flatVector(lowcard),
            maker.flatVector(runs),
            maker.flatVector(split),
            maker.flatVector(strings),
            maker.flatVector(fsstStrings),
            maker.flatVector(floats),
            maker.flatVector(doubles),
            maker.flatVector<bool>(
                rows,
                [seed](velox::vector_size_t row) {
                  return (row * 5 + seed) % 3 == 0;
                }),
        }));
  }
  return batches;
}

bool endsStripe(size_t batch) {
  return batch == 1 || batch == 3;
}

CompressionOptions compressionOptions(FixtureCompression compression) {
  CompressionOptions options;
  if (compression == FixtureCompression::kUncompressed) {
    options.compressionType = CompressionType::Uncompressed;
  } else {
    options.compressionType = CompressionType::Zstd;
    options.zstdMinCompressionSize = std::numeric_limits<uint64_t>::max();
  }
  return options;
}

WriterOptions fixtureOptions(const Fixture& fixture) {
  return fixtureOptions(fixture, compressionOptions(fixture.compression));
}

WriterOptions fixtureOptions(
    const Fixture& fixture,
    const CompressionOptions& compression) {
  WriterOptions options;
  // No default metadata (the host name and build info): the golden digests
  // cover the whole file, so they must not depend on the binary or the host
  // that writes it.
  options.metadata = {};
  options.compressionOptions = compression;
  options.fixedBitWidthUseExactBits = fixture.exactBits;
  options.enableVectorizedStats = false;
  options.metadataCompressionThreshold = std::numeric_limits<uint32_t>::max();
  options.encodingSelectionPolicyCreator =
      fixedBitWidthCreator(compression, /*dictionaryForIntegers=*/false);
  switch (fixture.path) {
    case FixturePath::kFreshDefaultFactors:
      options.encodingSelectionPolicyCreator =
          defaultFactorsCreator(compression);
      break;
    case FixturePath::kFreshFixedBitWidth:
      break;
    case FixturePath::kSelectionCache:
      options.encodingSelectionPolicyCreator =
          defaultFactorsCreator(compression);
      options.enableEncodingSelectionCache = true;
      break;
    case FixturePath::kLayoutTree:
      options.encodingLayoutTree.emplace(columnLayout(
          "i64_24",
          EncodingLayout{
              EncodingType::FixedBitWidth, {}, compression.compressionType}));
      break;
    case FixturePath::kReplayFallback:
      options.encodingLayoutTree.emplace(columnLayout(
          "i64_24",
          EncodingLayout{
              EncodingType::Constant, {}, CompressionType::Uncompressed}));
      break;
    case FixturePath::kSharedDictionary:
      options.encodingSelectionPolicyCreator =
          fixedBitWidthCreator(compression, /*dictionaryForIntegers=*/true);
      options.experimentalSharedDictionaryEncoding =
          SharedDictionaryEncodingConfig::builder()
              .addColumnDictionary(
                  "lowcard",
                  SharedDictionaryConfig{
                      .scope = SharedDictionaryScope::Stripe})
              .build();
      break;
    case FixturePath::kPinnedEncodings:
      options.encodingSelectionPolicyCreator =
          defaultFactorsCreator(compression);
      options.subIntSplitEncodingSubfields = {"split"};
      options.encodingLayoutTree.emplace(fsstColumnLayout(
          EncodingLayout{
              EncodingType::Fsst,
              {},
              compression.compressionType,
              {std::nullopt, std::nullopt},
          }));
      // Keeps FSST even where it does not shrink the strings, so that it is
      // written.
      options.fsstCompressionTargetRatio = 10.0;
      break;
  }
  return options;
}

WrittenFile writeFile(
    const std::vector<velox::RowVectorPtr>& batches,
    WriterOptions options,
    velox::memory::MemoryPool& rootPool) {
  options.enableChunking = true;
  options.minStreamChunkRawSize = 0;
  options.flushPolicyFactory = [] {
    return std::make_unique<LambdaFlushPolicy>(
        /*flushLambda=*/[](const StripeProgress&) { return false; },
        /*chunkLambda=*/[](const StripeProgress&) { return true; });
  };
  WrittenFile written;
  Writer writer(
      batches[0]->type(),
      std::make_unique<velox::InMemoryWriteFile>(&written.bytes),
      rootPool,
      std::move(options));
  for (size_t batch = 0; batch < batches.size(); ++batch) {
    writer.write(batches[batch]);
    if (endsStripe(batch)) {
      writer.flush();
    }
  }
  writer.close();
  for (const auto& [name, metric] : writer.runtimeStats()) {
    written.runtimeStatNames.push_back(name);
  }
  return written;
}

uint64_t fileDigest(std::string_view bytes) {
  return ChecksumFactory::create(ChecksumType::XXH3_64)
      ->computeChecksum64(bytes);
}

} // namespace facebook::nimble::test::options_off
