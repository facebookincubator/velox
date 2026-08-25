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

// Deterministic counterpart to NimbleWriterFuzzer. Nimble has four
// independent encoding read paths -- EncodingFactory, legacy::EncodingFactory,
// and the DefaultEncodingTrait and legacy::LegacyEncodingTrait visitor tables
// -- and an encoding can be writable while one of them cannot decode it.
// Adding ALP and FSST to the writer's candidate set previously required
// repairing two legacy tables by hand (D114295784).
//
// This test pins that invariant: for every (encoding, column type) pair the
// writer accepts, all four read paths must decode the file. It covers the same
// ground as the fuzzer but with fixed data and a fixed writer configuration, so
// it runs in seconds, names the failing pair exactly, and cannot flake.

#include <array>
#include <map>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <folly/init/Init.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzer.h"
#include "velox/dwio/nimble/fuzzer/encoding_selection/NimbleWriterFuzzerRunner.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::fuzzer {

// Must sit in the enum's own namespace, not the unnamed one below, for gtest
// to find it by ADL; otherwise a failing EXPECT_EQ prints the raw enum bytes.
void PrintTo(WriteOutcome outcome, std::ostream* stream) {
  switch (outcome) {
    case WriteOutcome::kApplied:
      *stream << "kApplied";
      return;
    case WriteOutcome::kNotApplied:
      *stream << "kNotApplied";
      return;
  }
  *stream << "unknown(" << static_cast<int>(outcome) << ")";
}

namespace {

using ::facebook::velox::VectorPtr;

constexpr velox::vector_size_t kNumRows = 256;

// A named single-column batch. Shapes are chosen so that between them every
// encoding family has at least one column it genuinely accepts: small-range
// repeating integers for the bit-packing and run-length families, decimal-like
// floats for ALP, a repeated token alphabet for FSST, and a single-valued
// column for Constant.
struct ColumnCase {
  std::string name;
  VectorPtr batch;
  // The Nimble DataType the column's values stream carries, which is what
  // decides whether an encoding is even applicable to it.
  DataType dataType;
  // Whether every non-null value is the same. ConstantEncoding refuses
  // anything else, so this is what decides its expected outcome.
  bool isSingleValued{false};
  // Whether the non-null values never decrease. DeltaBlockEncoding refuses
  // anything else (NIMBLE_CHECK_GE, "requires non-decreasing values").
  bool isNonDecreasing{false};
};

// The outcome a (column, encoding) pair must produce. Derived from the
// encoders' stated contracts rather than from a previous run, so a pair that
// stops applying to data it used to accept fails here instead of passing.
//
// Selection is estimator-gated, so a precondition miss shows up as
// kNotApplied: EncodingSizeEstimation declines the stream and the policy
// substitutes Trivial. The encoder is never reached, so it never refuses.
//
// Only the preconditions the column cases can actually trip are modelled. The
// rest hold by construction of the table, and a new case must preserve them or
// add its own rule here:
//
//  - kNumRows >= 2 and at most 97 distinct values, so Huffman clears both its
//    two-symbol minimum and kMaxSymbols.
//  - Frequency distributions stay near-uniform, so Huffman clears its 12-bit
//    code-length limit.
//  - kNumRows <= Encoding::Options::deltaBlockSize (default 256), so the whole
//    column is one DeltaBlock block and isNonDecreasing means what it says.
//    Past that, ordering is checked per block, and a column that decreases
//    only across a block boundary is accepted.
WriteOutcome expectedOutcome(
    const ColumnCase& columnCase,
    EncodingType encodingType) {
  if (encodingType == EncodingType::Constant && !columnCase.isSingleValued) {
    return WriteOutcome::kNotApplied;
  }
  if (encodingType == EncodingType::DeltaBlock && !columnCase.isNonDecreasing) {
    return WriteOutcome::kNotApplied;
  }
  // "Huffman encoding requires at least two symbols" -- a single-symbol
  // alphabet has no code to assign, and estimateSize declines it for the same
  // reason.
  if (encodingType == EncodingType::Huffman && columnCase.isSingleValued) {
    return WriteOutcome::kNotApplied;
  }
  return WriteOutcome::kApplied;
}

class EncodingDispatchConsistencyTest : public ::testing::Test {
 protected:
  void SetUp() override {
    rootPool_ = velox::memory::memoryManager()->addRootPool(
        "EncodingDispatchConsistencyTest");
    leafPool_ = rootPool_->addLeafChild("leaf");
    vectorMaker_ = std::make_unique<velox::test::VectorMaker>(leafPool_.get());
  }

  template <typename T>
  VectorPtr makeNumericColumn() {
    return vectorMaker_->rowVector(
        {"c0"}, {vectorMaker_->flatVector<T>(kNumRows, [](auto row) {
          return static_cast<T>((row / 4) % 32);
        })});
  }

  template <typename T>
  VectorPtr makeDecimalLikeColumn() {
    return vectorMaker_->rowVector(
        {"c0"}, {vectorMaker_->flatVector<T>(kNumRows, [](auto row) {
          return static_cast<T>(row % 97) / static_cast<T>(100);
        })});
  }

  std::vector<ColumnCase> makeColumnCases() {
    std::vector<ColumnCase> cases;
    cases.push_back(
        {.name = "bool",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<bool>(
                 kNumRows, [](auto row) { return (row / 8) % 2 == 0; })}),
         .dataType = DataType::Bool});
    cases.push_back(
        {.name = "tinyint",
         .batch = makeNumericColumn<int8_t>(),
         .dataType = DataType::Int8});
    cases.push_back(
        {.name = "smallint",
         .batch = makeNumericColumn<int16_t>(),
         .dataType = DataType::Int16});
    cases.push_back(
        {.name = "integer",
         .batch = makeNumericColumn<int32_t>(),
         .dataType = DataType::Int32});
    cases.push_back(
        {.name = "bigint",
         .batch = makeNumericColumn<int64_t>(),
         .dataType = DataType::Int64});
    // Non-decreasing: the one shape DeltaBlock accepts. makeNumericColumn
    // cycles, which DeltaBlock refuses outright.
    cases.push_back(
        {.name = "monotonic_bigint",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<int64_t>(
                 kNumRows,
                 [](auto row) { return static_cast<int64_t>(row / 2); })}),
         .dataType = DataType::Int64,
         .isNonDecreasing = true});
    cases.push_back(
        {.name = "real",
         .batch = makeDecimalLikeColumn<float>(),
         .dataType = DataType::Float});
    cases.push_back(
        {.name = "double",
         .batch = makeDecimalLikeColumn<double>(),
         .dataType = DataType::Double});
    cases.push_back(
        {.name = "varchar",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<velox::StringView>(
                 kNumRows,
                 [this](auto row) {
                   return velox::StringView(symbolRichValues()[row]);
                 })}),
         .dataType = DataType::String});
    cases.push_back(
        {.name = "constant_bigint",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<int64_t>(
                 kNumRows, [](auto /*row*/) { return int64_t{7}; })}),
         .dataType = DataType::Int64,
         .isSingleValued = true,
         .isNonDecreasing = true});
    // Null-bearing columns. Without them the forcing of Nullable::Data goes
    // untested here: selectNullable puts a Nullable wrapper at the top of the
    // stream and the requested encoding lands one level down, which is the
    // shape most fuzzer streams take.
    cases.push_back(
        {.name = "nullable_bigint",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<int64_t>(
                 kNumRows,
                 [](auto row) { return static_cast<int64_t>((row / 4) % 32); },
                 [](auto row) { return row % 7 == 0; })}),
         .dataType = DataType::Int64});
    cases.push_back(
        {.name = "nullable_varchar",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<velox::StringView>(
                 kNumRows,
                 [this](auto row) {
                   return velox::StringView(symbolRichValues()[row]);
                 },
                 [](auto row) { return row % 5 == 0; })}),
         .dataType = DataType::String});
    // Constant under a Nullable wrapper: the nulls travel in their own stream,
    // so the values stream is still single-valued and Constant must apply.
    cases.push_back(
        {.name = "nullable_constant_bigint",
         .batch = vectorMaker_->rowVector(
             {"c0"},
             {vectorMaker_->flatVector<int64_t>(
                 kNumRows,
                 [](auto /*row*/) { return int64_t{7}; },
                 [](auto row) { return row % 3 == 0; })}),
         .dataType = DataType::Int64,
         .isSingleValued = true,
         .isNonDecreasing = true});
    return cases;
  }

  // Strings built from a four-token alphabet so substrings repeat, which is
  // what FsstEncoding::estimateSize needs in order to accept the column.
  const std::vector<std::string>& symbolRichValues() {
    if (symbolRichValues_.empty()) {
      static const std::vector<std::string> kTokens = {
          "alphaalpha", "betabetaXX", "gammagamma", "deltadelta"};
      symbolRichValues_.reserve(kNumRows);
      for (velox::vector_size_t row = 0; row < kNumRows; ++row) {
        symbolRichValues_.push_back(
            kTokens[row % kTokens.size()] +
            kTokens[(row / 4) % kTokens.size()]);
      }
    }
    return symbolRichValues_;
  }

  std::shared_ptr<velox::memory::MemoryPool> rootPool_;
  std::shared_ptr<velox::memory::MemoryPool> leafPool_;
  std::unique_ptr<velox::test::VectorMaker> vectorMaker_;
  std::vector<std::string> symbolRichValues_;
};

TEST_F(
    EncodingDispatchConsistencyTest,
    withholdsIntegralOnlyEncodingsFromNullableFloat) {
  const auto batch = vectorMaker_->rowVector(
      {"c0"},
      {vectorMaker_->flatVector<float>(
          kNumRows,
          [](auto row) { return static_cast<float>(row); },
          [](auto row) { return row % 7 == 0; })});

  NimbleWriterFuzzerOptions options;
  options.seed = 1;
  options.randomizeWriterConfig = false;
  NimbleWriterFuzzer fuzzer(options, *rootPool_);

  // PFOR is integral-only but is also read-only, so it is rejected by the
  // writer config parser before dispatch is reached.
  // `TypesTest.ReadOnlyEncoding` covers that property.
  for (const auto encodingType :
       {EncodingType::DeltaBlock,
        EncodingType::SimdForBitpack,
        EncodingType::Huffman}) {
    SCOPED_TRACE(toString(encodingType));
    EXPECT_EQ(
        fuzzer.runFixed({batch}, encodingType), WriteOutcome::kNotApplied);
  }
}

TEST_F(
    EncodingDispatchConsistencyTest,
    unfilteredRoundsForceOnlyMissingEncodings) {
  NimbleWriterFuzzerOptions options;
  options.seed = 12'345;
  options.maxSchemaDepth = 1;
  options.batchSize = 32;
  options.numBatches = 1;
  options.randomizeWriterConfig = false;

  NimbleWriterFuzzer fuzzer(options, *rootPool_);
  fuzzer.run();

  EXPECT_EQ(fuzzer.numUnfilteredFilesWritten(), kNumUnfilteredRounds);
  for (const auto encodingType : allCandidateEncodings()) {
    SCOPED_TRACE(toString(encodingType));
    const auto entry = fuzzer.coverage().find(encodingType);
    ASSERT_NE(entry, fuzzer.coverage().end());
    const auto& stats = entry->second;

    EXPECT_EQ(
        stats.numForcedFilesWritten,
        stats.numUnfilteredFilesApplied == 0 ? 1 : 0);
    EXPECT_EQ(
        stats.numFilesOffered,
        (isIntegralOnlyEncoding(encodingType) ? 0 : kNumUnfilteredRounds) +
            stats.numForcedFilesWritten);
  }
}

TEST_F(
    EncodingDispatchConsistencyTest,
    everyWritableEncodingIsReadableEverywhere) {
  // Deliberately shares allCandidateEncodings() and isIntegralOnlyEncoding()
  // with the fuzzer: a private copy here would silently stop pinning any
  // encoding the fuzzer later adds.
  const auto encodings = allCandidateEncodings();

  NimbleWriterFuzzerOptions options;
  options.seed = 1;
  // Fixed writer configuration: with randomization on, whether a given
  // encoding is applied would depend on the draw order inside
  // randomizeWriterOptions, so inserting an unrelated random call could flip
  // an assertion below.
  options.randomizeWriterConfig = false;
  NimbleWriterFuzzer fuzzer(options, *rootPool_);

  // How many column shapes drove each (data type, encoding) pair.
  std::map<std::pair<DataType, EncodingType>, uint32_t> coveredPairs;

  for (const auto& columnCase : makeColumnCases()) {
    const bool isFloatingPoint = columnCase.dataType == DataType::Float ||
        columnCase.dataType == DataType::Double;
    for (const auto encodingType : encodings) {
      // Skip pairs the encoding can never handle: the encoder has no
      // implementation for the type, so the writer would silently use ordinary
      // selection and the assertion below would be about nothing.
      if (!isTypeCompatible(encodingType, columnCase.dataType)) {
        continue;
      }
      // Skip the four whose write-side gate admits floating point but whose
      // readers reject it -- see isIntegralOnlyEncoding and T283330065.
      if (isFloatingPoint && isIntegralOnlyEncoding(encodingType)) {
        continue;
      }
      SCOPED_TRACE(
          fmt::format(
              "column={} encoding={}",
              columnCase.name,
              toString(encodingType)));
      const auto outcome = fuzzer.runFixed({columnCase.batch}, encodingType);
      ++coveredPairs[{columnCase.dataType, encodingType}];
      EXPECT_EQ(outcome, expectedOutcome(columnCase, encodingType));

      // ColumnCase::dataType drives the skip logic above, so a wrong value
      // would silently exclude pairs instead of failing. An applied chunk
      // under that exact key is what proves the declaration matches the
      // stream the writer produced.
      if (outcome == WriteOutcome::kApplied) {
        const auto entry =
            fuzzer.pairCoverage().find({columnCase.dataType, encodingType});
        ASSERT_NE(entry, fuzzer.pairCoverage().end())
            << "Column declares " << toString(columnCase.dataType)
            << " but no stream of that type carried the encoding";
        EXPECT_GT(entry->second.numChunksApplied, 0u);
      }
    }
  }

  // The per-pair record, which the per-encoding table below cannot show: an
  // encoding applied to Int64 says nothing about whether it was ever driven on
  // Int8.
  std::map<DataType, std::vector<std::string>> encodingsByDataType;
  for (const auto& [pair, numColumnShapes] : coveredPairs) {
    encodingsByDataType[pair.first].emplace_back(toString(pair.second));
  }
  for (const auto& [dataType, encodingNames] : encodingsByDataType) {
    LOG(INFO) << fmt::format(
        "{:<8} driven on {} encodings: {}",
        toString(dataType),
        encodingNames.size(),
        fmt::join(encodingNames, ", "));
  }

  // Every DataType a leaf column can carry must have a shape above, or the
  // matrix quietly stops covering it when a scalar type is added to the
  // fuzzer. Uint32 is absent because only internal length and offset streams
  // carry it, and Uint16 because only a TIMESTAMP column's sub-microsecond
  // stream does -- neither is reachable from a single-column case here.
  static constexpr std::array<DataType, 8> kLeafDataTypes = {
      DataType::Bool,
      DataType::Int8,
      DataType::Int16,
      DataType::Int32,
      DataType::Int64,
      DataType::Float,
      DataType::Double,
      DataType::String};
  for (const auto dataType : kLeafDataTypes) {
    EXPECT_TRUE(encodingsByDataType.contains(dataType))
        << "No column case carries " << toString(dataType);
  }

  fuzzer.logCoverage();

  // Every encoding must have been genuinely applied by at least one column
  // shape. A zero here means the column cases no longer exercise that encoding
  // and the coverage this test claims is stale.
  for (const auto encodingType : encodings) {
    const auto entry = fuzzer.coverage().find(encodingType);
    ASSERT_NE(entry, fuzzer.coverage().end())
        << toString(encodingType) << " was never written";
    EXPECT_GT(entry->second.numChunksApplied, 0u)
        << toString(encodingType) << " was never actually applied to any chunk";
  }
}

} // namespace
} // namespace facebook::nimble::fuzzer

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);
  facebook::velox::memory::MemoryManager::initialize(
      facebook::velox::memory::MemoryManager::Options{});
  facebook::nimble::fuzzer::setUpFuzzerEnvironments();
  return RUN_ALL_TESTS();
}
