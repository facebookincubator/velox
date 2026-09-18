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
#include "velox/dwio/nimble/fuzzer/stripe_stats_pruning/StripeStatsPruningFuzzer.h"

#include <optional>
#include <random>
#include <vector>

#include <fmt/format.h>
#include <folly/ScopeGuard.h>

#include "velox/common/file/File.h"
#include "velox/dwio/common/BufferedInput.h"
#include "velox/dwio/common/Reader.h"
#include "velox/dwio/common/ReaderFactory.h"
#include "velox/dwio/common/ScanSpec.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/FeatureGate.h"
#include "velox/dwio/nimble/common/tests/NimbleFileWriter.h"
#include "velox/dwio/nimble/common/tests/ScopedFeatureGate.h"
#include "velox/dwio/nimble/velox/selective/SelectiveNimbleReader.h"
#include "velox/dwio/nimble/writer/FlushPolicy.h"
#include "velox/dwio/nimble/writer/WriterOptions.h"
#include "velox/type/Filter.h"
#include "velox/vector/DecodedVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::nimble::fuzzer {
namespace {

using namespace facebook::velox;

// Value families the reader prunes on. Each maps to a different Filter, so the
// fuzzer has to generate and verify each one separately.
enum class KeyKind { kIntegral, kFloating, kString };

struct KeySpec {
  TypePtr type;
  KeyKind kind;
  // Values in one stripe stay inside a single band, so stripe min/max separate
  // and pruning has something to act on. Width is bounded by the narrowest
  // value the type holds.
  int64_t bandWidth;
  int32_t numBands;
};

const std::vector<KeySpec>& keySpecs() {
  static const std::vector<KeySpec> kSpecs = {
      {TINYINT(), KeyKind::kIntegral, 8, 8},
      {SMALLINT(), KeyKind::kIntegral, 100, 20},
      {INTEGER(), KeyKind::kIntegral, 1'000, 20},
      {BIGINT(), KeyKind::kIntegral, 1'000, 20},
      {REAL(), KeyKind::kFloating, 1'000, 20},
      {DOUBLE(), KeyKind::kFloating, 1'000, 20},
      {VARCHAR(), KeyKind::kString, 1'000, 50},
  };
  return kSpecs;
}

// Fixed width keeps lexicographic order equal to numeric order, so a byte range
// selects the same rows the band arithmetic predicts.
std::string stringKey(int64_t value) {
  return fmt::format("{:08d}", value);
}

// Payload columns exist to vary the schema around the key column; their values
// are never asserted on.
const std::vector<TypePtr>& payloadTypes() {
  static const std::vector<TypePtr> kTypes = {
      BIGINT(), INTEGER(), DOUBLE(), VARCHAR(), BOOLEAN()};
  return kTypes;
}

int64_t nextInRange(std::mt19937& rng, int64_t exclusiveUpper) {
  return static_cast<int64_t>(rng() % static_cast<uint64_t>(exclusiveUpper));
}

} // namespace

StripeStatsPruningFuzzerStats StripeStatsPruningFuzzer::run() {
  // The writer only emits the stripe-stats section, and the reader only prunes
  // on it, when both gates are on. Without this the fuzzer silently degrades
  // into a plain round-trip test.
  test::ScopedFeatureGate stripeStatsGate{
      FeatureGate::FeatureSet::kStripeStatsWrite,
      FeatureGate::FeatureSet::kStripeStatsPruning};

  // The selective reader registers itself through an init hook that a plain
  // library link does not run, so the factory has to be installed explicitly.
  registerSelectiveNimbleReaderFactory();
  const auto unregisterFactory =
      folly::makeGuard([] { unregisterSelectiveNimbleReaderFactory(); });

  StripeStatsPruningFuzzerStats stats;
  std::mt19937 rng(options_.seed);

  auto leafPool = rootPool_.addLeafChild("stripe_stats_pruning_fuzzer");
  velox::test::VectorMaker vectorMaker{leafPool.get()};

  for (int32_t iteration = 0; iteration < options_.numIterations; ++iteration) {
    const auto& keySpec = keySpecs()[rng() % keySpecs().size()];
    const auto numStripes = 2 + nextInRange(rng, 6);
    const bool keyHasNulls = nextInRange(rng, 3) == 0;
    const bool nullAllowed = nextInRange(rng, 2) == 0;

    // Every key value written, in file order, so the predicate can be replayed
    // over the whole file independently of what the reader returned.
    std::vector<std::optional<int64_t>> writtenKeys;
    std::vector<VectorPtr> stripeVectors;
    RowTypePtr rowType;

    const auto numPayloadColumns = nextInRange(rng, 3);
    std::vector<TypePtr> payloadColumnTypes;
    payloadColumnTypes.reserve(numPayloadColumns);
    for (int64_t column = 0; column < numPayloadColumns; ++column) {
      payloadColumnTypes.push_back(
          payloadTypes()[rng() % payloadTypes().size()]);
    }

    for (int64_t stripe = 0; stripe < numStripes; ++stripe) {
      const auto numRows = 20 + nextInRange(rng, 80);
      const auto band = nextInRange(rng, keySpec.numBands);

      std::vector<std::optional<int64_t>> keyValues;
      keyValues.reserve(numRows);
      for (int64_t row = 0; row < numRows; ++row) {
        if (keyHasNulls && row % 7 == 3) {
          keyValues.emplace_back(std::nullopt);
        } else {
          keyValues.emplace_back(
              band * keySpec.bandWidth + nextInRange(rng, keySpec.bandWidth));
        }
      }
      writtenKeys.insert(writtenKeys.end(), keyValues.begin(), keyValues.end());

      VectorPtr keyVector;
      switch (keySpec.kind) {
        case KeyKind::kIntegral:
          if (keySpec.type->isTinyint()) {
            std::vector<std::optional<int8_t>> typed;
            typed.reserve(keyValues.size());
            for (const auto& value : keyValues) {
              typed.push_back(
                  value.has_value() ? std::optional<int8_t>(
                                          static_cast<int8_t>(value.value()))
                                    : std::nullopt);
            }
            keyVector = vectorMaker.flatVectorNullable<int8_t>(typed);
          } else if (keySpec.type->isSmallint()) {
            std::vector<std::optional<int16_t>> typed;
            typed.reserve(keyValues.size());
            for (const auto& value : keyValues) {
              typed.push_back(
                  value.has_value() ? std::optional<int16_t>(
                                          static_cast<int16_t>(value.value()))
                                    : std::nullopt);
            }
            keyVector = vectorMaker.flatVectorNullable<int16_t>(typed);
          } else if (keySpec.type->isInteger()) {
            std::vector<std::optional<int32_t>> typed;
            typed.reserve(keyValues.size());
            for (const auto& value : keyValues) {
              typed.push_back(
                  value.has_value() ? std::optional<int32_t>(
                                          static_cast<int32_t>(value.value()))
                                    : std::nullopt);
            }
            keyVector = vectorMaker.flatVectorNullable<int32_t>(typed);
          } else {
            keyVector = vectorMaker.flatVectorNullable<int64_t>(keyValues);
          }
          break;
        case KeyKind::kFloating:
          if (keySpec.type->isReal()) {
            std::vector<std::optional<float>> typed;
            typed.reserve(keyValues.size());
            for (const auto& value : keyValues) {
              typed.push_back(
                  value.has_value()
                      ? std::optional<float>(static_cast<float>(value.value()))
                      : std::nullopt);
            }
            keyVector = vectorMaker.flatVectorNullable<float>(typed);
          } else {
            std::vector<std::optional<double>> typed;
            typed.reserve(keyValues.size());
            for (const auto& value : keyValues) {
              typed.push_back(
                  value.has_value() ? std::optional<double>(
                                          static_cast<double>(value.value()))
                                    : std::nullopt);
            }
            keyVector = vectorMaker.flatVectorNullable<double>(typed);
          }
          break;
        case KeyKind::kString: {
          std::vector<std::optional<std::string>> typed;
          typed.reserve(keyValues.size());
          for (const auto& value : keyValues) {
            typed.push_back(
                value.has_value()
                    ? std::optional<std::string>(stringKey(value.value()))
                    : std::nullopt);
          }
          keyVector = vectorMaker.flatVectorNullable<std::string>(typed);
          break;
        }
      }

      std::vector<VectorPtr> children{keyVector};
      if (!payloadColumnTypes.empty()) {
        VectorFuzzer::Options fuzzerOptions;
        fuzzerOptions.vectorSize = static_cast<size_t>(numRows);
        fuzzerOptions.nullRatio = 0.1;
        fuzzerOptions.stringLength = 16;
        VectorFuzzer payloadFuzzer{
            fuzzerOptions, leafPool.get(), static_cast<size_t>(rng())};
        for (const auto& payloadType : payloadColumnTypes) {
          children.push_back(payloadFuzzer.fuzzFlat(payloadType));
        }
      }
      auto stripeVector = vectorMaker.rowVector(children);
      if (rowType == nullptr) {
        rowType = asRowType(stripeVector->type());
      }
      stripeVectors.push_back(std::move(stripeVector));
    }

    // One stripe per written batch, so per-stripe statistics exist to prune on.
    WriterOptions writerOptions;
    writerOptions.enableVectorizedStats = true;
    writerOptions.flushPolicyFactory = [] {
      return std::make_unique<LambdaFlushPolicy>(
          [](const StripeProgress&) { return true; });
    };
    const auto fileContent = test::createNimbleFile(
        *leafPool, stripeVectors, writerOptions, /*flushAfterWrite=*/false);

    const auto lowerBand = nextInRange(rng, keySpec.numBands);
    const int64_t lower = lowerBand * keySpec.bandWidth;
    const int64_t upper = lower + nextInRange(rng, 2 * keySpec.bandWidth);

    auto scanSpec = std::make_shared<common::ScanSpec>("root");
    scanSpec->addAllChildFields(*rowType);

    std::unique_ptr<common::Filter> filter;
    switch (keySpec.kind) {
      case KeyKind::kIntegral:
        filter =
            std::make_unique<common::BigintRange>(lower, upper, nullAllowed);
        break;
      case KeyKind::kFloating:
        // The filter kind has to match the column type: a REAL column paired
        // with a DoubleRange is rejected rather than evaluated.
        if (keySpec.type->isReal()) {
          filter = std::make_unique<common::FloatRange>(
              static_cast<float>(lower),
              /*lowerUnbounded=*/false,
              /*lowerExclusive=*/false,
              static_cast<float>(upper),
              /*upperUnbounded=*/false,
              /*upperExclusive=*/false,
              nullAllowed);
        } else {
          filter = std::make_unique<common::DoubleRange>(
              static_cast<double>(lower),
              /*lowerUnbounded=*/false,
              /*lowerExclusive=*/false,
              static_cast<double>(upper),
              /*upperUnbounded=*/false,
              /*upperExclusive=*/false,
              nullAllowed);
        }
        break;
      case KeyKind::kString:
        filter = std::make_unique<common::BytesRange>(
            stringKey(lower),
            /*lowerUnbounded=*/false,
            /*lowerExclusive=*/false,
            stringKey(upper),
            /*upperUnbounded=*/false,
            /*upperExclusive=*/false,
            nullAllowed);
        break;
    }
    scanSpec->childByName(rowType->nameOf(0))->setFilter(std::move(filter));

    // The predicate the reader must agree with, evaluated over every row that
    // was written rather than over what came back.
    std::vector<std::optional<int64_t>> expectedKeys;
    for (const auto& value : writtenKeys) {
      if (!value.has_value()) {
        if (nullAllowed) {
          expectedKeys.emplace_back(std::nullopt);
        }
        continue;
      }
      if (value.value() >= lower && value.value() <= upper) {
        expectedKeys.emplace_back(value);
      }
    }

    auto readFile = std::make_shared<InMemoryReadFile>(fileContent);
    auto factory =
        dwio::common::getReaderFactory(dwio::common::FileFormat::NIMBLE);
    dwio::common::ReaderOptions readerOptions(leafPool.get());
    readerOptions.setScanSpec(scanSpec);
    auto reader = factory->createReader(
        std::make_unique<dwio::common::BufferedInput>(readFile, *leafPool),
        readerOptions);

    dwio::common::RowReaderOptions rowOptions;
    rowOptions.setScanSpec(scanSpec);
    rowOptions.setRequestedType(rowType);
    auto rowReader = reader->createRowReader(rowOptions);

    std::vector<std::optional<int64_t>> readKeys;
    auto result = BaseVector::create(rowType, 0, leafPool.get());
    // The string reader hands back a dictionary when values repeat within a
    // stripe, which banding makes likely, so the key column is decoded rather
    // than assumed flat.
    DecodedVector decodedKey;
    while (rowReader->next(7, result) > 0) {
      auto* rowResult = result->as<RowVector>();
      decodedKey.decode(*rowResult->childAt(0)->loadedVector());
      for (vector_size_t row = 0; row < rowResult->size(); ++row) {
        if (decodedKey.isNullAt(row)) {
          readKeys.emplace_back(std::nullopt);
          continue;
        }
        switch (keySpec.kind) {
          case KeyKind::kIntegral:
            if (keySpec.type->isTinyint()) {
              readKeys.emplace_back(decodedKey.valueAt<int8_t>(row));
            } else if (keySpec.type->isSmallint()) {
              readKeys.emplace_back(decodedKey.valueAt<int16_t>(row));
            } else if (keySpec.type->isInteger()) {
              readKeys.emplace_back(decodedKey.valueAt<int32_t>(row));
            } else {
              readKeys.emplace_back(decodedKey.valueAt<int64_t>(row));
            }
            break;
          case KeyKind::kFloating:
            if (keySpec.type->isReal()) {
              readKeys.emplace_back(
                  static_cast<int64_t>(decodedKey.valueAt<float>(row)));
            } else {
              readKeys.emplace_back(
                  static_cast<int64_t>(decodedKey.valueAt<double>(row)));
            }
            break;
          case KeyKind::kString:
            readKeys.emplace_back(
                std::stoll(decodedKey.valueAt<StringView>(row).str()));
            break;
        }
      }
    }

    NIMBLE_CHECK_EQ(
        readKeys.size(),
        expectedKeys.size(),
        "Pruned read returned a different number of rows than the predicate "
        "admits; a stripe was skipped wrongly (seed {}, iteration {}, type {}, "
        "nulls {}, nullAllowed {}).",
        options_.seed,
        iteration,
        keySpec.type->toString(),
        keyHasNulls,
        nullAllowed);
    NIMBLE_CHECK(
        readKeys == expectedKeys,
        fmt::format(
            "Pruned read returned different values than the predicate admits "
            "(seed {}, iteration {}, type {}).",
            options_.seed,
            iteration,
            keySpec.type->toString()));

    dwio::common::RuntimeStats runtimeStats;
    rowReader->updateRuntimeStats(runtimeStats);
    stats.numStripesSkipped += runtimeStats.skippedStrides;
    stats.numIterations++;
    if (expectedKeys.empty()) {
      stats.numEmptyResultIterations++;
    }
    if (keyHasNulls) {
      stats.numNullableKeyIterations++;
    }
  }

  return stats;
}

} // namespace facebook::nimble::fuzzer
