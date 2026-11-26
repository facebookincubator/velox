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

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <string_view>
#include <vector>

#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Exceptions.h"
#include "velox/dwio/nimble/common/Varint.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/FsstEncoding.h"
#include "velox/dwio/nimble/encodings/NullableEncoding.h"
#include "velox/dwio/nimble/encodings/TrivialEncoding.h"
#include "velox/dwio/nimble/encodings/common/EncodingFactory.h"
#include "velox/dwio/nimble/encodings/common/EncodingPrefix.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/fuzzer/encoding/EncodingFuzzer.h"

namespace facebook::nimble::test {

/// Reads every row of an FSST encoding. Returns the strings, or nullopt when
/// the read throws a NimbleException.
using FsstDecodedLengthsReader = std::function<std::optional<
    std::vector<std::string>>(std::string_view encoded, uint32_t numRows)>;

namespace detail {

enum class DecodedLengthsMutation {
  kUnchanged,
  kSetOneRow,
  kSwapTwoRows,
  kShuffle,
  kRotate,
  kDoubleAll,
  kZeroAll,
  kDropRow,
  kAddRow,
  kUint64,
  kString,
  kNullable,
};

constexpr std::array<DecodedLengthsMutation, 12> kDecodedLengthsMutations{
    DecodedLengthsMutation::kUnchanged,
    DecodedLengthsMutation::kSetOneRow,
    DecodedLengthsMutation::kSwapTwoRows,
    DecodedLengthsMutation::kShuffle,
    DecodedLengthsMutation::kRotate,
    DecodedLengthsMutation::kDoubleAll,
    DecodedLengthsMutation::kZeroAll,
    DecodedLengthsMutation::kDropRow,
    DecodedLengthsMutation::kAddRow,
    DecodedLengthsMutation::kUint64,
    DecodedLengthsMutation::kString,
    DecodedLengthsMutation::kNullable,
};

struct MutatedDecodedLengths {
  // The nested encoding that replaces the decoded lengths.
  std::string encoded;

  // The lengths that 'encoded' holds when it is non-null Uint32 values, one for
  // each row of the FSST encoding. Empty otherwise.
  std::vector<uint32_t> lengths;

  // Whether 'encoded' is non-null Uint32 values, one for each row.
  bool wellFormed{true};
};

template <typename T>
std::string encodeTrivialValues(
    velox::memory::MemoryPool& pool,
    const std::vector<T>& values,
    const Encoding::Options& options) {
  Vector<T> input{&pool, values.size()};
  std::copy(values.begin(), values.end(), input.begin());
  Buffer buffer{pool};
  return std::string{Encoder<TrivialEncoding<T>>::encode(
      buffer, input, CompressionType::Uncompressed, options)};
}

inline MutatedDecodedLengths mutateDecodedLengths(
    velox::memory::MemoryPool& pool,
    std::mt19937& rng,
    DecodedLengthsMutation mutation,
    std::vector<uint32_t> lengths,
    const Encoding::Options& options) {
  const auto wellFormed = [&](std::vector<uint32_t> values) {
    return MutatedDecodedLengths{
        encodeTrivialValues<uint32_t>(pool, values, options),
        std::move(values),
        true};
  };
  const auto numRows = lengths.size();
  switch (mutation) {
    case DecodedLengthsMutation::kUnchanged:
      return wellFormed(std::move(lengths));
    case DecodedLengthsMutation::kSetOneRow: {
      const auto row = rng() % numRows;
      const std::array<uint32_t, 7> values{
          0,
          lengths[row] + 1,
          lengths[row] > 0 ? lengths[row] - 1 : 2,
          lengths[row] * 2,
          1u << 20,
          std::numeric_limits<uint32_t>::max(),
          static_cast<uint32_t>(rng())};
      lengths[row] = values[rng() % values.size()];
      return wellFormed(std::move(lengths));
    }
    case DecodedLengthsMutation::kSwapTwoRows:
      std::swap(lengths[rng() % numRows], lengths[rng() % numRows]);
      return wellFormed(std::move(lengths));
    case DecodedLengthsMutation::kShuffle:
      std::shuffle(lengths.begin(), lengths.end(), rng);
      return wellFormed(std::move(lengths));
    case DecodedLengthsMutation::kRotate:
      std::rotate(lengths.begin(), lengths.begin() + 1, lengths.end());
      return wellFormed(std::move(lengths));
    case DecodedLengthsMutation::kDoubleAll:
      for (auto& length : lengths) {
        length *= 2;
      }
      return wellFormed(std::move(lengths));
    case DecodedLengthsMutation::kZeroAll:
      std::fill(lengths.begin(), lengths.end(), 0);
      return wellFormed(std::move(lengths));
    case DecodedLengthsMutation::kDropRow:
      if (numRows > 1) {
        lengths.pop_back();
      } else {
        lengths.push_back(static_cast<uint32_t>(rng()));
      }
      return {encodeTrivialValues<uint32_t>(pool, lengths, options), {}, false};
    case DecodedLengthsMutation::kAddRow:
      lengths.push_back(static_cast<uint32_t>(rng() % 100));
      return {encodeTrivialValues<uint32_t>(pool, lengths, options), {}, false};
    case DecodedLengthsMutation::kUint64:
      return {
          encodeTrivialValues<uint64_t>(
              pool,
              std::vector<uint64_t>(lengths.begin(), lengths.end()),
              options),
          {},
          false};
    case DecodedLengthsMutation::kString: {
      std::vector<std::string> storage;
      storage.reserve(numRows);
      for (const auto length : lengths) {
        storage.push_back(std::to_string(length));
      }
      return {
          encodeTrivialValues<std::string_view>(
              pool,
              std::vector<std::string_view>(storage.begin(), storage.end()),
              options),
          {},
          false};
    }
    case DecodedLengthsMutation::kNullable: {
      Vector<uint32_t> input{&pool, numRows};
      std::copy(lengths.begin(), lengths.end(), input.begin());
      Vector<bool> nonNulls{&pool, numRows};
      std::fill(nonNulls.begin(), nonNulls.end(), true);
      Buffer buffer{pool};
      return {
          std::string{Encoder<NullableEncoding<uint32_t>>::encodeNullable(
              buffer, input, nonNulls, CompressionType::Uncompressed, options)},
          {},
          false};
    }
  }
  NIMBLE_UNREACHABLE("Unknown decoded lengths mutation.");
}

// Rebuilds an FSST encoding with other compressed lengths or decoded lengths,
// and keeps everything else, including the bytes that precede the symbol
// table.
inline std::string replaceLengths(
    std::string_view encoded,
    std::optional<std::string_view> compressedLengths,
    std::optional<std::string_view> decodedLengths,
    bool useVarintRowCount) {
  const auto prefixSize =
      EncodingPrefix::prefixSize(encoded, useVarintRowCount);
  const auto header = FsstEncoding::parseHeader(encoded, prefixSize);
  const auto appendSection = [](std::string& out, std::string_view section) {
    char size[5];
    char* cursor = size;
    varint::writeVarint(static_cast<uint32_t>(section.size()), &cursor);
    out.append(size, cursor);
    out.append(section);
  };

  // The prefix and the compression type, which holds the layout flag.
  std::string rebuilt{encoded.substr(0, prefixSize + 1)};
  appendSection(rebuilt, header.symbolTable);
  appendSection(rebuilt, compressedLengths.value_or(header.lengths));
  appendSection(rebuilt, decodedLengths.value_or(header.decodedLengths));
  rebuilt.append(header.blob);
  return rebuilt;
}

} // namespace detail

/// Fuzzes the decoded lengths of FSST encodings. Each iteration encodes random
/// strings, replaces the decoded lengths of the encoding with others that a
/// writer never produces, and reads every row with 'read'.
///
/// A read must throw a NimbleException when the replacement is not non-null
/// Uint32 values for each row, when a length cannot belong to its compressed
/// string, and when the lengths do not add up to the decoded bytes. A read that
/// does not throw may only cut the decoded bytes in other places: its strings
/// have the lengths that the encoding holds, and their bytes joined are the
/// original ones. Unchanged lengths must read the original strings.
inline void fuzzFsstDecodedLengths(
    velox::memory::MemoryPool& pool,
    uint32_t seed,
    uint32_t iterations,
    uint32_t maxRows,
    const Encoding::Options& options,
    const FsstDecodedLengthsReader& read) {
  using detail::DecodedLengthsMutation;
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "FSST decoded lengths fuzzer seed: " << seed
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    Buffer dataBuffer{pool};
    const uint32_t numRows = 1 + rng() % maxRows;
    const auto data = makeFsstData(pool, rng, numRows, &dataBuffer, rng() % 2);
    Buffer encodeBuffer{pool};
    const std::string encoded{Encoder<FsstEncoding>::encode(
        encodeBuffer, data, CompressionType::Uncompressed, options)};
    ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);

    std::vector<std::string> original;
    std::vector<uint32_t> lengths;
    std::string originalBytes;
    for (const auto value : data) {
      original.emplace_back(value);
      lengths.push_back(static_cast<uint32_t>(value.size()));
      originalBytes.append(value);
    }
    const auto header = FsstEncoding::parseHeader(
        encoded,
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
    auto compressed = EncodingFactory().create(
        pool,
        header.lengths,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options);
    std::vector<uint32_t> compressedLengths(numRows);
    compressed->materialize(numRows, compressedLengths.data());

    const auto fits = [](uint32_t compressedLength, uint32_t decodedLength) {
      try {
        FsstEncoding::validateDecodedLength(compressedLength, decodedLength);
        return true;
      } catch (const NimbleException&) {
        return false;
      }
    };
    for (const auto mutation : detail::kDecodedLengthsMutations) {
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iteration=" << iteration
          << " rows=" << numRows << " mutation=" << static_cast<int>(mutation));
      const auto mutated =
          detail::mutateDecodedLengths(pool, rng, mutation, lengths, options);
      const auto corrupted = detail::replaceLengths(
          encoded,
          /*compressedLengths=*/std::nullopt,
          mutated.encoded,
          options.useVarintRowCount);
      const auto actual = read(corrupted, numRows);

      bool mustReject = !mutated.wellFormed;
      if (mutated.wellFormed) {
        mustReject =
            std::accumulate(
                mutated.lengths.begin(), mutated.lengths.end(), uint64_t{0}) !=
            originalBytes.size();
        for (uint32_t row = 0; row < numRows && !mustReject; ++row) {
          mustReject = !fits(compressedLengths[row], mutated.lengths[row]);
        }
      }
      if (mustReject) {
        EXPECT_FALSE(actual.has_value());
        continue;
      }
      if (mutation == DecodedLengthsMutation::kUnchanged) {
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(*actual, original);
        continue;
      }
      if (!actual.has_value()) {
        continue;
      }
      ASSERT_EQ(actual->size(), numRows);
      std::string joined;
      for (uint32_t row = 0; row < numRows; ++row) {
        EXPECT_EQ((*actual)[row].size(), mutated.lengths[row]) << "row " << row;
        joined.append((*actual)[row]);
      }
      EXPECT_EQ(joined, originalBytes);
    }
  }
}

/// Fuzzes the compressed lengths of FSST encodings the way
/// fuzzFsstDecodedLengths fuzzes the decoded lengths: each iteration replaces
/// the compressed lengths of an encoding and reads every row with 'read'.
///
/// A read must throw a NimbleException when the replacement is not non-null
/// Uint32 values for each row, and when the lengths do not add up to the size
/// of the blob. A read that does not throw returns the original strings: the
/// decoded lengths cut the same decoded bytes, whatever the compressed lengths
/// split. Unchanged lengths must read the original strings.
inline void fuzzFsstCompressedLengths(
    velox::memory::MemoryPool& pool,
    uint32_t seed,
    uint32_t iterations,
    uint32_t maxRows,
    const Encoding::Options& options,
    const FsstDecodedLengthsReader& read) {
  using detail::DecodedLengthsMutation;
  if (seed == 0) {
    seed = folly::Random::rand32();
  }
  LOG(INFO) << "FSST compressed lengths fuzzer seed: " << seed
            << " iterations: " << iterations << " maxRows: " << maxRows;
  std::mt19937 rng(seed);
  for (uint32_t iteration = 0; iteration < iterations; ++iteration) {
    Buffer dataBuffer{pool};
    const uint32_t numRows = 1 + rng() % maxRows;
    const auto data = makeFsstData(pool, rng, numRows, &dataBuffer, rng() % 2);
    Buffer encodeBuffer{pool};
    const std::string encoded{Encoder<FsstEncoding>::encode(
        encodeBuffer, data, CompressionType::Uncompressed, options)};
    ASSERT_EQ(EncodingPrefix::encodingType(encoded), EncodingType::Fsst);

    std::vector<std::string> original;
    for (const auto value : data) {
      original.emplace_back(value);
    }
    const auto header = FsstEncoding::parseHeader(
        encoded,
        EncodingPrefix::prefixSize(encoded, options.useVarintRowCount));
    auto compressed = EncodingFactory().create(
        pool,
        header.lengths,
        [](uint32_t /*totalLength*/) -> void* { return nullptr; },
        options);
    std::vector<uint32_t> compressedLengths(numRows);
    compressed->materialize(numRows, compressedLengths.data());
    const uint64_t blobSize = std::accumulate(
        compressedLengths.begin(), compressedLengths.end(), uint64_t{0});

    for (const auto mutation : detail::kDecodedLengthsMutations) {
      SCOPED_TRACE(
          ::testing::Message()
          << "seed=" << seed << " iteration=" << iteration
          << " rows=" << numRows << " mutation=" << static_cast<int>(mutation));
      const auto mutated = detail::mutateDecodedLengths(
          pool, rng, mutation, compressedLengths, options);
      const auto corrupted = detail::replaceLengths(
          encoded,
          mutated.encoded,
          /*decodedLengths=*/std::nullopt,
          options.useVarintRowCount);
      const auto actual = read(corrupted, numRows);

      const bool mustReject = !mutated.wellFormed ||
          std::accumulate(
              mutated.lengths.begin(), mutated.lengths.end(), uint64_t{0}) !=
              blobSize;
      if (mustReject) {
        EXPECT_FALSE(actual.has_value());
        continue;
      }
      if (mutation == DecodedLengthsMutation::kUnchanged) {
        ASSERT_TRUE(actual.has_value());
        EXPECT_EQ(*actual, original);
        continue;
      }
      if (actual.has_value()) {
        EXPECT_EQ(*actual, original);
      }
    }
  }
}

} // namespace facebook::nimble::test
