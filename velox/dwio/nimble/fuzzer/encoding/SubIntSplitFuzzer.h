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

#include <algorithm>
#include <bit>
#include <random>
#include <span>
#include <string_view>
#include <vector>

#include <gtest/gtest.h>

#include "folly/Random.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/SubIntSplitEncoding.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelection.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/encodings/selection/Statistics.h"
#include "velox/dwio/nimble/encodings/subintsplit/TuningConfig.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"
#include "velox/dwio/nimble/velox/RowRange.h"

/// Encode and read-back checks shared by the SubIntSplit fuzzers, which drive
/// SubIntSplitEncoding directly so they can pass a TuningConfig.

namespace facebook::nimble::test {

/// Encodes `data` as SubIntSplit under `options` and `tuning`, with the
/// default manual policy selecting each section's encoding.
template <typename T>
std::string_view encodeSubIntSplit(
    const Vector<T>& data,
    Buffer& buffer,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning) {
  using physicalType = typename TypeTraits<T>::physicalType;
  const std::span<const physicalType> values{
      reinterpret_cast<const physicalType*>(data.data()), data.size()};
  ManualEncodingSelectionPolicyFactory factory;
  EncodingSelection<physicalType> selection{
      EncodingSelectionResult{.encodingType = EncodingType::SubIntSplit},
      Statistics<physicalType>::create(values),
      factory.createPolicy(TypeTraits<physicalType>::dataType)};
  return SubIntSplitEncoding<T>::encode(
      selection, values, buffer, options, tuning);
}

/// Checks `length` decoded rows against `expected` from row `offset`,
/// comparing bit patterns so NaN payloads count.
template <typename T>
void expectRows(
    const Vector<T>& expected,
    const T* actual,
    uint32_t offset,
    uint32_t length) {
  using physicalType = typename TypeTraits<T>::physicalType;
  for (uint32_t i = 0; i < length; ++i) {
    ASSERT_EQ(
        std::bit_cast<physicalType>(actual[i]),
        std::bit_cast<physicalType>(expected[offset + i]))
        << "Mismatch at row " << offset + i;
  }
}

/// Reads `encoded` back under `tuning` through a whole-stream materialize,
/// random chunks with random skips between them, and a skip to a random row.
template <typename T>
void verifySubIntSplitReads(
    std::mt19937& rng,
    velox::memory::MemoryPool& pool,
    std::string_view encoded,
    const Vector<T>& data,
    const Encoding::Options& options,
    const subintsplit::TuningConfig& tuning) {
  const auto rowCount = static_cast<uint32_t>(data.size());
  SubIntSplitEncoding<T> encoding{pool, encoded, nullptr, options, tuning};
  ASSERT_EQ(encoding.rowCount(), rowCount);
  Vector<T> actual(&pool, rowCount);

  encoding.materialize(rowCount, actual.data());
  expectRows(data, actual.data(), 0, rowCount);

  // Random chunks with random skips between them.
  encoding.reset();
  uint32_t row{0};
  while (row < rowCount) {
    if (folly::Random::oneIn(3, rng)) {
      const uint32_t skip =
          std::min(rowCount - row, folly::Random::rand32(rng) % 300);
      encoding.skip(skip);
      row += skip;
      continue;
    }
    const uint32_t length =
        std::min(rowCount - row, 1 + folly::Random::rand32(rng) % 700);
    encoding.materialize(length, actual.data());
    expectRows(data, actual.data(), row, length);
    row += length;
  }

  encoding.reset();
  const uint32_t tail = folly::Random::rand32(rng) % rowCount;
  encoding.skip(tail);
  encoding.materialize(rowCount - tail, actual.data());
  expectRows(data, actual.data(), tail, rowCount - tail);
}

/// Reads `encoded` back through the EncodingView createEncodingView returns:
/// random point reads, one gather, one contiguous range and one list of
/// ordered, disjoint row ranges.
template <typename T>
void verifySubIntSplitView(
    std::mt19937& rng,
    velox::memory::MemoryPool& pool,
    std::string_view encoded,
    const Vector<T>& data,
    const Encoding::Options& options) {
  const auto rowCount = static_cast<uint32_t>(data.size());
  auto view = createEncodingView(encoded, &pool, options);
  ASSERT_NE(view, nullptr);
  ASSERT_EQ(view->rowCount(), rowCount);

  std::vector<uint32_t> rows;
  for (uint32_t i = 0; i < std::min<uint32_t>(rowCount, 128); ++i) {
    rows.push_back(folly::Random::rand32(rng) % rowCount);
  }
  rows.push_back(rowCount - 1);
  for (const auto row : rows) {
    T actual;
    view->readAt(row, &actual);
    expectRows(data, &actual, row, 1);
  }
  Vector<T> gathered(&pool, rows.size());
  view->readAt(rows, gathered.data());
  for (size_t i = 0; i < rows.size(); ++i) {
    expectRows(data, &gathered[i], rows[i], 1);
  }

  const uint32_t offset = folly::Random::rand32(rng) % rowCount;
  const uint32_t length = 1 + folly::Random::rand32(rng) % (rowCount - offset);
  Vector<T> contiguous(&pool, length);
  view->read(offset, length, contiguous.data());
  expectRows(data, contiguous.data(), offset, length);

  std::vector<RowRange> ranges;
  uint32_t row = folly::Random::rand32(rng) % std::min<uint32_t>(rowCount, 64);
  while (row < rowCount && ranges.size() < 64) {
    const uint32_t end =
        std::min(rowCount, row + 1 + folly::Random::rand32(rng) % 96);
    ranges.emplace_back(row, end);
    row = end + folly::Random::rand32(rng) % 128;
  }
  uint32_t rangeRows{0};
  for (const auto& range : ranges) {
    rangeRows += range.numRows();
  }
  Vector<T> ranged(&pool, rangeRows);
  ASSERT_EQ(
      view->read(
          ranges, [](uint32_t /*outputIndex*/) {}, ranged.data()),
      rangeRows);
  uint32_t outputIndex{0};
  for (const auto& range : ranges) {
    expectRows(
        data, ranged.data() + outputIndex, range.startRow, range.numRows());
    outputIndex += range.numRows();
  }
}

} // namespace facebook::nimble::test
