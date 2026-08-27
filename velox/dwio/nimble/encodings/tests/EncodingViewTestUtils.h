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
#include <atomic>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "fmt/core.h"
#include "velox/common/memory/Memory.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/common/Encoding.h"
#include "velox/dwio/nimble/encodings/tests/TestUtils.h"
#include "velox/dwio/nimble/encodings/views/EncodingViewFactory.h"

namespace facebook::nimble::test {

class EncodingViewTest : public ::testing::Test {
 protected:
  static constexpr uint32_t kConcurrentRows = 1024;

  void SetUp() override {
    pool_ = facebook::velox::memory::deprecatedAddDefaultLeafMemoryPool();
    buffer_ = std::make_unique<nimble::Buffer>(*pool_);
  }

  template <typename T>
  nimble::Vector<T> makeVector(std::initializer_list<T> values) {
    nimble::Vector<T> out{pool_.get()};
    out.insert(out.end(), values.begin(), values.end());
    return out;
  }

  template <typename Encoding>
  void expectReads(
      const nimble::Vector<typename Encoding::cppDataType>& values,
      const std::vector<uint32_t>& positions,
      nimble::Encoding::Options baseOptions = {},
      // Nested encodings are forced to Trivial by default. Encodings whose
      // point is the sub-encodings they select, such as SubIntSplit, need the
      // real policy to produce a representative stream.
      bool realNestedSelection = false) {
    using T = typename Encoding::cppDataType;
    for (const auto useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      auto options = baseOptions;
      options.useVarintRowCount = useVarint;
      auto serialized = nimble::test::Encoder<Encoding>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options,
          realNestedSelection);
      auto view = nimble::createEncodingView(serialized, pool_.get(), options);
      ASSERT_NE(view, nullptr);
      for (const auto position : positions) {
        SCOPED_TRACE(fmt::format("position={}", position));
        T value;
        view->readAt(position, &value);
        EXPECT_EQ(value, values[position]);
      }
      const auto rowCount = static_cast<uint32_t>(values.size());
      expectRangeRead(*view, values, /*offset=*/0, /*length=*/0);
      expectRangeRead(*view, values, rowCount, /*length=*/0);
      expectRangeRead(
          *view, values, /*offset=*/0, std::min<uint32_t>(rowCount, 3));
      if (rowCount > 0) {
        const auto tailLength = std::min<uint32_t>(rowCount, 3);
        expectRangeRead(*view, values, rowCount - tailLength, tailLength);
      }
    }
  }

  template <typename Encoding>
  void expectConcurrentReads(
      const nimble::Vector<typename Encoding::cppDataType>& values,
      const std::vector<uint32_t>& positions,
      nimble::Encoding::Options baseOptions = {},
      // Nested encodings are forced to Trivial by default. Encodings whose
      // point is the sub-encodings they select, such as SubIntSplit, need the
      // real policy to produce a representative stream.
      bool realNestedSelection = false) {
    using T = typename Encoding::cppDataType;
    for (const auto useVarint : {false, true}) {
      SCOPED_TRACE(fmt::format("useVarint={}", useVarint));
      auto options = baseOptions;
      options.useVarintRowCount = useVarint;
      auto serialized = nimble::test::Encoder<Encoding>::encode(
          *buffer_,
          values,
          nimble::CompressionType::Uncompressed,
          options,
          realNestedSelection);
      auto view = nimble::createEncodingView(serialized, pool_.get(), options);
      ASSERT_NE(view, nullptr);

      constexpr auto kThreadCount = 8;
      constexpr auto kIterationCount = 64;
      std::atomic<bool> failed{false};
      std::vector<std::thread> threads;
      threads.reserve(kThreadCount);
      for (auto threadIndex = 0; threadIndex < kThreadCount; ++threadIndex) {
        threads.emplace_back([&, threadIndex] {
          std::mt19937 rng{
              positions[threadIndex % positions.size()] + threadIndex};
          std::uniform_int_distribution<size_t> positionIndex{
              0, positions.size() - 1};
          for (auto iteration = 0; iteration < kIterationCount; ++iteration) {
            for (size_t read = 0; read < positions.size() / kThreadCount;
                 ++read) {
              const auto position = positions[positionIndex(rng)];
              T value;
              view->readAt(position, &value);
              if (value != values[position]) {
                failed.store(true, std::memory_order_relaxed);
                return;
              }
            }
          }
        });
      }

      for (auto& thread : threads) {
        thread.join();
      }
      EXPECT_FALSE(failed.load(std::memory_order_relaxed));
    }
  }

  std::vector<uint32_t> randomizedPositions(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint32_t> position{0, kConcurrentRows - 1};
    std::vector<uint32_t> positions;
    positions.reserve(4096);
    for (uint32_t i = 0; i < 4096; ++i) {
      positions.push_back(position(rng));
    }
    return positions;
  }

  nimble::Vector<int32_t> constantInt32(int32_t value) {
    nimble::Vector<int32_t> values{pool_.get()};
    values.resize(kConcurrentRows, value);
    return values;
  }

  template <typename T>
  void expectRangeRead(
      const nimble::EncodingView& view,
      const nimble::Vector<T>& values,
      uint32_t offset,
      uint32_t length) {
    SCOPED_TRACE(fmt::format("offset={}, length={}", offset, length));
    using PhysicalType = typename nimble::TypeTraits<T>::physicalType;
    nimble::Vector<PhysicalType> actual{pool_.get(), length};
    view.read(offset, length, actual.data());
    const auto* expected =
        reinterpret_cast<const PhysicalType*>(values.data()) + offset;
    EXPECT_EQ(
        std::vector<PhysicalType>(actual.begin(), actual.end()),
        std::vector<PhysicalType>(expected, expected + length));
  }

  nimble::Vector<int32_t> randomInt32(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int32_t> value{-1024, 1024};
    nimble::Vector<int32_t> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back(value(rng));
    }
    return values;
  }

  template <typename T>
  nimble::Vector<T> randomNarrowUnsigned(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint32_t> value{0, 63};
    nimble::Vector<T> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back(static_cast<T>(value(rng)));
    }
    return values;
  }

  nimble::Vector<uint32_t> randomPforData(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint32_t> narrowValue{0, 31};
    std::uniform_int_distribution<uint32_t> exceptionValue{10000, 12000};
    nimble::Vector<uint32_t> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back(i % 23 == 0 ? exceptionValue(rng) : narrowValue(rng));
    }
    return values;
  }

  template <typename T>
  nimble::Vector<T> randomAlpData(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int32_t> value{-400, 400};
    nimble::Vector<T> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back(static_cast<T>(value(rng)) / 4);
    }
    return values;
  }

  nimble::Vector<bool> randomBool(uint32_t seed) {
    std::mt19937 rng{seed};
    nimble::Vector<bool> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back((rng() % 5) == 0);
    }
    return values;
  }

  nimble::Vector<int32_t> randomRleInt32(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<int32_t> value{-32, 32};
    std::uniform_int_distribution<uint32_t> runLength{1, 9};
    nimble::Vector<int32_t> values{pool_.get()};
    values.reserve(kConcurrentRows);
    while (values.size() < kConcurrentRows) {
      const auto runValue = value(rng);
      const auto count = std::min<uint32_t>(
          runLength(rng),
          static_cast<uint32_t>(kConcurrentRows - values.size()));
      for (uint32_t i = 0; i < count; ++i) {
        values.push_back(runValue);
      }
    }
    return values;
  }

  nimble::Vector<bool> randomRleBool(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint32_t> runLength{1, 9};
    nimble::Vector<bool> values{pool_.get()};
    values.reserve(kConcurrentRows);
    bool value = (rng() % 2) == 0;
    while (values.size() < kConcurrentRows) {
      const auto count = std::min<uint32_t>(
          runLength(rng),
          static_cast<uint32_t>(kConcurrentRows - values.size()));
      for (uint32_t i = 0; i < count; ++i) {
        values.push_back(value);
      }
      value = !value;
    }
    return values;
  }

  nimble::Vector<uint32_t> randomDictionaryUint32(uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint32_t> index{0, 15};
    nimble::Vector<uint32_t> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      values.push_back(1000 + index(rng) * 17);
    }
    return values;
  }

  nimble::Vector<std::string_view> randomStringViews(
      std::vector<std::string>& backing,
      uint32_t seed) {
    std::mt19937 rng{seed};
    std::uniform_int_distribution<uint32_t> suffix{0, 31};
    backing.clear();
    backing.reserve(kConcurrentRows);
    nimble::Vector<std::string_view> values{pool_.get()};
    values.reserve(kConcurrentRows);
    for (uint32_t i = 0; i < kConcurrentRows; ++i) {
      backing.push_back(fmt::format("value-{}", suffix(rng)));
      values.push_back(backing.back());
    }
    return values;
  }

  std::shared_ptr<velox::memory::MemoryPool> pool_;
  std::unique_ptr<nimble::Buffer> buffer_;
};

} // namespace facebook::nimble::test
