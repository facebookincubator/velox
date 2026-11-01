/*
 * Copyright (c) Facebook, Inc. and its affiliates.
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

#include "velox/experimental/cudf/exec/GpuResources.h"
#include "velox/experimental/cudf/exec/PrestoPageDevice.h"

#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <rmm/device_uvector.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

using detail::StringOffsets;

constexpr int64_t kNoRowCap = std::numeric_limits<int64_t>::max();

// Checks the device primitives on hand-built, sliced inputs: offsets children
// that start before the column's first row and masks whose first bit is not
// word aligned, which operator-level tests do not reach.
class PrestoPageDeviceTest : public testing::Test {
 protected:
  template <typename T>
  rmm::device_uvector<T> toDevice(const std::vector<T>& values) {
    return cudf::detail::make_device_uvector(
        cudf::host_span<const T>{values.data(), values.size()}, stream_, mr_);
  }

  template <typename T>
  std::vector<T> toHost(const rmm::device_uvector<T>& values) {
    return cudf::detail::make_std_vector(
        cudf::device_span<const T>{values.data(), values.size()}, stream_);
  }

  // Validity mask over 'numBits' bits.
  static std::vector<cudf::bitmask_type> makeMask(
      int64_t numBits,
      const std::function<bool(int64_t)>& valid) {
    std::vector<cudf::bitmask_type> words((numBits + 31) / 32, 0);
    for (int64_t i = 0; i < numBits; ++i) {
      if (valid(i)) {
        words[i / 32] |= 1u << (i % 32);
      }
    }
    return words;
  }

  std::vector<int64_t> bounds(
      int32_t numRows,
      int64_t fixedBitsPerRow,
      const std::vector<StringOffsets>& strings,
      int64_t pageBytes,
      int64_t maxRows) {
    return detail::prestoPageBoundaries(
        numRows, fixedBitsPerRow, strings, pageBytes, maxRows, stream_, mr_);
  }

  // Offsets of 8 strings that start at index 2: the strings are 3, 0, 10, 1,
  // 1, 40, 2 and 2 bytes long, 59 bytes in all.
  const std::vector<int32_t>
      offsets32_{0, 5, 9, 12, 12, 22, 23, 24, 64, 66, 68};
  const std::vector<int64_t> offsets64_{offsets32_.begin(), offsets32_.end()};
  static constexpr int64_t kFirstIndex = 2;
  static constexpr int32_t kNumStrings = 8;

  cuda::stream_ref stream_{cudfGlobalStreamPool().get_stream()};
  rmm::device_async_resource_ref mr_{cudf::get_current_device_resource_ref()};
};

TEST_F(PrestoPageDeviceTest, pageBoundariesWithoutStrings) {
  using Bounds = std::vector<int64_t>;
  // Equal rows make equal pages: 64-bit rows, 16-byte pages.
  EXPECT_EQ(bounds(10, 64, {}, 16, kNoRowCap), (Bounds{0, 2, 4, 6, 8, 10}));
  EXPECT_EQ(bounds(9, 64, {}, 16, kNoRowCap), (Bounds{0, 2, 4, 6, 8, 9}));
  // A row larger than a page is a page.
  EXPECT_EQ(bounds(3, 64, {}, 1, kNoRowCap), (Bounds{0, 1, 2, 3}));
  // The row cap bounds pages that bytes would not.
  EXPECT_EQ(bounds(12, 8, {}, 1024, 5), (Bounds{0, 5, 10, 12}));
  // Rows without columns are split by the row cap alone.
  EXPECT_EQ(bounds(7, 0, {}, 1024, 3), (Bounds{0, 3, 6, 7}));
  EXPECT_EQ(bounds(1, 0, {}, 1024, 3), (Bounds{0, 1}));
  EXPECT_EQ(bounds(0, 64, {}, 16, kNoRowCap), (Bounds{0}));
}

TEST_F(PrestoPageDeviceTest, pageBoundariesWithStrings) {
  using Bounds = std::vector<int64_t>;
  auto offsets32 = toDevice(offsets32_);
  auto offsets64 = toDevice(offsets64_);
  const StringOffsets strings32{offsets32.data(), false, kFirstIndex};
  const StringOffsets strings64{offsets64.data(), true, kFirstIndex};
  for (const auto& strings : {strings32, strings64}) {
    SCOPED_TRACE(
        testing::Message() << "int64 offsets = " << strings.offsetsInt64);
    // 8-byte pages and only the string bytes: the cumulative bytes are 3, 3,
    // 13, 14, 15, 55, 57, 59, so the pages end before rows 2 and 5, and the
    // 40-byte row 5 is a page by itself.
    EXPECT_EQ(
        bounds(kNumStrings, 0, {strings}, 8, kNoRowCap),
        (Bounds{0, 2, 5, 6, 8}));
    // The same with the row cap splitting the three-row page.
    EXPECT_EQ(
        bounds(kNumStrings, 0, {strings}, 8, 2), (Bounds{0, 2, 4, 5, 6, 8}));
    // With 32 bits per row besides: cumulative bits 56, 88, 200, 240, 280,
    // 632, 680, 728 against 64-bit pages.
    EXPECT_EQ(
        bounds(kNumStrings, 32, {strings}, 8, kNoRowCap),
        (Bounds{0, 1, 2, 4, 5, 6, 7, 8}));
    // One page when everything fits.
    EXPECT_EQ(
        bounds(kNumStrings, 32, {strings}, 1 << 20, kNoRowCap),
        (Bounds{0, kNumStrings}));
    EXPECT_EQ(bounds(1, 32, {strings}, 1, kNoRowCap), (Bounds{0, 1}));
  }
  // Two string columns; the second starts at index 0, so its rows are 5, 4,
  // 3, 0, 10, 1, 1, 40 bytes long and the rows cost 8, 4, 13, 1, 11, 41, 3,
  // 42 bytes in all.
  const StringOffsets second{offsets32.data(), false, 0};
  EXPECT_EQ(
      bounds(kNumStrings, 0, {strings32, second}, 8, kNoRowCap),
      (Bounds{0, 1, 2, 4, 5, 6, 7, 8}));
}

TEST_F(PrestoPageDeviceTest, nullBytes) {
  auto valid = [](int64_t bit) { return bit % 3 != 0 && bit != 77; };
  auto mask = toDevice(makeMask(200, valid));
  rmm::device_uvector<uint8_t> out(9 + 1 + 2, stream_, mr_);
  const std::vector<detail::NullBytesTask> tasks = {
      // Crosses two mask words and ends with a partial byte.
      {mask.data(), 37, 70, out.data()},
      {mask.data(), 0, 8, out.data() + 9},
      // The last bits of the mask.
      {mask.data(), 191, 9, out.data() + 10},
  };
  detail::fillPrestoNullBytes(tasks, stream_, mr_);
  const auto result = toHost(out);

  size_t position = 0;
  for (const auto& task : tasks) {
    SCOPED_TRACE(testing::Message() << "firstBit = " << task.firstBit);
    const size_t numBytes = (task.numRows + 7) / 8;
    std::vector<uint8_t> expected(numBytes, 0);
    for (int32_t row = 0; row < task.numRows; ++row) {
      if (!valid(task.firstBit + row)) {
        expected[row / 8] |= 0x80 >> (row % 8);
      }
    }
    const std::vector<uint8_t> actual(
        result.begin() + position, result.begin() + position + numBytes);
    EXPECT_EQ(actual, expected);
    position += numBytes;
  }
}

TEST_F(PrestoPageDeviceTest, stringOffsets) {
  auto offsets32 = toDevice(offsets32_);
  auto offsets64 = toDevice(offsets64_);
  const StringOffsets strings32{offsets32.data(), false, kFirstIndex};
  const StringOffsets strings64{offsets64.data(), true, kFirstIndex};
  for (const auto& strings : {strings32, strings64}) {
    SCOPED_TRACE(
        testing::Message() << "int64 offsets = " << strings.offsetsInt64);
    rmm::device_uvector<uint8_t> out(5 * sizeof(int32_t), stream_, mr_);
    const std::vector<detail::StringOffsetsTask> tasks = {
        // Rows 2 to 5 start at offset 12 and end at 22, 23, 24 and 64.
        {strings, 2, 4, out.data()},
        // Row 0 alone.
        {strings, 0, 1, out.data() + 4 * sizeof(int32_t)},
    };
    detail::fillPrestoStringOffsets(tasks, stream_, mr_);
    const auto bytes = toHost(out);
    std::vector<int32_t> actual(5);
    std::memcpy(actual.data(), bytes.data(), bytes.size());
    EXPECT_EQ(actual, (std::vector<int32_t>{10, 11, 12, 52, 3}));

    const std::vector<int64_t> rows = {0, 2, 5, kNumStrings};
    EXPECT_EQ(
        detail::gatherStringOffsets(strings, rows, stream_, mr_),
        (std::vector<int64_t>{9, 12, 24, 68}));
  }
}

TEST_F(PrestoPageDeviceTest, compactValid) {
  constexpr int32_t kNumRows = 50;
  constexpr int64_t kFirstBit = 33;
  auto valid = [](int64_t bit) { return bit % 2 == 0 || bit % 5 == 0; };
  auto mask = toDevice(makeMask(kFirstBit + kNumRows, valid));

  auto check = [&](auto zero) {
    using T = decltype(zero);
    SCOPED_TRACE(testing::Message() << "width = " << sizeof(T));
    std::vector<T> values;
    std::vector<T> expected;
    for (int32_t row = 0; row < kNumRows; ++row) {
      const T value = static_cast<T>(row * 2'654'435'761LL + 1);
      values.push_back(value);
      if (valid(kFirstBit + row)) {
        expected.push_back(value);
      }
    }
    auto src = toDevice(values);
    rmm::device_uvector<T> dst(kNumRows, stream_, mr_);
    const auto copied = detail::compactValid(
        reinterpret_cast<const uint8_t*>(src.data()),
        sizeof(T),
        mask.data(),
        kFirstBit,
        kNumRows,
        reinterpret_cast<uint8_t*>(dst.data()),
        stream_);
    ASSERT_EQ(copied, expected.size());
    auto actual = toHost(dst);
    actual.resize(copied);
    EXPECT_TRUE(actual == expected);
  };
  check(uint8_t{});
  check(uint16_t{});
  check(uint32_t{});
  check(uint64_t{});
  check(__int128_t{});
}

TEST_F(PrestoPageDeviceTest, timestampsToMillis) {
  auto convert = [&](const std::vector<int64_t>& values,
                     int64_t mul,
                     int64_t div,
                     std::vector<int64_t>* result) {
    auto src = toDevice(values);
    rmm::device_uvector<int64_t> dst(values.size(), stream_, mr_);
    const bool ok = detail::timestampsToMillis(
        src.data(), values.size(), mul, div, dst.data(), stream_, mr_);
    *result = toHost(dst);
    return ok;
  };
  std::vector<int64_t> result;

  // Seconds, up to the last values whose milliseconds fit in int64.
  constexpr int64_t kMaxSeconds = std::numeric_limits<int64_t>::max() / 1'000;
  constexpr int64_t kMinSeconds = std::numeric_limits<int64_t>::min() / 1'000;
  EXPECT_TRUE(convert({0, 1, -1, kMaxSeconds, kMinSeconds}, 1'000, 1, &result));
  EXPECT_EQ(
      result,
      (std::vector<int64_t>{
          0, 1'000, -1'000, kMaxSeconds * 1'000, kMinSeconds * 1'000}));
  EXPECT_FALSE(convert({0, kMinSeconds - 1}, 1'000, 1, &result));
  EXPECT_FALSE(convert({kMaxSeconds + 1}, 1'000, 1, &result));

  // Microseconds and nanoseconds round down.
  EXPECT_TRUE(
      convert({-1, -1'000, -1'001, 1'999, 0, 1'000}, 1, 1'000, &result));
  EXPECT_EQ(result, (std::vector<int64_t>{-1, -1, -2, 1, 0, 1}));
  EXPECT_TRUE(convert({-1, 1'999'999, -1'000'000}, 1, 1'000'000, &result));
  EXPECT_EQ(result, (std::vector<int64_t>{-1, 1, -1}));

  // Milliseconds pass through.
  EXPECT_TRUE(convert({5, -5}, 1, 1, &result));
  EXPECT_EQ(result, (std::vector<int64_t>{5, -5}));

  // In place.
  auto values = toDevice(std::vector<int64_t>{7, -7, 1'001});
  EXPECT_TRUE(
      detail::timestampsToMillis(
          values.data(), 3, 1'000, 1, values.data(), stream_, mr_));
  EXPECT_EQ(toHost(values), (std::vector<int64_t>{7'000, -7'000, 1'001'000}));
}

TEST_F(PrestoPageDeviceTest, decimalsToJava) {
  const auto kTopBit = static_cast<unsigned __int128>(1) << 127;
  const auto kBig = static_cast<__int128_t>(1) << 70;
  const std::vector<__int128_t> values = {0, 5, -1, -kBig, kBig};
  const std::vector<unsigned __int128> expected = {
      0, 5, 1 | kTopBit, static_cast<unsigned __int128>(kBig) | kTopBit, kBig};
  auto check = [&](const std::vector<__int128_t>& actual) {
    ASSERT_EQ(actual.size(), expected.size());
    for (size_t i = 0; i < expected.size(); ++i) {
      EXPECT_TRUE(static_cast<unsigned __int128>(actual[i]) == expected[i])
          << "value " << i;
    }
  };
  auto src = toDevice(values);
  rmm::device_uvector<__int128_t> dst(values.size(), stream_, mr_);
  detail::decimalsToJava(src.data(), values.size(), dst.data(), stream_);
  check(toHost(dst));
  // In place.
  detail::decimalsToJava(src.data(), values.size(), src.data(), stream_);
  check(toHost(src));
}

TEST_F(PrestoPageDeviceTest, batchedCopy) {
  std::vector<uint8_t> source(20);
  for (size_t i = 0; i < source.size(); ++i) {
    source[i] = i;
  }
  auto src = toDevice(source);
  auto dst = toDevice(std::vector<uint8_t>(20, 0xff));
  const std::vector<detail::CopyTask> tasks = {
      {src.data(), dst.data() + 5, 3},
      {src.data(), dst.data() + 15, 0},
      {src.data() + 10, dst.data(), 4},
  };
  detail::batchedDeviceCopy(tasks, stream_, mr_);
  std::vector<uint8_t> expected(20, 0xff);
  expected[0] = 10;
  expected[1] = 11;
  expected[2] = 12;
  expected[3] = 13;
  expected[5] = 0;
  expected[6] = 1;
  expected[7] = 2;
  EXPECT_EQ(toHost(dst), expected);
}

} // namespace
} // namespace facebook::velox::cudf_velox
