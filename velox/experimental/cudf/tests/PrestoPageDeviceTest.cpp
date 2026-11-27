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
#include "velox/experimental/cudf/serializers/PrestoPageDevice.h"

#include "velox/common/base/VeloxException.h"
#include "velox/common/base/tests/GTestUtils.h"

#include <cudf/detail/utilities/buffer_factories.hpp>
#include <cudf/detail/utilities/cuda.hpp>
#include <cudf/detail/utilities/integer_utils.hpp>
#include <cudf/detail/utilities/vector_factories.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/error.hpp>
#include <cudf/utilities/memory_resource.hpp>
#include <cudf/utilities/span.hpp>

#include <cuda/buffer>
#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <vector>

namespace facebook::velox::cudf_velox {
namespace {

using detail::StringOffsets;

constexpr int64_t kNoRowCap = std::numeric_limits<int64_t>::max();
constexpr cudf::data_type kInt32{cudf::type_id::INT32};
constexpr cudf::data_type kInt64{cudf::type_id::INT64};

// Checks the device primitives on hand-built, sliced inputs: offsets children
// that start before the column's first row and masks whose first bit is not
// word aligned, which operator-level tests do not reach.
class PrestoPageDeviceTest : public testing::Test {
 protected:
  template <typename T>
  cuda::device_buffer<T> toDevice(const std::vector<T>& values) {
    auto result = cudf::detail::make_device_buffer_async(
        cudf::host_span<const T>{values},
        stream_,
        cudf::memory_resources{mr_, mr_});
    cudf::detail::sync_stream(stream_);
    return result;
  }

  template <typename T>
  std::vector<T> toHost(const cuda::device_buffer<T>& values) {
    return cudf::detail::make_std_vector(
        cudf::device_span<const T>{values.data(), values.size()}, stream_);
  }

  // Validity mask over 'numBits' bits.
  static std::vector<cudf::bitmask_type> makeMask(
      int64_t numBits,
      const std::function<bool(int64_t)>& valid) {
    std::vector<cudf::bitmask_type> words(
        cudf::detail::div_rounding_up_safe<int64_t>(numBits, 32), 0);
    for (int64_t i = 0; i < numBits; ++i) {
      if (valid(i)) {
        cudf::set_bit_unsafe(words.data(), static_cast<cudf::size_type>(i));
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
        numRows, fixedBitsPerRow, strings, pageBytes, maxRows, stream_);
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
  // Pages need a positive size and row cap, and rows a non-negative cost.
  VELOX_ASSERT_THROW(bounds(3, 64, {}, 0, kNoRowCap), "pageBytes");
  VELOX_ASSERT_THROW(
      bounds(3, 64, {}, std::numeric_limits<int64_t>::max(), kNoRowCap),
      "pageBytes");
  VELOX_ASSERT_THROW(bounds(3, 64, {}, 16, 0), "maxRows");
  VELOX_ASSERT_THROW(bounds(3, -1, {}, 16, kNoRowCap), "fixedBitsPerRow");
}

TEST_F(PrestoPageDeviceTest, pageBoundariesWithStrings) {
  using Bounds = std::vector<int64_t>;
  auto offsets32 = toDevice(offsets32_);
  auto offsets64 = toDevice(offsets64_);
  const StringOffsets strings32{offsets32.data(), kInt32, kFirstIndex};
  const StringOffsets strings64{offsets64.data(), kInt64, kFirstIndex};
  for (const auto& strings : {strings32, strings64}) {
    SCOPED_TRACE(
        testing::Message() << "int64 offsets = "
                           << (strings.offsetsType == kInt64));
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
  const StringOffsets second{offsets32.data(), kInt32, 0};
  EXPECT_EQ(
      bounds(kNumStrings, 0, {strings32, second}, 8, kNoRowCap),
      (Bounds{0, 1, 2, 4, 5, 6, 7, 8}));
  // One-byte strings in one-byte pages cut before every row, filling the cut
  // capacity sized from the last cumulative bit; empty strings in rows that
  // cost nothing make one page.
  auto ones = toDevice(std::vector<int32_t>{0, 1, 2, 3, 4, 5});
  const StringOffsets oneByteEach{ones.data(), kInt32, 0};
  EXPECT_EQ(
      bounds(5, 0, {oneByteEach}, 1, kNoRowCap), (Bounds{0, 1, 2, 3, 4, 5}));
  auto empty = toDevice(std::vector<int32_t>{7, 7, 7, 7});
  const StringOffsets emptyStrings{empty.data(), kInt32, 0};
  EXPECT_EQ(bounds(3, 0, {emptyStrings}, 1, kNoRowCap), (Bounds{0, 3}));
}

TEST_F(PrestoPageDeviceTest, nullBytes) {
  auto valid = [](int64_t bit) { return bit % 3 != 0 && bit != 77; };
  auto mask = toDevice(makeMask(200, valid));
  cuda::device_buffer<uint8_t> out(stream_, mr_, 9 + 1 + 2, cuda::no_init);
  const std::vector<detail::NullBytesTask> tasks = {
      // Crosses two mask words and ends with a partial byte.
      {mask.data(), 37, 70, out.data()},
      {mask.data(), 0, 8, out.data() + 9},
      // The last bits of the mask.
      {mask.data(), 191, 9, out.data() + 10},
  };
  std::vector<uint8_t> expected;
  for (const auto& task : tasks) {
    const size_t numBytes = cudf::detail::div_rounding_up_safe(task.numRows, 8);
    std::vector<uint8_t> bytes(numBytes, 0);
    for (int32_t row = 0; row < task.numRows; ++row) {
      if (!valid(task.firstBit + row)) {
        bytes[row / 8] |= 0x80 >> (row % 8);
      }
    }
    expected.insert(expected.end(), bytes.begin(), bytes.end());
  }

  // The same tasks from pinned memory, which the function reads in stream
  // order instead of before it returns.
  auto pinnedTasks =
      cudf::detail::make_pinned_vector_async<detail::NullBytesTask>(
          tasks.size(), stream_);
  std::copy(tasks.begin(), tasks.end(), pinnedTasks.begin());
  ASSERT_TRUE(
      cudf::host_span<const detail::NullBytesTask>(pinnedTasks)
          .is_device_accessible());
  for (const bool pinned : {false, true}) {
    SCOPED_TRACE(testing::Message() << "pinned = " << pinned);
    CUDF_CUDA_TRY(cudaMemsetAsync(out.data(), 0xff, out.size(), stream_.get()));
    if (pinned) {
      detail::fillPrestoNullBytes(pinnedTasks, stream_);
    } else {
      detail::fillPrestoNullBytes(tasks, stream_);
    }
    EXPECT_EQ(toHost(out), expected);
  }
}

TEST_F(PrestoPageDeviceTest, stringOffsets) {
  auto offsets32 = toDevice(offsets32_);
  auto offsets64 = toDevice(offsets64_);
  const StringOffsets strings32{offsets32.data(), kInt32, kFirstIndex};
  const StringOffsets strings64{offsets64.data(), kInt64, kFirstIndex};
  for (const auto& strings : {strings32, strings64}) {
    SCOPED_TRACE(
        testing::Message() << "int64 offsets = "
                           << (strings.offsetsType == kInt64));
    cuda::device_buffer<uint8_t> out(
        stream_, mr_, 5 * sizeof(int32_t), cuda::no_init);
    const std::vector<detail::StringOffsetsTask> tasks = {
        // Rows 2 to 5 start at offset 12 and end at 22, 23, 24 and 64.
        {strings, 2, 4, out.data()},
        // Row 0 alone.
        {strings, 0, 1, out.data() + 4 * sizeof(int32_t)},
    };
    EXPECT_TRUE(detail::fillPrestoStringOffsets(tasks, stream_));
    const auto bytes = toHost(out);
    std::vector<int32_t> actual(5);
    std::memcpy(actual.data(), bytes.data(), bytes.size());
    EXPECT_EQ(actual, (std::vector<int32_t>{10, 11, 12, 52, 3}));

    const std::vector<int64_t> rows = {0, 2, 5, kNumStrings};
    EXPECT_EQ(
        detail::gatherStringOffsets(strings, rows, stream_),
        (std::vector<int64_t>{9, 12, 24, 68}));
  }
}

TEST_F(PrestoPageDeviceTest, stringOffsetsBeyondInt32) {
  // A row of 2^31 - 1 bytes, which only int64 offsets can describe, is the
  // most an int32 Presto offset holds; a range with one byte more is refused.
  constexpr int64_t kLimit = std::numeric_limits<int32_t>::max();
  auto offsets = toDevice(std::vector<int64_t>{0, 4, 4 + kLimit, 5 + kLimit});
  const StringOffsets strings{offsets.data(), kInt64, 0};
  cuda::device_buffer<uint8_t> out(
      stream_, mr_, 2 * sizeof(int32_t), cuda::no_init);
  EXPECT_TRUE(
      detail::fillPrestoStringOffsets(
          std::vector<detail::StringOffsetsTask>{{strings, 1, 1, out.data()}},
          stream_));
  EXPECT_FALSE(
      detail::fillPrestoStringOffsets(
          std::vector<detail::StringOffsetsTask>{{strings, 0, 2, out.data()}},
          stream_));
  EXPECT_FALSE(
      detail::fillPrestoStringOffsets(
          std::vector<detail::StringOffsetsTask>{{strings, 1, 2, out.data()}},
          stream_));
}

TEST_F(PrestoPageDeviceTest, alignmentAndArgumentChecks) {
  // Values follow the null bitmap in a Presto page, so a destination inside
  // a page is generally unaligned; the conversions work on aligned scratch
  // buffers and refuse anything else.
  // 64 bytes aligned to 16, which cuda::buffer only guarantees for a 16-byte
  // element type.
  cuda::device_buffer<__int128_t> buffer(stream_, mr_, 4, cuda::no_init);
  auto* const aligned = reinterpret_cast<uint8_t*>(buffer.data());
  auto* const i64 = reinterpret_cast<int64_t*>(aligned);
  auto* const i64Shifted = reinterpret_cast<int64_t*>(aligned + 4);
  constexpr auto kSeconds = cudf::type_id::TIMESTAMP_SECONDS;
  VELOX_ASSERT_THROW(
      detail::timestampsToMillis(i64, 2, kSeconds, i64Shifted, stream_),
      "'dst' must be 8-byte aligned");
  VELOX_ASSERT_THROW(
      detail::timestampsToMillis(i64Shifted, 2, kSeconds, i64, stream_),
      "'src' must be 8-byte aligned");
  // Only the four 64-bit timestamp units have a scale to milliseconds.
  VELOX_ASSERT_THROW(
      detail::timestampsToMillis(
          i64, 2, cudf::type_id::TIMESTAMP_DAYS, i64, stream_),
      "Not a cuDF timestamp type with 64-bit ticks");
  VELOX_ASSERT_THROW(
      detail::timestampsToMillis(i64, 2, cudf::type_id::INT64, i64, stream_),
      "Not a cuDF timestamp type with 64-bit ticks");

  auto* const i128 = reinterpret_cast<__int128_t*>(aligned);
  auto* const i128Shifted = reinterpret_cast<__int128_t*>(aligned + 8);
  VELOX_ASSERT_THROW(
      detail::decimalsToJava(i128, 2, i128Shifted, stream_),
      "'dst' must be 16-byte aligned");
  VELOX_ASSERT_THROW(
      detail::decimalsToJava(i128Shifted, 2, i128, stream_),
      "'src' must be 16-byte aligned");

  // The offsets of a strings column are INT32 or INT64.
  auto offsets = toDevice(std::vector<int32_t>{0, 1, 2});
  const StringOffsets unsignedOffsets{
      offsets.data(), cudf::data_type{cudf::type_id::UINT32}, 0};
  VELOX_ASSERT_THROW(
      bounds(2, 0, {unsignedOffsets}, 8, kNoRowCap), "INT32 or INT64");
  VELOX_ASSERT_THROW(
      detail::fillPrestoStringOffsets(
          std::vector<detail::StringOffsetsTask>{
              {unsignedOffsets, 0, 2, aligned}},
          stream_),
      "INT32 or INT64");
  VELOX_ASSERT_THROW(
      detail::gatherStringOffsets(
          unsignedOffsets, std::vector<int64_t>{0}, stream_),
      "INT32 or INT64");
}

TEST_F(PrestoPageDeviceTest, timestampsToMillis) {
  auto convert = [&](const std::vector<int64_t>& values,
                     cudf::type_id unit,
                     std::vector<int64_t>* result) {
    auto src = toDevice(values);
    cuda::device_buffer<int64_t> dst(
        stream_, mr_, values.size(), cuda::no_init);
    const bool ok = detail::timestampsToMillis(
        src.data(), values.size(), unit, dst.data(), stream_);
    *result = toHost(dst);
    return ok;
  };
  std::vector<int64_t> result;

  // Seconds, up to the last values whose milliseconds fit in int64.
  constexpr auto kSeconds = cudf::type_id::TIMESTAMP_SECONDS;
  constexpr int64_t kMaxSeconds = std::numeric_limits<int64_t>::max() / 1'000;
  constexpr int64_t kMinSeconds = std::numeric_limits<int64_t>::min() / 1'000;
  EXPECT_TRUE(convert({0, 1, -1, kMaxSeconds, kMinSeconds}, kSeconds, &result));
  EXPECT_EQ(
      result,
      (std::vector<int64_t>{
          0, 1'000, -1'000, kMaxSeconds * 1'000, kMinSeconds * 1'000}));
  EXPECT_FALSE(convert({0, kMinSeconds - 1}, kSeconds, &result));
  EXPECT_FALSE(convert({kMaxSeconds + 1}, kSeconds, &result));

  // Microseconds and nanoseconds round down.
  EXPECT_TRUE(convert(
      {-1, -1'000, -1'001, 1'999, 0, 1'000},
      cudf::type_id::TIMESTAMP_MICROSECONDS,
      &result));
  EXPECT_EQ(result, (std::vector<int64_t>{-1, -1, -2, 1, 0, 1}));
  EXPECT_TRUE(convert(
      {-1, 1'999'999, -1'000'000},
      cudf::type_id::TIMESTAMP_NANOSECONDS,
      &result));
  EXPECT_EQ(result, (std::vector<int64_t>{-1, 1, -1}));

  // Milliseconds pass through.
  EXPECT_TRUE(convert({5, -5}, cudf::type_id::TIMESTAMP_MILLISECONDS, &result));
  EXPECT_EQ(result, (std::vector<int64_t>{5, -5}));

  // In place.
  auto values = toDevice(std::vector<int64_t>{7, -7, 1'001});
  EXPECT_TRUE(
      detail::timestampsToMillis(
          values.data(), 3, kSeconds, values.data(), stream_));
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
  cuda::device_buffer<__int128_t> dst(
      stream_, mr_, values.size(), cuda::no_init);
  detail::decimalsToJava(src.data(), values.size(), dst.data(), stream_);
  check(toHost(dst));
  // In place.
  detail::decimalsToJava(src.data(), values.size(), src.data(), stream_);
  check(toHost(src));
}

} // namespace
} // namespace facebook::velox::cudf_velox
