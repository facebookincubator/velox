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

#include "velox/experimental/cudf/exec/KeyNormalization.h"

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/type/tz/TimeZoneMap.h"

#include <cudf/column/column_factories.hpp>
#include <cudf/copying.hpp>
#include <cudf/utilities/bit.hpp>
#include <cudf/utilities/default_stream.hpp>
#include <cudf/utilities/memory_resource.hpp>

#include <rmm/device_buffer.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <optional>
#include <vector>

using namespace facebook::velox;
using namespace facebook::velox::cudf_velox;

namespace {

// The two zones used throughout: the extremes of the offset range, so their
// zone keys differ as widely as their offsets (Kiritimati 2137, Midway 2142).
constexpr const char* kKiritimati = "Pacific/Kiritimati";
constexpr const char* kMidway = "Pacific/Midway";

class KeyNormalizationTest : public testing::Test {
 protected:
  rmm::cuda_stream_view stream() const {
    return cudf::get_default_stream();
  }

  rmm::device_async_resource_ref mr() const {
    return cudf::get_current_device_resource_ref();
  }

  // Builds an INT64 column from host values, with an optional validity mask.
  std::unique_ptr<cudf::column> int64Column(
      const std::vector<std::optional<int64_t>>& values) {
    std::vector<int64_t> data;
    data.reserve(values.size());
    for (const auto& v : values) {
      data.push_back(v.value_or(0));
    }
    auto column = cudf::make_numeric_column(
        cudf::data_type{cudf::type_id::INT64},
        static_cast<cudf::size_type>(values.size()),
        cudf::mask_state::UNALLOCATED,
        stream(),
        mr());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        column->mutable_view().data<int64_t>(),
        data.data(),
        data.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice,
        stream().value()));
    stream().synchronize();

    const bool hasNulls = std::any_of(
        values.begin(), values.end(), [](const auto& v) { return !v; });
    if (!hasNulls) {
      return column;
    }
    // Built by hand rather than with cudf::detail::valid_if, which lives in a
    // .cuh and so is unavailable to a .cpp translation unit.
    const auto maskBytes = cudf::bitmask_allocation_size_bytes(values.size());
    std::vector<cudf::bitmask_type> hostMask(
        maskBytes / sizeof(cudf::bitmask_type), 0);
    cudf::size_type nullCount = 0;
    for (size_t i = 0; i < values.size(); ++i) {
      if (values[i].has_value()) {
        cudf::set_bit_unsafe(hostMask.data(), static_cast<cudf::size_type>(i));
      } else {
        ++nullCount;
      }
    }
    rmm::device_buffer maskBuffer(hostMask.data(), maskBytes, stream(), mr());
    column->set_null_mask(std::move(maskBuffer), nullCount);
    return column;
  }

  // Reads a column back to host, so a test can assert on exact bit patterns.
  std::vector<int64_t> toHost(const cudf::column_view& view) {
    std::vector<int64_t> out(view.size());
    CUDF_CUDA_TRY(cudaMemcpyAsync(
        out.data(),
        view.data<int64_t>(),
        out.size() * sizeof(int64_t),
        cudaMemcpyDeviceToHost,
        stream().value()));
    stream().synchronize();
    return out;
  }

  // "UTC" resolves to zone key 0 (see TimeZoneMap.cpp:60), which is the
  // interesting boundary: a UTC-keyed value has no low bits to clear, so it
  // must come out of normalization unchanged.
  int64_t packed(int64_t millis, const char* zone) {
    return pack(millis, tz::getTimeZoneID(zone));
  }
};

// The property the whole fix rests on: two values for the same instant under
// different zone keys are DIFFERENT packed values and become IDENTICAL after
// normalization, which is what makes every downstream hash, equality and sort
// agree with Velox.
TEST_F(KeyNormalizationTest, sameInstantDifferentZonesBecomeIdentical) {
  const int64_t millis = 1'623'758'400'000;
  auto column = int64Column(
      {packed(millis, kKiritimati),
       packed(millis, kMidway),
       packed(millis, "UTC")});
  auto raw = toHost(column->view());
  // Precondition: the inputs really do differ, so the test cannot pass
  // trivially.
  EXPECT_NE(raw[0], raw[1]);
  EXPECT_NE(raw[1], raw[2]);

  auto keys = cudf::table_view{{column->view()}};
  auto normalized = normalizeKeyColumns(keys, {true}, stream(), mr());
  ASSERT_TRUE(normalized.normalizedAny());
  auto out = toHost(normalized.view.column(0));
  EXPECT_EQ(out[0], out[1]);
  EXPECT_EQ(out[1], out[2]);
  // And the surviving value is the instant, shifted back into place.
  EXPECT_EQ(out[0], millis << kMillisShift);
}

// Pre-epoch instants are the case a divide by 4096 gets wrong: it truncates
// toward zero and lands a millisecond late, where the AND floors. Asserted
// against unpackMillisUtc rather than a hand-computed constant, so the test
// tracks the definition rather than restating it.
TEST_F(KeyNormalizationTest, preEpochInstantsFloorRatherThanTruncate) {
  const int64_t millis = -14'182'940'000;
  auto column = int64Column(
      {packed(millis, kKiritimati),
       packed(millis, kMidway),
       packed(millis, "UTC"),
       // Maximum zone key: the largest possible low-bit contamination.
       (millis << kMillisShift) | kTimezoneMask});
  auto keys = cudf::table_view{{column->view()}};
  auto normalized = normalizeKeyColumns(keys, {true}, stream(), mr());
  auto out = toHost(normalized.view.column(0));
  for (const auto value : out) {
    EXPECT_EQ(unpackMillisUtc(value), millis)
        << "normalization must floor to the instant for a pre-epoch value";
    EXPECT_EQ(value, out[0]) << "every zone key must collapse to one value";
  }
}

// Normalization must not over-collapse: instants one millisecond apart stay
// distinct. This is what fails if the mask ever clears more than kMillisShift
// bits.
TEST_F(KeyNormalizationTest, adjacentMillisStayDistinct) {
  auto column = int64Column(
      {packed(1'623'758'400'000, kKiritimati),
       packed(1'623'758'400'001, "UTC"),
       packed(-14'182'940'001, kMidway),
       packed(-14'182'940'000, "UTC")});
  auto keys = cudf::table_view{{column->view()}};
  auto normalized = normalizeKeyColumns(keys, {true}, stream(), mr());
  auto out = toHost(normalized.view.column(0));
  EXPECT_NE(out[0], out[1]);
  EXPECT_NE(out[2], out[3]);
}

// Ordering is preserved, which is what makes the same helper usable for sort
// and TopN keys and not only for hashing.
TEST_F(KeyNormalizationTest, orderingIsPreserved) {
  auto column = int64Column(
      {packed(-14'182'940'000, kMidway),
       packed(0, kKiritimati),
       packed(1'623'758'400'000, "UTC")});
  auto keys = cudf::table_view{{column->view()}};
  auto normalized = normalizeKeyColumns(keys, {true}, stream(), mr());
  auto out = toHost(normalized.view.column(0));
  EXPECT_LT(out[0], out[1]);
  EXPECT_LT(out[1], out[2]);
}

// A null key must stay null rather than masking to a value: under
// cudf::null_equality::UNEQUAL a null key is distinct from every real one, and
// turning it into 0 would silently make it equal to the epoch in UTC.
TEST_F(KeyNormalizationTest, nullsArePreserved) {
  const int64_t millis = 1'623'758'400'000;
  auto column = int64Column(
      {packed(millis, kKiritimati), std::nullopt, packed(millis, kMidway)});
  ASSERT_EQ(column->null_count(), 1);

  auto keys = cudf::table_view{{column->view()}};
  auto normalized = normalizeKeyColumns(keys, {true}, stream(), mr());
  const auto view = normalized.view.column(0);
  EXPECT_EQ(view.null_count(), 1);
  EXPECT_TRUE(view.nullable());
  auto out = toHost(view);
  EXPECT_EQ(out[0], out[2]) << "the two non-null rows still collapse";
}

// Columns not flagged as TIMESTAMP WITH TIME ZONE must be passed through
// untouched, and by reference rather than copied.
TEST_F(KeyNormalizationTest, nonTswtzColumnsArePassedThroughUnchanged) {
  auto tswtz = int64Column({packed(1'623'758'400'000, kKiritimati)});
  auto plain = int64Column({1'623'758'400'123});
  auto keys = cudf::table_view{{tswtz->view(), plain->view()}};

  auto normalized = normalizeKeyColumns(keys, {true, false}, stream(), mr());
  ASSERT_EQ(normalized.view.num_columns(), 2);
  // Exactly one column was rewritten, so exactly one is owned.
  EXPECT_EQ(normalized.owned.size(), 1);
  EXPECT_EQ(
      normalized.view.column(1).data<int64_t>(), plain->view().data<int64_t>())
      << "a non-TSWTZ key column must be referenced, not copied";
  EXPECT_EQ(toHost(normalized.view.column(1))[0], 1'623'758'400'123);
}

// With no TSWTZ key at all the helper is a no-op and must not allocate: this is
// the common case for every operator, so it has to stay free.
TEST_F(KeyNormalizationTest, noTswtzKeyIsANoOp) {
  auto a = int64Column({1, 2, 3});
  auto b = int64Column({4, 5, 6});
  auto keys = cudf::table_view{{a->view(), b->view()}};

  auto normalized = normalizeKeyColumns(keys, {false, false}, stream(), mr());
  EXPECT_FALSE(normalized.normalizedAny());
  EXPECT_TRUE(normalized.owned.empty());
  EXPECT_EQ(
      normalized.view.column(0).data<int64_t>(), a->view().data<int64_t>());
  EXPECT_EQ(
      normalized.view.column(1).data<int64_t>(), b->view().data<int64_t>());
}

// The row-type overload must derive the same flags the explicit one takes, so
// an operator can pass its channels rather than precomputing.
TEST_F(KeyNormalizationTest, rowTypeOverloadDerivesTheFlags) {
  const int64_t millis = 1'623'758'400'000;
  auto tswtz =
      int64Column({packed(millis, kKiritimati), packed(millis, kMidway)});
  auto plain = int64Column({7, 8});
  auto keys = cudf::table_view{{tswtz->view(), plain->view()}};

  auto rowType =
      ROW({"a", "t", "b"}, {BIGINT(), TIMESTAMP_WITH_TIME_ZONE(), BIGINT()});
  // Key channels deliberately out of order and not starting at 0, so a fix that
  // assumed channel == column position would fail here.
  const std::vector<column_index_t> keyChannels{1, 2};

  EXPECT_TRUE(anyKeyNeedsNormalization(rowType, keyChannels));
  auto normalized =
      normalizeKeyColumns(keys, rowType, keyChannels, stream(), mr());
  ASSERT_EQ(normalized.owned.size(), 1);
  auto out = toHost(normalized.view.column(0));
  EXPECT_EQ(out[0], out[1]);
  EXPECT_EQ(toHost(normalized.view.column(1))[0], 7);
}

TEST_F(KeyNormalizationTest, anyKeyNeedsNormalizationIsFalseWithoutTswtz) {
  auto rowType = ROW({"a", "b"}, {BIGINT(), VARCHAR()});
  EXPECT_FALSE(anyKeyNeedsNormalization(rowType, {0, 1}));
}

TEST_F(KeyNormalizationTest, flagCountMustMatchColumnCount) {
  auto column = int64Column({1});
  auto keys = cudf::table_view{{column->view()}};
  VELOX_ASSERT_THROW(
      normalizeKeyColumns(keys, {true, false}, stream(), mr()),
      "One normalization flag is required per key column");
}

} // namespace
