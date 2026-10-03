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

#pragma once

#include "velox/dwio/parquet/reader/IntegerColumnReader.h"
#include "velox/dwio/parquet/reader/ParquetColumnReader.h"
#include "velox/type/Filter.h"

namespace facebook::velox::parquet {
namespace {

// Converts the big-endian 128-bit integer decoded from a Parquet UUID column
// into the UUID value Presto holds in memory.
//
// Presto stores a UUID as two longs in little-endian order (see
// UuidType.java), but its Parquet writer emits each of those longs
// big-endian (UuidValuesWriter.java), so the 16 bytes on disk are
// byte-reversed within each 64-bit half compared to the big-endian order the
// Parquet spec prescribes. Presto's own readers undo this the same way, so
// Presto round-trips its files; matching it here is what makes a native scan
// agree with a Java scan of the same file. Reading the column as VARBINARY
// still yields the raw file bytes.
FOLLY_ALWAYS_INLINE int128_t toPrestoUuid(const int128_t& value) {
  const auto unsignedValue = static_cast<uint128_t>(value);
  const auto high =
      __builtin_bswap64(static_cast<uint64_t>(unsignedValue >> 64));
  const auto low = __builtin_bswap64(static_cast<uint64_t>(unsignedValue));
  return static_cast<int128_t>((static_cast<uint128_t>(high) << 64) | low);
}

// Wraps a filter built over Presto UUID values so that it can be applied to
// the raw values decoded from the file, which are in the file's byte order.
// The conversion is not order preserving, so the filter cannot be rewritten
// into the file's value domain instead.
class PrestoUuidFilter final : public common::Filter {
 public:
  explicit PrestoUuidFilter(const common::Filter* filter)
      : Filter(
            filter->isDeterministic(),
            filter->nullAllowed(),
            filter->kind()),
        filter_{filter} {}

  bool testInt128(const int128_t& value) const final {
    return filter_->testInt128(toPrestoUuid(value));
  }

  // The file and Presto value domains are not ordered the same way, so a
  // range of file values says nothing about the Presto values it covers.
  bool testInt128Range(
      const int128_t& /*min*/,
      const int128_t& /*max*/,
      bool /*hasNull*/) const final {
    return true;
  }

  bool testNonNull() const final {
    return filter_->testNonNull();
  }

  std::unique_ptr<Filter> clone(
      std::optional<bool> /*nullAllowed*/ = std::nullopt) const final {
    VELOX_UNREACHABLE("PrestoUuidFilter only lives for one read call");
  }

  bool testingEquals(const Filter& /*other*/) const final {
    VELOX_UNREACHABLE("PrestoUuidFilter only lives for one read call");
  }

  folly::dynamic serialize() const final {
    VELOX_UNREACHABLE("PrestoUuidFilter only lives for one read call");
  }

  std::string toString() const final {
    return fmt::format("PrestoUuid({})", filter_->toString());
  }

 private:
  const common::Filter* const filter_;
};

} // namespace

/// Reads a Parquet UUID column (16 byte FIXED_LEN_BYTE_ARRAY annotated with
/// the UUID logical type) into hugeint values carrying Presto's UUID byte
/// order. See toPrestoUuid() for how that differs from the Parquet spec.
class UuidColumnReader : public IntegerColumnReader {
 public:
  UuidColumnReader(
      const TypePtr& requestedType,
      std::shared_ptr<const dwio::common::TypeWithId> fileType,
      ParquetParams& params,
      common::ScanSpec& scanSpec)
      : IntegerColumnReader(requestedType, fileType, params, scanSpec) {}

  void getValues(const RowSet& rows, VectorPtr* result) override {
    getIntValues(rows, requestedType_, result);
    if (allNull_) {
      return;
    }

    auto* values = (*result)->asUnchecked<FlatVector<int128_t>>();
    auto* rawValues = values->mutableRawValues();
    for (auto i = 0; i < numValues_; ++i) {
      if (!values->isNullAt(i)) {
        rawValues[i] = toPrestoUuid(rawValues[i]);
      }
    }
  }

  template <
      typename Reader,
      typename TFilter,
      bool isDense,
      typename ExtractValues>
  void readHelper(
      const velox::common::Filter* filter,
      const RowSet& rows,
      ExtractValues extractValues) {
    if constexpr (
        std::is_same_v<TFilter, velox::common::AlwaysTrue> ||
        std::is_same_v<TFilter, velox::common::IsNull> ||
        std::is_same_v<TFilter, velox::common::IsNotNull>) {
      // These do not look at the value, so they need no conversion.
      this->readWithVisitor(
          rows,
          dwio::common::
              ColumnVisitor<int128_t, TFilter, ExtractValues, isDense>(
                  *static_cast<const TFilter*>(filter),
                  this,
                  rows,
                  extractValues));
    } else {
      const PrestoUuidFilter uuidFilter{filter};
      this->readWithVisitor(
          rows,
          dwio::common::ColumnVisitor<
              int128_t,
              velox::common::Filter,
              ExtractValues,
              isDense>(uuidFilter, this, rows, extractValues));
    }
  }

  void read(
      int64_t offset,
      const RowSet& rows,
      const uint64_t* /*incomingNulls*/) override {
    prepareRead<int128_t>(offset, rows, nullptr);
    readCommon<UuidColumnReader, true>(rows);
    readOffset_ += rows.back() + 1;
  }
};

} // namespace facebook::velox::parquet
