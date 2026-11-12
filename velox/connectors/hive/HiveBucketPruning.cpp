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
#include "velox/connectors/hive/HiveBucketPruning.h"

#include <folly/container/F14Set.h>
#include <algorithm>
#include <limits>

#include "velox/connectors/hive/HivePartitionFunction.h"
#include "velox/functions/lib/HiveHash.h"

namespace facebook::velox::connector::hive {
namespace {

using functions::HiveHash;

// Returns the values in [minValue, maxValue] that 'filter' passes, or
// std::nullopt if they can't be listed or number more than 'maxValues'.
std::optional<std::vector<int64_t>> integerValues(
    const common::Filter& filter,
    int64_t minValue,
    int64_t maxValue,
    int64_t maxValues) {
  std::vector<int64_t> values;
  auto addInRange = [&](const std::vector<int64_t>& candidates) {
    for (auto value : candidates) {
      if (value >= minValue && value <= maxValue) {
        values.push_back(value);
      }
    }
  };
  switch (filter.kind()) {
    case common::FilterKind::kBigintRange: {
      const auto& range = static_cast<const common::BigintRange&>(filter);
      const auto lower = std::max(range.lower(), minValue);
      const auto upper = std::min(range.upper(), maxValue);
      if (lower > upper) {
        return values;
      }
      // Wraps to 0 for the whole int64_t range.
      const auto numValues =
          static_cast<uint64_t>(upper) - static_cast<uint64_t>(lower) + 1;
      if (numValues == 0 || numValues > static_cast<uint64_t>(maxValues)) {
        return std::nullopt;
      }
      values.reserve(numValues);
      for (uint64_t i = 0; i < numValues; ++i) {
        values.push_back(lower + static_cast<int64_t>(i));
      }
      return values;
    }
    case common::FilterKind::kBigintValuesUsingHashTable:
      addInRange(
          static_cast<const common::BigintValuesUsingHashTable&>(filter)
              .values());
      break;
    case common::FilterKind::kBigintValuesUsingBitmask:
      addInRange(
          static_cast<const common::BigintValuesUsingBitmask&>(filter)
              .values());
      break;
    default:
      return std::nullopt;
  }
  if (static_cast<int64_t>(values.size()) > maxValues) {
    return std::nullopt;
  }
  return values;
}

template <TypeKind kind>
std::optional<std::vector<uint32_t>> integerHashes(
    const common::Filter& filter,
    int64_t maxValues) {
  using T = typename TypeTraits<kind>::NativeType;
  const auto values = integerValues(
      filter,
      std::numeric_limits<T>::min(),
      std::numeric_limits<T>::max(),
      maxValues);
  if (!values.has_value()) {
    return std::nullopt;
  }
  std::vector<uint32_t> hashes;
  hashes.reserve(values->size());
  for (auto value : *values) {
    hashes.push_back(HiveHash::hashOne<kind>(static_cast<T>(value)));
  }
  return hashes;
}

std::optional<std::vector<uint32_t>> stringHashes(
    const common::Filter& filter,
    int64_t maxValues) {
  std::vector<uint32_t> hashes;
  switch (filter.kind()) {
    case common::FilterKind::kBytesValues: {
      const auto& values =
          static_cast<const common::BytesValues&>(filter).values();
      if (static_cast<int64_t>(values.size()) > maxValues) {
        return std::nullopt;
      }
      for (const auto& value : values) {
        hashes.push_back(
            HiveHash::hashOne<TypeKind::VARCHAR>(StringView(value)));
      }
      return hashes;
    }
    case common::FilterKind::kBytesRange: {
      const auto& range = static_cast<const common::BytesRange&>(filter);
      if (!range.isSingleValue()) {
        return std::nullopt;
      }
      hashes.push_back(
          HiveHash::hashOne<TypeKind::VARCHAR>(StringView(range.lower())));
      return hashes;
    }
    default:
      return std::nullopt;
  }
}

// Returns the Hive hashes of the values of 'type' that 'filter' passes, or
// std::nullopt if the values can't be listed or number more than 'maxValues'.
// 'filter' passes no NULL.
std::optional<std::vector<uint32_t>>
valueHashes(const common::Filter& filter, const Type& type, int64_t maxValues) {
  switch (type.kind()) {
    case TypeKind::TINYINT:
      return integerHashes<TypeKind::TINYINT>(filter, maxValues);
    case TypeKind::SMALLINT:
      return integerHashes<TypeKind::SMALLINT>(filter, maxValues);
    case TypeKind::INTEGER:
      return integerHashes<TypeKind::INTEGER>(filter, maxValues);
    case TypeKind::BIGINT:
      return integerHashes<TypeKind::BIGINT>(filter, maxValues);
    case TypeKind::VARCHAR:
      return stringHashes(filter, maxValues);
    case TypeKind::BOOLEAN:
      if (!filter.is(common::FilterKind::kBoolValue)) {
        return std::nullopt;
      }
      return std::vector<uint32_t>{
          HiveHash::hashOne<TypeKind::BOOLEAN>(filter.testBool(true))};
    default:
      return std::nullopt;
  }
}

} // namespace

// static
std::optional<std::vector<int32_t>> HiveBucketPruning::buckets(
    const std::vector<const common::Filter*>& filters,
    const std::vector<TypePtr>& types,
    int32_t numBuckets) {
  VELOX_CHECK_EQ(filters.size(), types.size());
  VELOX_CHECK_GT(filters.size(), 0);
  VELOX_CHECK_GT(numBuckets, 0);

  for (const auto* filter : filters) {
    if (filter != nullptr && filter->is(common::FilterKind::kAlwaysFalse)) {
      return std::vector<int32_t>{};
    }
  }

  // The row hashes of all combinations of the values of the bucket columns so
  // far. Folding a column into 0 gives that column's hash, so starting from
  // {0} handles the first column like the others.
  folly::F14FastSet<uint32_t> rowHashes{0};
  for (size_t i = 0; i < filters.size(); ++i) {
    if (filters[i] == nullptr || filters[i]->testNull()) {
      return std::nullopt;
    }
    const auto hashes = valueHashes(
        *filters[i],
        *types[i],
        kMaxCombinations / static_cast<int64_t>(rowHashes.size()));
    if (!hashes.has_value()) {
      return std::nullopt;
    }
    if (hashes->empty()) {
      return std::vector<int32_t>{};
    }
    folly::F14FastSet<uint32_t> nextRowHashes;
    for (auto hash : *hashes) {
      for (auto rowHash : rowHashes) {
        HiveHash::mergeHash(/*mix=*/true, hash, rowHash);
        nextRowHashes.insert(rowHash);
      }
    }
    rowHashes = std::move(nextRowHashes);
  }

  folly::F14FastSet<int32_t> buckets;
  for (auto hash : rowHashes) {
    buckets.insert(HivePartitionFunction::bucket(hash, numBuckets));
  }
  return std::vector<int32_t>(buckets.begin(), buckets.end());
}

} // namespace facebook::velox::connector::hive
