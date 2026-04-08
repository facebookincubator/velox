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

#include "velox/dwio/common/StatisticsBuilder.h"

namespace facebook::velox::dwio::stats {

// Import column statistics types from dwio::common.
using common::BinaryColumnStatistics;
using common::BooleanColumnStatistics;
using common::ColumnStatistics;
using common::DoubleColumnStatistics;
using common::IntegerColumnStatistics;
using common::StringColumnStatistics;

namespace {

template <typename T>
void addWithOverflowCheck(std::optional<T>& to, T value, uint64_t count) {
  if (to.has_value()) {
    T result;
    auto overflow = __builtin_mul_overflow(value, count, &result);
    if (!overflow) {
      overflow = __builtin_add_overflow(to.value(), result, &to.value());
    }
    if (overflow) {
      to.reset();
    }
  }
}

template <typename T>
void mergeWithOverflowCheck(
    std::optional<T>& to,
    const std::optional<T>& from) {
  if (to.has_value()) {
    if (from.has_value()) {
      auto overflow =
          __builtin_add_overflow(to.value(), from.value(), &to.value());
      if (overflow) {
        to.reset();
      }
    } else {
      to.reset();
    }
  }
}

template <typename T>
void mergeCount(std::optional<T>& to, const std::optional<T>& from) {
  if (to.has_value()) {
    if (from.has_value()) {
      to.value() += from.value();
    } else {
      to.reset();
    }
  }
}

template <typename T>
void mergeMin(std::optional<T>& to, const std::optional<T>& from) {
  if (to.has_value()) {
    if (!from.has_value()) {
      to.reset();
    } else if (from.value() < to.value()) {
      to = from;
    }
  }
}

template <typename T>
void mergeMax(std::optional<T>& to, const std::optional<T>& from) {
  if (to.has_value()) {
    if (!from.has_value()) {
      to.reset();
    } else if (from.value() > to.value()) {
      to = from;
    }
  }
}

bool isValidLength(const std::optional<uint64_t>& length) {
  return length.has_value() &&
      length.value() <= std::numeric_limits<int64_t>::max();
}

bool shouldKeepString(
    const std::optional<std::string>& val,
    uint32_t lengthLimit) {
  return val.has_value() && val.value().size() <= lengthLimit;
}

} // namespace

std::unique_ptr<ColumnStatistics> StatisticsBuilder::build() const {
  auto result = std::make_unique<ColumnStatistics>(
      valueCount_, hasNull_, rawSize_, size_, estimateNumDistinct());

  // For the base builder, there are no typed stats to add.
  return result;
}

void StatisticsBuilder::incrementSize(uint64_t size) {
  if (LIKELY(size_.has_value())) {
    addWithOverflowCheck(size_, size, /*count=*/1);
  }
}

void StatisticsBuilder::merge(const ColumnStatistics& other, bool ignoreSize) {
  mergeCount(valueCount_, other.getNumberOfValues());

  if (!hasNull_.has_value() || !hasNull_.value()) {
    auto otherHasNull = other.hasNull();
    if (otherHasNull.has_value()) {
      if (otherHasNull.value()) {
        hasNull_ = true;
      }
    } else if (hasNull_.has_value()) {
      hasNull_.reset();
    }
  }
  mergeCount(rawSize_, other.getRawSize());
  if (!ignoreSize) {
    mergeCount(size_, other.getSize());
  }
  if (hll_) {
    auto* otherBuilder = dynamic_cast<const StatisticsBuilder*>(&other);
    VELOX_CHECK_NOT_NULL(otherBuilder);
    VELOX_CHECK_NOT_NULL(otherBuilder->hll_);
    hll_->mergeWith(*otherBuilder->hll_);
  }
}

std::unique_ptr<StatisticsBuilder> StatisticsBuilder::create(
    const Type& type,
    const StatisticsBuilderOptions& options) {
  switch (type.kind()) {
    case TypeKind::BOOLEAN:
      return std::make_unique<BooleanStatisticsBuilder>(options);
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
      return std::make_unique<IntegerStatisticsBuilder>(options);
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
      return std::make_unique<DoubleStatisticsBuilder>(options);
    case TypeKind::VARCHAR:
      return std::make_unique<StringStatisticsBuilder>(options);
    case TypeKind::VARBINARY:
      return std::make_unique<BinaryStatisticsBuilder>(options);
    default:
      return std::make_unique<StatisticsBuilder>(options);
  }
}

void StatisticsBuilder::createTree(
    std::vector<std::unique_ptr<StatisticsBuilder>>& statBuilders,
    const Type& type,
    const StatisticsBuilderOptions& options) {
  auto kind = type.kind();
  switch (kind) {
    case TypeKind::BOOLEAN:
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
    case TypeKind::BIGINT:
    case TypeKind::REAL:
    case TypeKind::DOUBLE:
    case TypeKind::VARCHAR:
    case TypeKind::VARBINARY:
    case TypeKind::TIMESTAMP:
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      break;

    case TypeKind::ARRAY: {
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      const auto& arrayType = dynamic_cast<const ArrayType&>(type);
      createTree(statBuilders, *arrayType.elementType(), options);
      break;
    }

    case TypeKind::MAP: {
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      const auto& mapType = dynamic_cast<const MapType&>(type);
      createTree(statBuilders, *mapType.keyType(), options);
      createTree(statBuilders, *mapType.valueType(), options);
      break;
    }

    case TypeKind::ROW: {
      statBuilders.push_back(StatisticsBuilder::create(type, options));
      const auto& rowType = dynamic_cast<const RowType&>(type);
      for (const auto& childType : rowType.children()) {
        createTree(statBuilders, *childType, options);
      }
      break;
    }
    default:
      VELOX_FAIL("Not supported type: {}", kind);
      break;
  }
}

void BooleanStatisticsBuilder::addValues(bool value, uint64_t count) {
  increaseValueCount(count);
  if (trueCount_.has_value() && value) {
    trueCount_.value() += count;
  }
}

void BooleanStatisticsBuilder::merge(
    const ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const BooleanColumnStatistics*>(&other);
  if (!stats) {
    if (!other.isAllNull() && trueCount_.has_value()) {
      trueCount_.reset();
    }
    return;
  }
  mergeCount(trueCount_, stats->getTrueCount());
}

std::unique_ptr<ColumnStatistics> BooleanStatisticsBuilder::build() const {
  auto trueCount = isAllNull() ? std::nullopt : trueCount_;
  auto result = std::make_unique<BooleanColumnStatistics>(
      static_cast<const ColumnStatistics&>(*this), trueCount);
  if (auto numDistinct = estimateNumDistinct()) {
    result->setNumDistinct(*numDistinct);
  }
  return result;
}

void IntegerStatisticsBuilder::addValues(int64_t value, uint64_t count) {
  increaseValueCount(count);
  if (min_.has_value() && value < min_.value()) {
    min_ = value;
  }
  if (max_.has_value() && value > max_.value()) {
    max_ = value;
  }
  addWithOverflowCheck(sum_, value, count);
  addHash(value);
}

void IntegerStatisticsBuilder::merge(
    const ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const IntegerColumnStatistics*>(&other);
  if (!stats) {
    if (!other.isAllNull()) {
      min_.reset();
      max_.reset();
      sum_.reset();
    }
    return;
  }
  mergeMin(min_, stats->getMinimum());
  mergeMax(max_, stats->getMaximum());
  mergeWithOverflowCheck(sum_, stats->getSum());
}

std::unique_ptr<ColumnStatistics> IntegerStatisticsBuilder::build() const {
  auto min = isAllNull() ? std::nullopt : min_;
  auto max = isAllNull() ? std::nullopt : max_;
  auto sum = isAllNull() ? std::nullopt : sum_;
  auto result = std::make_unique<IntegerColumnStatistics>(
      static_cast<const ColumnStatistics&>(*this), min, max, sum);
  if (auto numDistinct = estimateNumDistinct()) {
    result->setNumDistinct(*numDistinct);
  }
  return result;
}

void DoubleStatisticsBuilder::addValues(double value, uint64_t count) {
  increaseValueCount(count);
  if (std::isnan(value)) {
    clear();
    return;
  }

  if (min_.has_value() && value < min_.value()) {
    min_ = value;
  }
  if (max_.has_value() && value > max_.value()) {
    max_ = value;
  }
  addHash(value);
  if (sum_.has_value()) {
    for (uint64_t i = 0; i < count; ++i) {
      sum_.value() += value;
    }
    if (std::isnan(sum_.value())) {
      sum_.reset();
    }
  }
}

void DoubleStatisticsBuilder::merge(
    const ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const DoubleColumnStatistics*>(&other);
  if (!stats) {
    if (!other.isAllNull()) {
      clear();
    }
    return;
  }
  mergeMin(min_, stats->getMinimum());
  mergeMax(max_, stats->getMaximum());
  mergeCount(sum_, stats->getSum());
  if (sum_.has_value() && std::isnan(sum_.value())) {
    sum_.reset();
  }
}

std::unique_ptr<ColumnStatistics> DoubleStatisticsBuilder::build() const {
  auto min = isAllNull() ? std::nullopt : min_;
  auto max = isAllNull() ? std::nullopt : max_;
  auto sum = isAllNull() ? std::nullopt : sum_;
  auto result = std::make_unique<DoubleColumnStatistics>(
      static_cast<const ColumnStatistics&>(*this), min, max, sum);
  if (auto numDistinct = estimateNumDistinct()) {
    result->setNumDistinct(*numDistinct);
  }
  return result;
}

void StringStatisticsBuilder::addValues(
    std::string_view value,
    uint64_t count) {
  auto isSelfEmpty = isAllNull();
  increaseValueCount(count);
  if (isSelfEmpty) {
    min_ = value;
    max_ = value;
  } else {
    if (min_.has_value() && value < std::string_view{min_.value()}) {
      min_ = value;
    }
    if (max_.has_value() && value > std::string_view{max_.value()}) {
      max_ = value;
    }
  }
  addHash(value);

  addWithOverflowCheck<uint64_t>(length_, value.size(), count);
}

void StringStatisticsBuilder::merge(
    const ColumnStatistics& other,
    bool ignoreSize) {
  auto isSelfEmpty = isAllNull();
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const StringColumnStatistics*>(&other);
  if (!stats) {
    if (!other.isAllNull()) {
      min_.reset();
      max_.reset();
      length_.reset();
    }
    return;
  }

  if (other.isAllNull()) {
    return;
  }

  if (isSelfEmpty) {
    min_ = stats->getMinimum();
    max_ = stats->getMaximum();
  } else {
    mergeMin(min_, stats->getMinimum());
    mergeMax(max_, stats->getMaximum());
  }

  mergeWithOverflowCheck(length_, stats->getTotalLength());
}

std::unique_ptr<ColumnStatistics> StringStatisticsBuilder::build() const {
  std::optional<std::string> min;
  std::optional<std::string> max;
  std::optional<int64_t> length;
  if (!isAllNull()) {
    if (shouldKeepString(min_, lengthLimit_)) {
      min = min_;
    }
    if (shouldKeepString(max_, lengthLimit_)) {
      max = max_;
    }
    if (isValidLength(length_)) {
      length = length_.value();
    }
  }
  auto result = std::make_unique<StringColumnStatistics>(
      static_cast<const ColumnStatistics&>(*this), min, max, length);
  if (auto numDistinct = estimateNumDistinct()) {
    result->setNumDistinct(*numDistinct);
  }
  return result;
}

void BinaryStatisticsBuilder::addValues(uint64_t length, uint64_t count) {
  increaseValueCount(count);
  addWithOverflowCheck(length_, length, count);
}

void BinaryStatisticsBuilder::merge(
    const ColumnStatistics& other,
    bool ignoreSize) {
  StatisticsBuilder::merge(other, ignoreSize);
  auto stats = dynamic_cast<const BinaryColumnStatistics*>(&other);
  if (!stats) {
    if (!other.isAllNull() && length_.has_value()) {
      length_.reset();
    }
    return;
  }
  mergeWithOverflowCheck(length_, stats->getTotalLength());
}

std::unique_ptr<ColumnStatistics> BinaryStatisticsBuilder::build() const {
  auto length =
      (!isAllNull() && isValidLength(length_)) ? length_ : std::nullopt;
  auto result = std::make_unique<BinaryColumnStatistics>(
      static_cast<const ColumnStatistics&>(*this), length);
  if (auto numDistinct = estimateNumDistinct()) {
    result->setNumDistinct(*numDistinct);
  }
  return result;
}

} // namespace facebook::velox::dwio::stats
