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
#include "velox/connectors/hive/HiveIndexSourceUtils.h"

#include <cmath>
#include <limits>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive {

std::optional<variant> extractPointLookupValue(const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  if (filter->nullAllowed()) {
    return std::nullopt;
  }

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      const auto* range = filter->as<common::BigintRange>();
      if (range->isSingleValue()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      if (!range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      if (!range->lowerUnbounded() && !range->upperUnbounded() &&
          range->lower() == range->upper() && !range->lowerExclusive() &&
          !range->upperExclusive()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      if (range->isSingleValue()) {
        return variant(range->lower());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBytesValues: {
      const auto* values = filter->as<common::BytesValues>();
      if (values->values().size() == 1) {
        return variant(*values->values().begin());
      }
      return std::nullopt;
    }
    case common::FilterKind::kBoolValue: {
      const auto* boolFilter = filter->as<common::BoolValue>();
      return variant(boolFilter->testBool(true));
    }
    default:
      return std::nullopt;
  }
}

std::optional<std::pair<variant, variant>> extractRangeBounds(
    const common::Filter* filter) {
  VELOX_CHECK_NOT_NULL(filter);

  if (filter->nullAllowed()) {
    return std::nullopt;
  }

  switch (filter->kind()) {
    case common::FilterKind::kBigintRange: {
      // BigintRange is always inclusive: exclusive comparisons are normalized
      // into the bound value at filter construction (e.g. 'x > 1' becomes
      // lower=2), so no exclusive-to-inclusive adjustment is needed here.
      const auto* range = filter->as<common::BigintRange>();
      return std::make_pair(variant(range->lower()), variant(range->upper()));
    }
    case common::FilterKind::kDoubleRange: {
      const auto* range = filter->as<common::DoubleRange>();
      if (range->lowerUnbounded() || range->upperUnbounded()) {
        return std::nullopt;
      }
      double lower = range->lower();
      double upper = range->upper();
      if (range->lowerExclusive()) {
        lower = std::nextafter(lower, std::numeric_limits<double>::infinity());
      }
      if (range->upperExclusive()) {
        upper = std::nextafter(upper, -std::numeric_limits<double>::infinity());
      }
      return std::make_pair(variant(lower), variant(upper));
    }
    case common::FilterKind::kFloatRange: {
      const auto* range = filter->as<common::FloatRange>();
      if (range->lowerUnbounded() || range->upperUnbounded()) {
        return std::nullopt;
      }
      double lower = range->lower();
      double upper = range->upper();
      if (range->lowerExclusive()) {
        lower = std::nextafter(lower, std::numeric_limits<double>::infinity());
      }
      if (range->upperExclusive()) {
        upper = std::nextafter(upper, -std::numeric_limits<double>::infinity());
      }
      return std::make_pair(variant(lower), variant(upper));
    }
    case common::FilterKind::kBytesRange: {
      const auto* range = filter->as<common::BytesRange>();
      if (!range->isSingleValue()) {
        return std::make_pair(variant(range->lower()), variant(range->upper()));
      }
      return std::nullopt;
    }
    default:
      return std::nullopt;
  }
}

core::IndexLookupConditionPtr createEqualConditionWithConstant(
    const std::string& columnName,
    const TypePtr& type,
    const variant& value) {
  auto keyExpr = std::make_shared<core::FieldAccessTypedExpr>(type, columnName);
  auto constantExpr = std::make_shared<core::ConstantTypedExpr>(type, value);
  return std::make_shared<core::EqualIndexLookupCondition>(
      std::move(keyExpr), std::move(constantExpr));
}

core::IndexLookupConditionPtr createBetweenConditionWithConstants(
    const std::string& columnName,
    const TypePtr& type,
    const variant& lowerValue,
    const variant& upperValue) {
  auto keyExpr = std::make_shared<core::FieldAccessTypedExpr>(type, columnName);
  auto lowerExpr = std::make_shared<core::ConstantTypedExpr>(type, lowerValue);
  auto upperExpr = std::make_shared<core::ConstantTypedExpr>(type, upperValue);
  return std::make_shared<core::BetweenIndexLookupCondition>(
      std::move(keyExpr), std::move(lowerExpr), std::move(upperExpr));
}

} // namespace facebook::velox::connector::hive
