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

#include "velox/expression/ExprToSubfieldFilter.h"

#include "velox/expression/Expr.h"

using namespace facebook::velox;

namespace facebook::velox::exec {

namespace {

VectorPtr toConstant(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator) {
  auto exprSet = evaluator->compile(expr);
  if (!exprSet->exprs()[0]->isConstantExpr()) {
    return nullptr;
  }
  RowVector input(
      evaluator->pool(), ROW({}, {}), nullptr, 1, std::vector<VectorPtr>{});
  SelectivityVector rows(1);
  VectorPtr result;
  try {
    evaluator->evaluate(exprSet.get(), rows, input, result);
  } catch (const VeloxUserError&) {
    return nullptr;
  }
  return result;
}

template <typename T>
T singleValue(const VectorPtr& vector) {
  auto simpleVector = vector->as<SimpleVector<T>>();
  VELOX_CHECK_NOT_NULL(simpleVector);
  return simpleVector->valueAt(0);
}

bool isBigintRange(const std::unique_ptr<common::Filter>& filter) {
  return filter->is(common::FilterKind::kBigintRange);
}

bool isBigintMultiRange(const std::unique_ptr<common::Filter>& filter) {
  return filter->is(common::FilterKind::kBigintMultiRange);
}

bool isBytesValues(const std::unique_ptr<common::Filter>& filter) {
  return filter->is(common::FilterKind::kBytesValues);
}

bool isBytesRange(const std::unique_ptr<common::Filter>& filter) {
  return filter->is(common::FilterKind::kBytesRange);
}

bool isSingleValueBytesRange(const std::unique_ptr<common::Filter>& filter) {
  if (!isBytesRange(filter)) {
    return false;
  }
  const auto* bytesRange = filter->as<common::BytesRange>();
  return bytesRange->isSingleValue();
}

template <typename T, typename U>
std::unique_ptr<T> asUniquePtr(std::unique_ptr<U> ptr) {
  return std::unique_ptr<T>(static_cast<T*>(ptr.release()));
}

std::unique_ptr<common::BigintRange> asBigintRange(
    std::unique_ptr<common::Filter>& ptr) {
  return asUniquePtr<common::BigintRange>(std::move(ptr));
}

template <typename T>
std::unique_ptr<common::Filter> toInt64In(
    const VectorPtr& vector,
    vector_size_t start,
    vector_size_t size,
    bool negated) {
  auto ints = vector->as<SimpleVector<T>>();
  std::vector<int64_t> values;
  values.reserve(size);
  bool hasNull = false;
  if (!ints->mayHaveNulls()) {
    for (auto i = 0; i < size; i++) {
      values.push_back(ints->valueAt(start + i));
    }
  } else {
    for (auto i = 0; i < size; i++) {
      if (!ints->isNullAt(start + i)) {
        values.push_back(ints->valueAt(start + i));
      } else if (!hasNull) {
        hasNull = true;
      }
    }
  }

  if (negated) {
    if (hasNull) {
      // If values contain NULL, the input cannot pass the filter, e.g. '1
      // not in (2, NULL)' evaluates to NULL.
      return std::make_unique<common::AlwaysFalse>();
    }
    return notIn(values);
  }

  return in(values);
}

} // namespace

std::shared_ptr<ExprToSubfieldFilterParser>
    ExprToSubfieldFilterParser::parser_ =
        std::make_shared<PrestoExprToSubfieldFilterParser>();

// static
bool ExprToSubfieldFilterParser::toSubfield(
    const core::ITypedExpr* field,
    common::Subfield& subfield) {
  std::vector<std::unique_ptr<common::Subfield::PathElement>> path;
  for (auto* current = field;;) {
    if (auto* fieldAccess =
            dynamic_cast<const core::FieldAccessTypedExpr*>(current)) {
      path.push_back(
          std::make_unique<common::Subfield::NestedField>(fieldAccess->name()));
    } else if (
        auto* dereference =
            dynamic_cast<const core::DereferenceTypedExpr*>(current)) {
      const auto& name = dereference->name();
      // When the field name is empty string, it typically means that the
      // field name was not set in the parent type.
      if (name.empty()) {
        return false;
      }
      path.push_back(std::make_unique<common::Subfield::NestedField>(name));
    } else if (dynamic_cast<const core::InputTypedExpr*>(current) == nullptr) {
      return false;
    } else {
      break;
    }

    if (current->inputs().empty()) {
      break;
    }
    if (current->inputs().size() != 1) {
      return false;
    }
    current = current->inputs()[0].get();
    if (current == nullptr) {
      return false;
    }
  }
  std::reverse(path.begin(), path.end());
  subfield = common::Subfield(std::move(path));
  return true;
}

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeNotEqualFilter(
    const core::TypedExprPtr& valueExpr,
    core::ExpressionEvaluator* evaluator) {
  auto value = toConstant(valueExpr, evaluator);
  if (!value) {
    return nullptr;
  }
  if (value->isNullAt(0)) {
    return std::make_unique<common::AlwaysFalse>();
  }

  auto lessThanFilter = makeLessThanFilter(valueExpr, evaluator);
  if (!lessThanFilter) {
    return nullptr;
  }

  auto greaterThanFilter = makeGreaterThanFilter(valueExpr, evaluator);
  if (!greaterThanFilter) {
    return nullptr;
  }

  const auto typeKind = value->typeKind();

  if (typeKind == TypeKind::TINYINT || typeKind == TypeKind::SMALLINT ||
      typeKind == TypeKind::INTEGER || typeKind == TypeKind::BIGINT) {
    VELOX_CHECK(isBigintRange(lessThanFilter));
    VELOX_CHECK(isBigintRange(greaterThanFilter));

    std::vector<std::unique_ptr<common::BigintRange>> filters;
    filters.emplace_back(asBigintRange(lessThanFilter));
    filters.emplace_back(asBigintRange(greaterThanFilter));
    return std::make_unique<common::BigintMultiRange>(
        std::move(filters), false);
  }

  if (typeKind == TypeKind::HUGEINT) {
    VELOX_NYI();
  } else {
    std::vector<std::unique_ptr<common::Filter>> filters;
    filters.emplace_back(std::move(lessThanFilter));
    filters.emplace_back(std::move(greaterThanFilter));

    return std::make_unique<common::MultiRange>(std::move(filters), false);
  }
}

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeEqualFilter(
    const core::TypedExprPtr& valueExpr,
    core::ExpressionEvaluator* evaluator) {
  auto value = toConstant(valueExpr, evaluator);
  if (!value) {
    return nullptr;
  }
  if (value->isNullAt(0)) {
    return std::make_unique<common::AlwaysFalse>();
  }
  switch (value->typeKind()) {
    case TypeKind::BOOLEAN:
      return boolEqual(singleValue<bool>(value));
    case TypeKind::TINYINT:
      return equal(singleValue<int8_t>(value));
    case TypeKind::SMALLINT:
      return equal(singleValue<int16_t>(value));
    case TypeKind::INTEGER:
      return equal(singleValue<int32_t>(value));
    case TypeKind::BIGINT:
      return equal(singleValue<int64_t>(value));
    case TypeKind::HUGEINT:
      return equalHugeint(singleValue<int128_t>(value));
    case TypeKind::VARCHAR:
      return equal(std::string(singleValue<StringView>(value)));
    case TypeKind::TIMESTAMP:
      return equal(singleValue<Timestamp>(value));
    default:
      return nullptr;
  }
}

// static
std::unique_ptr<common::Filter>
ExprToSubfieldFilterParser::makeGreaterThanFilter(
    const core::TypedExprPtr& lowerExpr,
    core::ExpressionEvaluator* evaluator) {
  auto lower = toConstant(lowerExpr, evaluator);
  if (!lower) {
    return nullptr;
  }
  if (lower->isNullAt(0)) {
    return std::make_unique<common::AlwaysFalse>();
  }
  switch (lower->typeKind()) {
    case TypeKind::TINYINT:
      return greaterThan(singleValue<int8_t>(lower));
    case TypeKind::SMALLINT:
      return greaterThan(singleValue<int16_t>(lower));
    case TypeKind::INTEGER:
      return greaterThan(singleValue<int32_t>(lower));
    case TypeKind::BIGINT:
      return greaterThan(singleValue<int64_t>(lower));
    case TypeKind::HUGEINT:
      return greaterThanHugeint(singleValue<int128_t>(lower));
    case TypeKind::DOUBLE:
      return greaterThanDouble(singleValue<double>(lower));
    case TypeKind::REAL:
      return greaterThanFloat(singleValue<float>(lower));
    case TypeKind::VARCHAR:
      return greaterThan(std::string(singleValue<StringView>(lower)));
    case TypeKind::TIMESTAMP:
      return greaterThan(singleValue<Timestamp>(lower));
    default:
      return nullptr;
  }
}

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeLessThanFilter(
    const core::TypedExprPtr& upperExpr,
    core::ExpressionEvaluator* evaluator) {
  auto upper = toConstant(upperExpr, evaluator);
  if (!upper) {
    return nullptr;
  }
  if (upper->isNullAt(0)) {
    return std::make_unique<common::AlwaysFalse>();
  }
  switch (upper->typeKind()) {
    case TypeKind::TINYINT:
      return lessThan(singleValue<int8_t>(upper));
    case TypeKind::SMALLINT:
      return lessThan(singleValue<int16_t>(upper));
    case TypeKind::INTEGER:
      return lessThan(singleValue<int32_t>(upper));
    case TypeKind::BIGINT:
      return lessThan(singleValue<int64_t>(upper));
    case TypeKind::HUGEINT:
      return lessThanHugeint(singleValue<int128_t>(upper));
    case TypeKind::DOUBLE:
      return lessThanDouble(singleValue<double>(upper));
    case TypeKind::REAL:
      return lessThanFloat(singleValue<float>(upper));
    case TypeKind::VARCHAR:
      return lessThan(std::string(singleValue<StringView>(upper)));
    case TypeKind::TIMESTAMP:
      return lessThan(singleValue<Timestamp>(upper));
    default:
      return nullptr;
  }
}

// static
std::unique_ptr<common::Filter>
ExprToSubfieldFilterParser::makeLessThanOrEqualFilter(
    const core::TypedExprPtr& upperExpr,
    core::ExpressionEvaluator* evaluator) {
  auto upper = toConstant(upperExpr, evaluator);
  if (!upper) {
    return nullptr;
  }
  if (upper->isNullAt(0)) {
    return std::make_unique<common::AlwaysFalse>();
  }
  switch (upper->typeKind()) {
    case TypeKind::TINYINT:
      return lessThanOrEqual(singleValue<int8_t>(upper));
    case TypeKind::SMALLINT:
      return lessThanOrEqual(singleValue<int16_t>(upper));
    case TypeKind::INTEGER:
      return lessThanOrEqual(singleValue<int32_t>(upper));
    case TypeKind::BIGINT:
      return lessThanOrEqual(singleValue<int64_t>(upper));
    case TypeKind::HUGEINT:
      return lessThanOrEqualHugeint(singleValue<int128_t>(upper));
    case TypeKind::DOUBLE:
      return lessThanOrEqualDouble(singleValue<double>(upper));
    case TypeKind::REAL:
      return lessThanOrEqualFloat(singleValue<float>(upper));
    case TypeKind::VARCHAR:
      return lessThanOrEqual(std::string(singleValue<StringView>(upper)));
    case TypeKind::TIMESTAMP:
      return lessThanOrEqual(singleValue<Timestamp>(upper));
    default:
      return nullptr;
  }
}

// static
std::unique_ptr<common::Filter>
ExprToSubfieldFilterParser::makeGreaterThanOrEqualFilter(
    const core::TypedExprPtr& lowerExpr,
    core::ExpressionEvaluator* evaluator) {
  auto lower = toConstant(lowerExpr, evaluator);
  if (!lower) {
    return nullptr;
  }
  if (lower->isNullAt(0)) {
    return std::make_unique<common::AlwaysFalse>();
  }
  switch (lower->typeKind()) {
    case TypeKind::TINYINT:
      return greaterThanOrEqual(singleValue<int8_t>(lower));
    case TypeKind::SMALLINT:
      return greaterThanOrEqual(singleValue<int16_t>(lower));
    case TypeKind::INTEGER:
      return greaterThanOrEqual(singleValue<int32_t>(lower));
    case TypeKind::BIGINT:
      return greaterThanOrEqual(singleValue<int64_t>(lower));
    case TypeKind::HUGEINT:
      return greaterThanOrEqualHugeint(singleValue<int128_t>(lower));
    case TypeKind::DOUBLE:
      return greaterThanOrEqualDouble(singleValue<double>(lower));
    case TypeKind::REAL:
      return greaterThanOrEqualFloat(singleValue<float>(lower));
    case TypeKind::VARCHAR:
      return greaterThanOrEqual(std::string(singleValue<StringView>(lower)));
    case TypeKind::TIMESTAMP:
      return greaterThanOrEqual(singleValue<Timestamp>(lower));
    default:
      return nullptr;
  }
}

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeInFilter(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    bool negated) {
  auto vector = toConstant(expr, evaluator);
  if (!(vector && vector->type()->isArray())) {
    return nullptr;
  }

  auto arrayVector = vector->valueVector()->as<ArrayVector>();
  auto index = vector->as<ConstantVector<ComplexType>>()->index();
  auto offset = arrayVector->offsetAt(index);
  auto size = arrayVector->sizeAt(index);
  auto elements = arrayVector->elements();

  auto elementType = arrayVector->type()->asArray().elementType();
  switch (elementType->kind()) {
    case TypeKind::TINYINT:
      return toInt64In<int8_t>(elements, offset, size, negated);
    case TypeKind::SMALLINT:
      return toInt64In<int16_t>(elements, offset, size, negated);
    case TypeKind::INTEGER:
      return toInt64In<int32_t>(elements, offset, size, negated);
    case TypeKind::BIGINT:
      return toInt64In<int64_t>(elements, offset, size, negated);
    case TypeKind::VARCHAR: {
      auto stringElements = elements->as<SimpleVector<StringView>>();
      std::vector<std::string> values;
      values.reserve(size);
      bool hasNull = false;
      if (!stringElements->mayHaveNulls()) {
        for (auto i = 0; i < size; i++) {
          values.push_back(std::string(stringElements->valueAt(offset + i)));
        }
      } else {
        for (auto i = 0; i < size; i++) {
          if (!stringElements->isNullAt(offset + i)) {
            values.push_back(std::string(stringElements->valueAt(offset + i)));
          } else if (!hasNull) {
            hasNull = true;
          }
        }
      }

      if (negated) {
        if (hasNull) {
          // If values contain NULL, the input cannot pass the filter, e.g. 'a'
          // not in ('b', NULL) evaluates to NULL.
          return std::make_unique<common::AlwaysFalse>();
        }
        return notIn(values);
      }
      return in(values);
    }
    default:
      return nullptr;
  }
}

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeBetweenFilter(
    const core::TypedExprPtr& lowerExpr,
    const core::TypedExprPtr& upperExpr,
    core::ExpressionEvaluator* evaluator,
    bool negated) {
  auto lower = toConstant(lowerExpr, evaluator);
  if (!lower) {
    return nullptr;
  }
  auto upper = toConstant(upperExpr, evaluator);
  if (!upper) {
    return nullptr;
  }
  switch (lower->typeKind()) {
    case TypeKind::INTEGER:
      if (negated) {
        return notBetween(
            singleValue<int32_t>(lower), singleValue<int32_t>(upper));
      }
      return between(singleValue<int32_t>(lower), singleValue<int32_t>(upper));
    case TypeKind::BIGINT:
      if (negated) {
        return notBetween(
            singleValue<int64_t>(lower), singleValue<int64_t>(upper));
      }
      return between(singleValue<int64_t>(lower), singleValue<int64_t>(upper));
    case TypeKind::DOUBLE:
      return negated
          ? nullptr
          : betweenDouble(
                singleValue<double>(lower), singleValue<double>(upper));
    case TypeKind::REAL:
      return negated
          ? nullptr
          : betweenFloat(singleValue<float>(lower), singleValue<float>(upper));
    case TypeKind::VARCHAR:
      if (negated) {
        return notBetween(
            std::string(singleValue<StringView>(lower)),
            std::string(singleValue<StringView>(upper)));
      }
      return between(
          std::string(singleValue<StringView>(lower)),
          std::string(singleValue<StringView>(upper)));
    case TypeKind::TIMESTAMP:
      return negated
          ? nullptr
          : between(
                singleValue<Timestamp>(lower), singleValue<Timestamp>(upper));
    default:
      return nullptr;
  }
}

namespace {

bool isNullAllowed(
    const std::vector<std::unique_ptr<common::Filter>>& disjuncts) {
  return std::any_of(
      disjuncts.begin(), disjuncts.end(), [](const auto& filter) {
        return filter->nullAllowed();
      });
}

// Combines overlapping ranges into one using OR semantic. Returns nullptr if
// ranges do not overlap. Ignores nullAllowed flag.
// @pre a.lower() <= b.lower()
std::unique_ptr<common::BigintRange> tryMergeOverlappingRanges(
    const common::BigintRange& a,
    const common::BigintRange& b,
    bool& alwaysTrue) {
  static constexpr auto kMax = std::numeric_limits<int64_t>::max();
  static constexpr auto kMin = std::numeric_limits<int64_t>::min();

  if (a.upper() == kMax || a.upper() + 1 >= b.lower()) {
    if (a.lower() == kMin && (a.upper() == kMax || b.upper() == kMax)) {
      alwaysTrue = true;
      return nullptr;
    }

    return std::make_unique<common::BigintRange>(
        a.lower(), std::max(a.upper(), b.upper()), /*nullAllowed=*/false);
  }
  return nullptr;
}

// Returns a single range that represents "a OR b" or nullptr if no such range
// exists.
// @pre a.lower() <= b.lower()
template <typename T>
std::unique_ptr<common::FloatingPointRange<T>> tryMergeOverlappingRanges(
    const common::FloatingPointRange<T>& a,
    const common::FloatingPointRange<T>& b,
    bool& alwaysTrue) {
  if (!a.upperUnbounded() && !b.lowerUnbounded() &&
      (a.upper() < b.lower() ||
       (a.upper() == b.lower() && a.upperExclusive() && b.lowerExclusive()))) {
    return nullptr;
  }

  const bool lowerUnbounded = a.lowerUnbounded() || b.lowerUnbounded();
  const bool upperUnbounded = a.upperUnbounded() || b.upperUnbounded();

  const T lower = lowerUnbounded ? std::numeric_limits<T>::lowest()
                                 : std::min(a.lower(), b.lower());

  bool lowerExclusive = lowerUnbounded;
  if (!lowerUnbounded) {
    if (a.lower() < b.lower()) {
      lowerExclusive = a.lowerExclusive();
    } else {
      lowerExclusive = a.lowerExclusive() && b.lowerExclusive();
    }
  }

  const T upper = upperUnbounded ? std::numeric_limits<T>::max()
                                 : std::max(a.upper(), b.upper());

  bool upperExclusive = upperUnbounded;
  if (!upperUnbounded) {
    if (a.upper() > b.upper()) {
      upperExclusive = a.upperExclusive();
    } else if (a.upper() < b.upper()) {
      upperExclusive = b.upperExclusive();
    } else {
      upperExclusive = a.upperExclusive() && b.upperExclusive();
    }
  }

  if (lowerUnbounded && upperUnbounded) {
    alwaysTrue = true;
    return nullptr;
  }

  return std::make_unique<common::FloatingPointRange<T>>(
      lower,
      lowerUnbounded,
      lowerExclusive,
      upper,
      upperUnbounded,
      upperExclusive,
      /*nullAllowed=*/false);
}

template <typename T, typename TToMultiRange>
std::unique_ptr<common::Filter> mergeOverlappingDisjuncts(
    std::vector<std::unique_ptr<T>>& ranges,
    bool nullAllowed,
    const TToMultiRange& toMultiRange) {
  std::vector<std::unique_ptr<T>> newRanges;
  newRanges.emplace_back(asUniquePtr<T>(ranges.front()->clone(nullAllowed)));

  for (auto i = 1; i < ranges.size(); i++) {
    bool alwaysTrue = false;
    if (auto merged = tryMergeOverlappingRanges(
            *newRanges.back(), *ranges[i], alwaysTrue)) {
      newRanges.back() = std::move(merged);
    } else {
      if (alwaysTrue) {
        if (nullAllowed) {
          return std::make_unique<common::AlwaysTrue>();
        }
        return isNotNull();
      }
      newRanges.emplace_back(std::move(ranges[i]));
    }
  }

  if (newRanges.size() == 1) {
    return std::move(newRanges.front());
  }

  return toMultiRange(newRanges, nullAllowed);
}

std::unique_ptr<common::Filter> tryMergeBigintRanges(
    std::vector<std::unique_ptr<common::Filter>>& disjuncts) {
  // Check if all filters are single-value equalities: a = 5. Convert these to
  // an IN list.
  if (std::all_of(disjuncts.begin(), disjuncts.end(), [](const auto& filter) {
        return isBigintRange(filter) &&
            filter->template as<common::BigintRange>()->isSingleValue();
      })) {
    std::vector<int64_t> values;
    values.reserve(disjuncts.size());

    for (auto& filter : disjuncts) {
      values.emplace_back(filter->as<common::BigintRange>()->lower());
    }

    return common::createBigintValues(values, isNullAllowed(disjuncts));
  }

  if (!std::all_of(disjuncts.begin(), disjuncts.end(), [](const auto& filter) {
        return isBigintRange(filter) || isBigintMultiRange(filter);
      })) {
    return nullptr;
  }

  const bool nullAllowed = isNullAllowed(disjuncts);

  std::vector<std::unique_ptr<common::BigintRange>> ranges;
  for (auto& filter : disjuncts) {
    if (isBigintRange(filter)) {
      ranges.emplace_back(asBigintRange(filter));
    } else {
      for (const auto& range :
           filter->as<common::BigintMultiRange>()->ranges()) {
        ranges.emplace_back(std::make_unique<common::BigintRange>(*range));
      }
    }
  }

  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    return a->lower() < b->lower();
  });

  return mergeOverlappingDisjuncts(
      ranges, nullAllowed, [](auto& newRanges, bool nullAllowed) {
        return std::make_unique<common::BigintMultiRange>(
            std::move(newRanges), nullAllowed);
      });
}

template <typename T>
std::unique_ptr<common::Filter> tryMergeFloatingPointRanges(
    std::vector<std::unique_ptr<common::Filter>>& disjuncts) {
  constexpr auto filterKind = std::is_same_v<T, double>
      ? common::FilterKind::kDoubleRange
      : common::FilterKind::kFloatRange;

  if (!std::all_of(disjuncts.begin(), disjuncts.end(), [](const auto& filter) {
        return filter->is(filterKind);
      })) {
    return nullptr;
  }

  const bool nullAllowed = isNullAllowed(disjuncts);

  std::vector<std::unique_ptr<common::FloatingPointRange<T>>> ranges;
  ranges.reserve(disjuncts.size());
  for (auto& filter : disjuncts) {
    ranges.emplace_back(
        asUniquePtr<common::FloatingPointRange<T>>(std::move(filter)));
  }

  std::sort(ranges.begin(), ranges.end(), [](const auto& a, const auto& b) {
    if (a->lowerUnbounded() && b->lowerUnbounded()) {
      return false;
    }

    if (a->lowerUnbounded()) {
      return true;
    }

    if (b->lowerUnbounded()) {
      return false;
    }

    return a->lower() < b->lower();
  });

  return mergeOverlappingDisjuncts(
      ranges, nullAllowed, [](auto& newRanges, bool nullAllowed) {
        std::vector<std::unique_ptr<common::Filter>> filters;
        filters.reserve(newRanges.size());
        for (auto& range : newRanges) {
          filters.emplace_back(std::move(range));
        }
        return std::make_unique<common::MultiRange>(
            std::move(filters), nullAllowed);
      });
}

std::unique_ptr<common::Filter> tryMergeBytesValues(
    std::vector<std::unique_ptr<common::Filter>>& disjuncts) {
  if (!std::all_of(disjuncts.begin(), disjuncts.end(), [](const auto& filter) {
        return isBytesValues(filter) || isSingleValueBytesRange(filter);
      })) {
    return nullptr;
  }

  const bool nullAllowed = isNullAllowed(disjuncts);

  std::vector<std::string> values;
  for (auto& filter : disjuncts) {
    if (isBytesValues(filter)) {
      const auto* bytesValues = filter->as<common::BytesValues>();
      for (const auto& value : bytesValues->values()) {
        values.push_back(value);
      }
    } else {
      const auto* bytesRange = filter->as<common::BytesRange>();
      values.push_back(bytesRange->lower());
    }
  }

  return std::make_unique<common::BytesValues>(values, nullAllowed);
}

} // namespace

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeOrFilter(
    std::vector<std::unique_ptr<common::Filter>> disjuncts) {
  VELOX_CHECK_GE(disjuncts.size(), 2);

  if (auto merged = tryMergeBigintRanges(disjuncts)) {
    return merged;
  }

  if (auto merged = tryMergeFloatingPointRanges<double>(disjuncts)) {
    return merged;
  }

  if (auto merged = tryMergeFloatingPointRanges<float>(disjuncts)) {
    return merged;
  }

  if (auto merged = tryMergeBytesValues(disjuncts)) {
    return merged;
  }

  return nullptr;
}

namespace {
std::optional<std::pair<common::Subfield, std::unique_ptr<common::Filter>>>
combine(common::Subfield& subfield, std::unique_ptr<common::Filter>& filter) {
  if (filter != nullptr) {
    return std::make_pair(std::move(subfield), std::move(filter));
  }

  return std::nullopt;
}
} // namespace

std::optional<std::pair<common::Subfield, std::unique_ptr<common::Filter>>>
PrestoExprToSubfieldFilterParser::leafCallToSubfieldFilter(
    const core::CallTypedExpr& call,
    core::ExpressionEvaluator* evaluator,
    bool negated) {
  if (call.inputs().empty()) {
    return std::nullopt;
  }

  const auto* leftSide = call.inputs()[0].get();

  common::Subfield subfield;
  if (call.name() == "eq") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = negated ? makeNotEqualFilter(call.inputs()[1], evaluator)
                            : makeEqualFilter(call.inputs()[1], evaluator);

      return combine(subfield, filter);
    }
  } else if (call.name() == "neq") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = negated ? makeEqualFilter(call.inputs()[1], evaluator)
                            : makeNotEqualFilter(call.inputs()[1], evaluator);
      return combine(subfield, filter);
    }
  } else if (call.name() == "lte") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = negated
          ? makeGreaterThanFilter(call.inputs()[1], evaluator)
          : makeLessThanOrEqualFilter(call.inputs()[1], evaluator);
      return combine(subfield, filter);
    }
  } else if (call.name() == "lt") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = negated
          ? makeGreaterThanOrEqualFilter(call.inputs()[1], evaluator)
          : makeLessThanFilter(call.inputs()[1], evaluator);
      return combine(subfield, filter);
    }
  } else if (call.name() == "gte") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = negated
          ? makeLessThanFilter(call.inputs()[1], evaluator)
          : makeGreaterThanOrEqualFilter(call.inputs()[1], evaluator);
      return combine(subfield, filter);
    }
  } else if (call.name() == "gt") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = negated
          ? makeLessThanOrEqualFilter(call.inputs()[1], evaluator)
          : makeGreaterThanFilter(call.inputs()[1], evaluator);
      return combine(subfield, filter);
    }
  } else if (call.name() == "between") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = makeBetweenFilter(
          call.inputs()[1], call.inputs()[2], evaluator, negated);
      return combine(subfield, filter);
    }
  } else if (call.name() == "in") {
    if (toSubfield(leftSide, subfield)) {
      auto filter = makeInFilter(call.inputs()[1], evaluator, negated);
      return combine(subfield, filter);
    }
  } else if (call.name() == "is_null") {
    if (toSubfield(leftSide, subfield)) {
      if (negated) {
        return std::make_pair(std::move(subfield), isNotNull());
      }
      return std::make_pair(std::move(subfield), isNull());
    }
  }
  return std::nullopt;
}

} // namespace facebook::velox::exec
