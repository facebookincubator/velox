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

#include "velox/expression/ExprToSubfieldFilter.h"

#include "velox/expression/Expr.h"

using namespace facebook::velox;

namespace facebook::velox::exec {

namespace {

VectorPtr toConstant(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator) {
  auto exprSet = evaluator->compile(expr);
  VELOX_DCHECK_EQ(exprSet->exprs().size(), 1);
  if (!exprSet->exprs()[0]->isConstantExpr()) {
    return nullptr;
  }
  RowVector input{evaluator->pool(), ROW({}), {}, 1, {}};
  SelectivityVector rows{1};
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
  // TODO: Is throw possible here?
  // Should we return null filter in caller in such case instead of throw?
  const auto* simpleVector = vector->as<SimpleVector<T>>();
  VELOX_CHECK_NOT_NULL(simpleVector);
  return simpleVector->valueAt(0);
}

const core::CallTypedExpr* asCall(const core::ITypedExpr* expr) {
  return expr->isCallKind() ? expr->asUnchecked<core::CallTypedExpr>()
                            : nullptr;
}

common::MultiRange* asMultiRange(std::unique_ptr<common::Filter>& filter) {
  return filter->kind() == common::FilterKind::kMultiRange
      ? static_cast<common::MultiRange*>(filter.get())
      : nullptr;
}

common::BigintRange* asBigintRange(std::unique_ptr<common::Filter>& filter) {
  return filter->kind() == common::FilterKind::kBigintRange
      ? static_cast<common::BigintRange*>(filter.get())
      : nullptr;
}

common::BigintMultiRange* asBigintMultiRange(
    std::unique_ptr<common::Filter>& filter) {
  return filter->kind() == common::FilterKind::kBigintMultiRange
      ? static_cast<common::BigintMultiRange*>(filter.get())
      : nullptr;
}

common::NegatedBigintRange* asNegatedBigintRange(
    std::unique_ptr<common::Filter>& filter) {
  return filter->kind() == common::FilterKind::kNegatedBigintRange
      ? static_cast<common::NegatedBigintRange*>(filter.get())
      : nullptr;
}

template <typename T, typename U>
std::unique_ptr<T> asUniquePtr(std::unique_ptr<U> ptr) {
  return std::unique_ptr<T>{static_cast<T*>(ptr.release())};
}

std::unique_ptr<common::Filter> makeOrFilter(
    std::unique_ptr<common::Filter> a,
    std::unique_ptr<common::Filter> b) {
  VELOX_DCHECK_NOT_NULL(a);
  VELOX_DCHECK_NOT_NULL(b);

  if (a->kind() == common::FilterKind::kAlwaysFalse) {
    return b;
  }
  if (b->kind() == common::FilterKind::kAlwaysFalse) {
    return a;
  }

  if (a->kind() == common::FilterKind::kAlwaysTrue) {
    return a;
  }
  if (b->kind() == common::FilterKind::kAlwaysTrue) {
    return b;
  }

  const auto* aMulti = asMultiRange(a);
  const auto* bMulti = asMultiRange(b);
  if (aMulti || bMulti) {
    std::vector<std::unique_ptr<common::Filter>> filters;
    auto add = [&](const auto& f, const auto* fMulti) {
      if (fMulti) {
        for (const auto& filter : fMulti->filters()) {
          filters.emplace_back(filter->clone());
        }
      } else {
        filters.emplace_back(f->clone());
      }
    };
    add(a, aMulti);
    add(b, bMulti);
    return std::make_unique<common::MultiRange>(std::move(filters), false);
  }

  const auto* aBigintRange = asBigintRange(a);
  const auto* bBigintRange = asBigintRange(b);
  const auto* aBigintMultiRange = asBigintMultiRange(a);
  const auto* bBigintMultiRange = asBigintMultiRange(b);
  // TODO: Add negated bigint range handling when implementing range merging.
  // Before it's pointless because they would produce overlapping ranges.
  if ((aBigintRange || aBigintMultiRange) &&
      (bBigintRange || bBigintMultiRange)) {
    std::vector<std::unique_ptr<common::BigintRange>> newRanges;
    auto add = [&](const auto& r, const auto* rMulti) {
      if (rMulti) {
        for (const auto& range : rMulti->ranges()) {
          newRanges.emplace_back(
              asUniquePtr<common::BigintRange>(range->clone()));
        }
      } else {
        newRanges.emplace_back(asUniquePtr<common::BigintRange>(r->clone()));
      }
    };
    add(a, aBigintMultiRange);
    add(b, bBigintMultiRange);

    // TODO: merge overlapping ranges
    std::sort(
        newRanges.begin(), newRanges.end(), [](const auto& a, const auto& b) {
          return a->lower() < b->lower();
        });

    try {
      return std::make_unique<common::BigintMultiRange>(
          std::move(newRanges), false);
    } catch (...) {
      // Found overlapping ranges, fall back to MultiRange.
    }
  }

  return orFilter(std::move(a), std::move(b));
}

template <typename From, typename To>
std::vector<To>
toList(const VectorPtr& vector, vector_size_t start, vector_size_t size) {
  const auto* elements = vector->as<SimpleVector<From>>();
  std::vector<To> values;
  values.reserve(size);
  for (vector_size_t i = 0; i < size; ++i) {
    values.emplace_back(elements->valueAt(start + i));
  }
  return values;
}

PrestoExprToSubfieldFilterParser gDefaultParser;

} // namespace

std::shared_ptr<ExprToSubfieldFilterParser>
    ExprToSubfieldFilterParser::parser_ =
        std::shared_ptr<ExprToSubfieldFilterParser>{
            std::shared_ptr<ExprToSubfieldFilterParser>{},
            &gDefaultParser};

// static
bool ExprToSubfieldFilterParser::toSubfield(
    const core::ITypedExpr* field,
    common::Subfield& subfield) {
  std::vector<std::unique_ptr<common::Subfield::PathElement>> path;
  for (auto* current = field;;) {
    if (current->isFieldAccessKind()) {
      path.push_back(
          std::make_unique<common::Subfield::NestedField>(
              current->asUnchecked<core::FieldAccessTypedExpr>()->name()));
    } else if (current->isDereferenceKind()) {
      const auto& name =
          current->asUnchecked<core::DereferenceTypedExpr>()->name();
      // When the field name is empty string, it typically means that the
      // field name was not set in the parent type.
      if (name.empty()) {
        return false;
      }
      path.push_back(std::make_unique<common::Subfield::NestedField>(name));
    } else if (!current->isInputKind()) {
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
  const auto value = toConstant(valueExpr, evaluator);
  if (!value) {
    return nullptr;
  }
  switch (value->typeKind()) {
    case TypeKind::BOOLEAN:
      return boolEqual(!singleValue<bool>(value));
    case TypeKind::TINYINT:
      return notEqual(singleValue<int8_t>(value));
    case TypeKind::SMALLINT:
      return notEqual(singleValue<int16_t>(value));
    case TypeKind::INTEGER:
      return notEqual(singleValue<int32_t>(value));
    case TypeKind::BIGINT:
      return notEqual(singleValue<int64_t>(value));
    case TypeKind::VARCHAR:
      return notEqual(std::string(singleValue<StringView>(value)));
    default:
      break;
  }
  auto lessThanFilter = makeLessThanFilter(valueExpr, evaluator);
  if (!lessThanFilter) {
    return nullptr;
  }
  auto greaterThanFilter = makeGreaterThanFilter(valueExpr, evaluator);
  if (!greaterThanFilter) {
    return nullptr;
  }
  return makeOrFilter(std::move(lessThanFilter), std::move(greaterThanFilter));
}

std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeEqualFilter(
    const core::TypedExprPtr& valueExpr,
    core::ExpressionEvaluator* evaluator) {
  const auto value = toConstant(valueExpr, evaluator);
  if (!value) {
    return nullptr;
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
  const auto lower = toConstant(lowerExpr, evaluator);
  if (!lower) {
    return nullptr;
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
  const auto upper = toConstant(upperExpr, evaluator);
  if (!upper) {
    return nullptr;
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
  const auto upper = toConstant(upperExpr, evaluator);
  if (!upper) {
    return nullptr;
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
  const auto lower = toConstant(lowerExpr, evaluator);
  if (!lower) {
    return nullptr;
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
  const auto vector = toConstant(expr, evaluator);
  if (!vector || !vector->type()->isArray()) {
    return nullptr;
  }

  const auto* arrayVector = vector->valueVector()->as<ArrayVector>();
  const auto index = vector->as<ConstantVector<ComplexType>>()->index();
  const auto offset = arrayVector->offsetAt(index);
  const auto size = arrayVector->sizeAt(index);
  const auto& elements = arrayVector->elements();

  // TODO: Implement fallback case
  // - negated -- false: OR(equal(values[i]), ...)
  // - negated -- true: OR(notEqual(values[i]), ...)
  switch (elements->typeKind()) {
    case TypeKind::TINYINT: {
      auto values = toList<int8_t, int64_t>(elements, offset, size);
      return negated ? notIn(values) : in(values);
    }
    case TypeKind::SMALLINT: {
      auto values = toList<int16_t, int64_t>(elements, offset, size);
      return negated ? notIn(values) : in(values);
    }
    case TypeKind::INTEGER: {
      auto values = toList<int32_t, int64_t>(elements, offset, size);
      return negated ? notIn(values) : in(values);
    }
    case TypeKind::BIGINT: {
      auto values = toList<int64_t, int64_t>(elements, offset, size);
      return negated ? notIn(values) : in(values);
    }
    case TypeKind::VARCHAR: {
      auto values = toList<StringView, std::string>(elements, offset, size);
      if (negated) {
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
  const auto lower = toConstant(lowerExpr, evaluator);
  if (!lower) {
    return nullptr;
  }
  const auto upper = toConstant(upperExpr, evaluator);
  if (!upper) {
    return nullptr;
  }
  // TODO: Implement fallback case
  // - negated -- false: AND(greaterThanOrEqual(lower), lessThanOrEqual(upper))
  // - negated -- true: OR(lessThan(lower), greaterThan(upper))
  if (lower->typeKind() != upper->typeKind()) {
    return nullptr;
  }
  switch (lower->typeKind()) {
    case TypeKind::TINYINT:
      if (negated) {
        return notBetween(
            singleValue<int8_t>(lower), singleValue<int8_t>(upper));
      }
      return between(singleValue<int8_t>(lower), singleValue<int8_t>(upper));
    case TypeKind::SMALLINT:
      if (negated) {
        return notBetween(
            singleValue<int16_t>(lower), singleValue<int16_t>(upper));
      }
      return between(singleValue<int16_t>(lower), singleValue<int16_t>(upper));
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

// static
std::unique_ptr<common::Filter> ExprToSubfieldFilterParser::makeOrFilter(
    std::unique_ptr<common::Filter> a,
    std::unique_ptr<common::Filter> b) {
  if (asBigintRange(a) && asBigintRange(b)) {
    return bigintOr(
        asUniquePtr<common::BigintRange>(std::move(a)),
        asUniquePtr<common::BigintRange>(std::move(b)));
  }

  if (asBigintRange(a) && asBigintMultiRange(b)) {
    const auto& ranges = asBigintMultiRange(b)->ranges();
    std::vector<std::unique_ptr<common::BigintRange>> newRanges;
    newRanges.emplace_back(asUniquePtr<common::BigintRange>(std::move(a)));
    for (const auto& range : ranges) {
      newRanges.emplace_back(asUniquePtr<common::BigintRange>(range->clone()));
    }

    std::sort(
        newRanges.begin(), newRanges.end(), [](const auto& a, const auto& b) {
          return a->lower() < b->lower();
        });

    return std::make_unique<common::BigintMultiRange>(
        std::move(newRanges), false);
  }

  if (asBigintMultiRange(a) && asBigintRange(b)) {
    return makeOrFilter(std::move(b), std::move(a));
  }

  return orFilter(std::move(a), std::move(b));
}

std::unique_ptr<common::Filter>
PrestoExprToSubfieldFilterParser::leafCallToSubfieldFilter(
    const core::CallTypedExpr& call,
    common::Subfield& subfield,
    core::ExpressionEvaluator* evaluator,
    bool negated) {
  const auto& inputs = call.inputs();
  if (inputs.empty() || !toSubfield(inputs[0].get(), subfield)) {
    return nullptr;
  }

  if (call.name() == "is_null") {
    VELOX_CHECK_EQ(inputs.size(), 1);
    if (negated) {
      return isNotNull();
    }
    return isNull();
  }

  if (call.name() == "between") {
    VELOX_CHECK_EQ(inputs.size(), 3);
    return makeBetweenFilter(inputs[1], inputs[2], evaluator, negated);
  }

  if (call.name() == "eq") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return negated ? makeNotEqualFilter(inputs[1], evaluator)
                   : makeEqualFilter(inputs[1], evaluator);
  }
  if (call.name() == "neq") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return negated ? makeEqualFilter(inputs[1], evaluator)
                   : makeNotEqualFilter(inputs[1], evaluator);
  }
  if (call.name() == "lte") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return negated ? makeGreaterThanFilter(inputs[1], evaluator)
                   : makeLessThanOrEqualFilter(inputs[1], evaluator);
  }
  if (call.name() == "lt") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return negated ? makeGreaterThanOrEqualFilter(inputs[1], evaluator)
                   : makeLessThanFilter(inputs[1], evaluator);
  }
  if (call.name() == "gte") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return negated ? makeLessThanFilter(inputs[1], evaluator)
                   : makeGreaterThanOrEqualFilter(inputs[1], evaluator);
  }
  if (call.name() == "gt") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return negated ? makeLessThanOrEqualFilter(inputs[1], evaluator)
                   : makeGreaterThanFilter(inputs[1], evaluator);
  }
  if (call.name() == "in") {
    VELOX_CHECK_EQ(inputs.size(), 2);
    return makeInFilter(inputs[1], evaluator, negated);
  }
  return nullptr;
}

std::pair<common::Subfield, std::unique_ptr<common::Filter>>
PrestoExprToSubfieldFilterParser::toSubfieldFilter(
    const core::TypedExprPtr& expr,
    core::ExpressionEvaluator* evaluator,
    bool negated) {
  const auto* call = asCall(expr.get());
  if (!call) {
    return {};
  }
  const auto& inputs = call->inputs();

  if (call->name() == "not") {
    VELOX_CHECK_EQ(inputs.size(), 1);
    return toSubfieldFilter(inputs[0], evaluator, !negated);
  }

  bool conjunction = call->name() == "and";
  if (conjunction || call->name() == "or") {
    if (inputs.empty()) {
      // We return not supported because in such case there's no subfield to
      // return. We expect user to handle constant true/false cases separately.
      return {};
    }
    auto result = toSubfieldFilter(inputs[0], evaluator, negated);
    if (!result.second) {
      return {};
    }
    conjunction = negated ? !conjunction : conjunction;
    for (size_t i = 1; i != inputs.size(); ++i) {
      auto current = toSubfieldFilter(inputs[i], evaluator, negated);
      if (!current.second) {
        return {};
      }
      if (result.first != current.first) {
        return {};
      }
      if (conjunction) {
        result.second = result.second->mergeWith(current.second.get());
      } else {
        result.second =
            makeOrFilter(std::move(result.second), std::move(current.second));
      }
    }
    return result;
  }

  common::Subfield subfield;
  auto filter = leafCallToSubfieldFilter(*call, subfield, evaluator, negated);
  return {std::move(subfield), std::move(filter)};
}

} // namespace facebook::velox::exec
