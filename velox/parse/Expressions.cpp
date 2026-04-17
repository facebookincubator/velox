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
#include "velox/parse/Expressions.h"

namespace facebook::velox::core {

namespace {
const auto& windowTypeNames() {
  static const folly::F14FastMap<WindowCallExpr::WindowType, std::string_view>
      kNames = {
          {WindowCallExpr::WindowType::kRows, "ROWS"},
          {WindowCallExpr::WindowType::kRange, "RANGE"},
          {WindowCallExpr::WindowType::kGroups, "GROUPS"},
      };
  return kNames;
}

const auto& boundTypeNames() {
  static const folly::F14FastMap<WindowCallExpr::BoundType, std::string_view>
      kNames = {
          {WindowCallExpr::BoundType::kCurrentRow, "CURRENT ROW"},
          {WindowCallExpr::BoundType::kUnboundedPreceding,
           "UNBOUNDED PRECEDING"},
          {WindowCallExpr::BoundType::kUnboundedFollowing,
           "UNBOUNDED FOLLOWING"},
          {WindowCallExpr::BoundType::kPreceding, "PRECEDING"},
          {WindowCallExpr::BoundType::kFollowing, "FOLLOWING"},
      };
  return kNames;
}

const auto& kindNames() {
  static const folly::F14FastMap<IExpr::Kind, std::string_view> kNames = {
      {IExpr::Kind::kInput, "Input"},
      {IExpr::Kind::kFieldAccess, "FieldAccess"},
      {IExpr::Kind::kCall, "Call"},
      {IExpr::Kind::kCast, "Cast"},
      {IExpr::Kind::kConstant, "Constant"},
      {IExpr::Kind::kLambda, "Lambda"},
      {IExpr::Kind::kSubquery, "Subquery"},
      {IExpr::Kind::kAggregate, "Aggregate"},
      {IExpr::Kind::kWindow, "Window"},
  };
  return kNames;
}
} // namespace

VELOX_DEFINE_EMBEDDED_ENUM_NAME(IExpr, Kind, kindNames)
VELOX_DEFINE_EMBEDDED_ENUM_NAME(WindowCallExpr, WindowType, windowTypeNames);
VELOX_DEFINE_EMBEDDED_ENUM_NAME(WindowCallExpr, BoundType, boundTypeNames);

bool FieldAccessExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kFieldAccess)) {
    return false;
  }

  auto* otherField = other.as<FieldAccessExpr>();
  return name_ == otherField->name_ && compareAliasAndInputs(other);
}

size_t FieldAccessExpr::localHash() const {
  return std::hash<std::string>{}(name_);
}

namespace {
std::string escapeName(const std::string& name) {
  return folly::cEscape<std::string>(name);
}
} // namespace

std::string FieldAccessExpr::toString() const {
  if (isRootColumn()) {
    return appendAliasIfExists(fmt::format("\"{}\"", escapeName(name_)));
  }

  return appendAliasIfExists(
      fmt::format("dot({},\"{}\")", input()->toString(), escapeName(name_)));
}

bool CallExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kCall)) {
    return false;
  }

  auto* otherCall = other.as<CallExpr>();
  return name_ == otherCall->name_ && compareAliasAndInputs(other);
}

size_t CallExpr::localHash() const {
  return std::hash<std::string>{}(name_);
}

std::string CallExpr::toString() const {
  std::string buf{name_ + "("};
  bool first = true;
  for (auto& f : inputs()) {
    if (!first) {
      buf += ",";
    }
    buf += f->toString();
    first = false;
  }
  buf += ")";
  return appendAliasIfExists(buf);
}

bool AggregateCallExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kAggregate)) {
    return false;
  }

  auto* otherAgg = other.as<AggregateCallExpr>();
  if (name() != otherAgg->name() || distinct_ != otherAgg->distinct_) {
    return false;
  }

  // Compare filter.
  if ((filter_ == nullptr) != (otherAgg->filter_ == nullptr)) {
    return false;
  }
  if (filter_ != nullptr && *filter_ != *otherAgg->filter_) {
    return false;
  }

  // Compare orderBy.
  if (orderBy_.size() != otherAgg->orderBy_.size()) {
    return false;
  }
  for (size_t i = 0; i < orderBy_.size(); ++i) {
    if (orderBy_[i].ascending != otherAgg->orderBy_[i].ascending ||
        orderBy_[i].nullsFirst != otherAgg->orderBy_[i].nullsFirst ||
        *orderBy_[i].expr != *otherAgg->orderBy_[i].expr) {
      return false;
    }
  }

  return compareAliasAndInputs(other);
}

size_t AggregateCallExpr::localHash() const {
  auto hash = std::hash<std::string>{}(name());
  hash = bits::hashMix(hash, std::hash<bool>{}(distinct_));
  if (filter_ != nullptr) {
    hash = bits::hashMix(hash, filter_->hash());
  }
  for (const auto& key : orderBy_) {
    hash = bits::hashMix(hash, key.expr->hash());
    hash = bits::hashMix(hash, std::hash<bool>{}(key.ascending));
    hash = bits::hashMix(hash, std::hash<bool>{}(key.nullsFirst));
  }
  return hash;
}

std::string AggregateCallExpr::toString() const {
  std::ostringstream out;
  out << name() << "(";
  if (distinct_) {
    out << "DISTINCT ";
  }
  for (size_t i = 0; i < inputs().size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << inputs()[i]->toString();
  }
  out << ")";
  if (filter_ != nullptr) {
    out << " FILTER(WHERE " << filter_->toString() << ")";
  }
  if (!orderBy_.empty()) {
    out << " ORDER BY ";
    for (size_t i = 0; i < orderBy_.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << orderBy_[i].expr->toString();
      out << (orderBy_[i].ascending ? " ASC" : " DESC");
      out << (orderBy_[i].nullsFirst ? " NULLS FIRST" : " NULLS LAST");
    }
  }
  return appendAliasIfExists(out.str());
}

bool WindowCallExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kWindow)) {
    return false;
  }

  auto* otherWin = other.as<WindowCallExpr>();
  if (name() != otherWin->name() || ignoreNulls_ != otherWin->ignoreNulls_) {
    return false;
  }

  // Compare partition keys.
  if (partitionKeys_.size() != otherWin->partitionKeys_.size()) {
    return false;
  }
  for (size_t i = 0; i < partitionKeys_.size(); ++i) {
    if (*partitionKeys_[i] != *otherWin->partitionKeys_[i]) {
      return false;
    }
  }

  // Compare order by keys.
  if (orderByKeys_.size() != otherWin->orderByKeys_.size()) {
    return false;
  }
  for (size_t i = 0; i < orderByKeys_.size(); ++i) {
    if (orderByKeys_[i].ascending != otherWin->orderByKeys_[i].ascending ||
        orderByKeys_[i].nullsFirst != otherWin->orderByKeys_[i].nullsFirst ||
        *orderByKeys_[i].expr != *otherWin->orderByKeys_[i].expr) {
      return false;
    }
  }

  // Compare frame.
  if (frame_.has_value() != otherWin->frame_.has_value()) {
    return false;
  }
  if (frame_.has_value()) {
    const auto& frame = frame_.value();
    const auto& otherFrame = otherWin->frame_.value();
    if (frame.type != otherFrame.type ||
        frame.startType != otherFrame.startType ||
        frame.endType != otherFrame.endType) {
      return false;
    }
    if ((frame.startValue == nullptr) != (otherFrame.startValue == nullptr)) {
      return false;
    }
    if (frame.startValue != nullptr &&
        *frame.startValue != *otherFrame.startValue) {
      return false;
    }
    if ((frame.endValue == nullptr) != (otherFrame.endValue == nullptr)) {
      return false;
    }
    if (frame.endValue != nullptr && *frame.endValue != *otherFrame.endValue) {
      return false;
    }
  }

  return compareAliasAndInputs(other);
}

size_t WindowCallExpr::localHash() const {
  auto hash = std::hash<std::string>{}(name());
  hash = bits::hashMix(hash, std::hash<bool>{}(ignoreNulls_));
  for (const auto& key : partitionKeys_) {
    hash = bits::hashMix(hash, key->hash());
  }
  for (const auto& key : orderByKeys_) {
    hash = bits::hashMix(hash, key.expr->hash());
    hash = bits::hashMix(hash, std::hash<bool>{}(key.ascending));
    hash = bits::hashMix(hash, std::hash<bool>{}(key.nullsFirst));
  }
  if (frame_.has_value()) {
    hash =
        bits::hashMix(hash, std::hash<int>{}(static_cast<int>(frame_->type)));
    hash = bits::hashMix(
        hash, std::hash<int>{}(static_cast<int>(frame_->startType)));
    hash = bits::hashMix(
        hash, std::hash<int>{}(static_cast<int>(frame_->endType)));
  }
  return hash;
}

std::string WindowCallExpr::toString() const {
  std::ostringstream out;
  out << name() << "(";
  for (size_t i = 0; i < inputs().size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << inputs()[i]->toString();
  }
  out << ") OVER (";

  bool needsSpace = false;
  if (!partitionKeys_.empty()) {
    out << "PARTITION BY ";
    for (size_t i = 0; i < partitionKeys_.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << partitionKeys_[i]->toString();
    }
    needsSpace = true;
  }

  if (!orderByKeys_.empty()) {
    if (needsSpace) {
      out << " ";
    }
    out << "ORDER BY ";
    for (size_t i = 0; i < orderByKeys_.size(); ++i) {
      if (i > 0) {
        out << ",";
      }
      out << orderByKeys_[i].expr->toString();
      out << (orderByKeys_[i].ascending ? " ASC" : " DESC");
      out << (orderByKeys_[i].nullsFirst ? " NULLS FIRST" : " NULLS LAST");
    }
    needsSpace = true;
  }

  if (frame_.has_value()) {
    if (needsSpace) {
      out << " ";
    }
    const auto& frame = frame_.value();
    switch (frame.type) {
      case WindowType::kRows:
        out << "ROWS";
        break;
      case WindowType::kRange:
        out << "RANGE";
        break;
      case WindowType::kGroups:
        out << "GROUPS";
        break;
    }
    out << " BETWEEN ";

    auto formatBound = [&](BoundType boundType, const ExprPtr& value) {
      switch (boundType) {
        case BoundType::kUnboundedPreceding:
          out << "UNBOUNDED PRECEDING";
          break;
        case BoundType::kPreceding:
          out << value->toString() << " PRECEDING";
          break;
        case BoundType::kCurrentRow:
          out << "CURRENT ROW";
          break;
        case BoundType::kFollowing:
          out << value->toString() << " FOLLOWING";
          break;
        case BoundType::kUnboundedFollowing:
          out << "UNBOUNDED FOLLOWING";
          break;
      }
    };

    formatBound(frame.startType, frame.startValue);
    out << " AND ";
    formatBound(frame.endType, frame.endValue);
  }

  out << ")";
  return appendAliasIfExists(out.str());
}

bool CastExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kCast)) {
    return false;
  }

  auto* otherCast = other.as<CastExpr>();
  return *type_ == (*otherCast->type_) && isTryCast_ == otherCast->isTryCast_ &&
      compareAliasAndInputs(other);
}

size_t CastExpr::localHash() const {
  return bits::hashMix(type_->hashKind(), std::hash<bool>{}(isTryCast_));
}

std::string CastExpr::toString() const {
  return appendAliasIfExists(
      fmt::format(
          "{}({} as {})",
          isTryCast_ ? "try_cast" : "cast",
          input()->toString(),
          type_->toString()));
}

bool ConstantExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kConstant)) {
    return false;
  }

  auto* otherConstant = other.as<ConstantExpr>();
  return *type_ == (*otherConstant->type_) && value_ == otherConstant->value_ &&
      compareAliasAndInputs(other);
}

size_t ConstantExpr::localHash() const {
  return bits::hashMix(type_->hashKind(), value_.hash());
}

std::string ConstantExpr::toString() const {
  return appendAliasIfExists(value_.toStringAsVector(type_));
}

bool LambdaExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kLambda)) {
    return false;
  }

  auto* otherLambda = other.as<LambdaExpr>();
  return arguments_ == otherLambda->arguments_ && *body_ == *otherLambda->body_;
}

size_t LambdaExpr::localHash() const {
  size_t hash = 0;
  for (const auto& arg : arguments_) {
    hash = bits::hashMix(hash, std::hash<std::string>{}(arg));
  }
  return bits::hashMix(hash, body_->hash());
}

std::string LambdaExpr::toString() const {
  std::ostringstream out;

  if (arguments_.size() > 1) {
    out << "(" << folly::join(", ", arguments_) << ")";
  } else {
    out << arguments_[0];
  }
  out << " -> " << body_->toString();
  return out.str();
}

std::string ConcatExpr::toString() const {
  std::ostringstream out;
  out << "row(";
  for (size_t i = 0; i < inputs().size(); ++i) {
    if (i > 0) {
      out << ", ";
    }
    out << inputs()[i]->toString() << " as " << fieldNames_[i];
  }
  out << ")";
  return appendAliasIfExists(out.str());
}

bool ConcatExpr::operator==(const IExpr& other) const {
  if (!other.is(Kind::kConcat)) {
    return false;
  }
  auto* otherConcat = other.as<ConcatExpr>();
  return fieldNames_ == otherConcat->fieldNames_ &&
      compareAliasAndInputs(other);
}

size_t ConcatExpr::localHash() const {
  size_t hash = 0;
  for (const auto& name : fieldNames_) {
    hash = bits::hashMix(hash, std::hash<std::string>{}(name));
  }
  return hash;
}

} // namespace facebook::velox::core
