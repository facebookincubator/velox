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
#include "velox/parse/SqlReservedWords.h"

namespace facebook::velox::core {

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

/// Escapes a string for SQL representation.
/// Single quotes are doubled, and other SQL special characters are handled.
std::string escapeSqlString(const std::string& str) {
  std::string result;
  result.reserve(str.size() * 2); // Reserve space for potential escaping

  for (char c : str) {
    if (c == '\'') {
      // Single quote becomes two single quotes
      result += "''";
    } else if (c == '\\') {
      // Backslash becomes double backslash
      result += "\\\\";
    } else if (c == '\n') {
      // Newline
      result += "\\n";
    } else if (c == '\r') {
      // Carriage return
      result += "\\r";
    } else if (c == '\t') {
      // Tab
      result += "\\t";
    } else if (c == '\0') {
      // Null character
      result += "\\0";
    } else {
      result += c;
    }
  }

  return result;
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
  std::string buf;

  // Escape function name if it's a SQL reserved word
  if (isSqlReservedWord(name_)) {
    buf = "\"" + name_ + "\"(";
  } else {
    buf = name_ + "(";
  }

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
      "cast(" + input()->toString() + " as " + type_->toString() + ")");
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
  // Special handling for VARCHAR type
  if (value_.kind() == TypeKind::VARCHAR) {
    std::string strValue = value_.toStringAsVector(type_);
    // Apply SQL escaping, wrap in single quotes, and then add alias
    return appendAliasIfExists("'" + escapeSqlString(strValue) + "'");
  }

  // Special handling for ARRAY type
  if (value_.kind() == TypeKind::ARRAY) {
    const auto& arrayValue = value_.value<TypeKind::ARRAY>();
    auto elementType = type_->childAt(0);
    std::string result = "array[";

    for (size_t i = 0; i < arrayValue.size(); ++i) {
      if (i > 0) {
        result += ", ";
      }

      // Create a constant expression for each element and call toString recursively
      auto elementExpr = std::make_shared<ConstantExpr>(
							elementType, arrayValue.at(i), std::nullopt);
      result += elementExpr->toString();
    }

    result += "]";
    return appendAliasIfExists(result);
  }

  // Default behavior for non-VARCHAR types
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

} // namespace facebook::velox::core
