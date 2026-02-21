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

#include "velox/type/fbhive/HiveTypeParser.h"

#include <cctype>
#include <string>
#include <utility>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::type::fbhive {
namespace {

/// Returns true only if 'str' contains digits.
bool isPositiveInteger(std::string_view str) {
  return !str.empty() &&
      std::find_if(str.begin(), str.end(), [](unsigned char c) {
        return !std::isdigit(c);
      }) == str.end();
}

bool isSupportedSpecialChar(char c) {
  static std::unordered_set<char> supported{'_', '$', '#'};
  return supported.count(c) == 1;
}

/// Returns whether `prefix` is non-empty case-insensitive prefix of `input`.
bool isPrefix(std::string_view input, std::string_view prefix) {
  return prefix.size() > 0 && input.size() >= prefix.size() &&
      std::equal(
             prefix.begin(),
             prefix.end(),
             input.begin(),
             folly::AsciiCaseInsensitive{});
}

} // namespace

HiveTypeParser::HiveTypeParser() {
  metadata_.resize(static_cast<size_t>(TokenType::MaxTokenType));
  setupMetadata<TokenType::Boolean, TypeKind::BOOLEAN>("boolean");
  setupMetadata<TokenType::Byte, TypeKind::TINYINT>("tinyint");
  setupMetadata<TokenType::Short, TypeKind::SMALLINT>("smallint");
  setupMetadata<TokenType::Integer, TypeKind::INTEGER>({"integer", "int"});
  setupMetadata<TokenType::Long, TypeKind::BIGINT>("bigint");
  setupMetadata<TokenType::Date, TypeKind::INTEGER>("date");
  setupMetadata<TokenType::Time, TypeKind::BIGINT>("time");
  setupMetadata<TokenType::Float, TypeKind::REAL>({"float", "real"});
  setupMetadata<TokenType::Double, TypeKind::DOUBLE>("double");
  setupMetadata<TokenType::Decimal, TypeKind::BIGINT>("decimal");
  setupMetadata<TokenType::String, TypeKind::VARCHAR>({"string", "varchar"});
  setupMetadata<TokenType::Binary, TypeKind::VARBINARY>(
      {"binary", "varbinary"});
  setupMetadata<TokenType::Timestamp, TypeKind::TIMESTAMP>("timestamp");
  setupMetadata<TokenType::Opaque, TypeKind::OPAQUE>("opaque");
  setupMetadata<TokenType::List, TypeKind::ARRAY>("array");
  setupMetadata<TokenType::Map, TypeKind::MAP>("map");
  setupMetadata<TokenType::Struct, TypeKind::ROW>({"struct", "row"});
  setupMetadata<TokenType::StartSubType, TypeKind::INVALID>("<");
  setupMetadata<TokenType::EndSubType, TypeKind::INVALID>(">");
  setupMetadata<TokenType::Colon, TypeKind::INVALID>(":");
  setupMetadata<TokenType::Comma, TypeKind::INVALID>(",");
  setupMetadata<TokenType::LeftRoundBracket, TypeKind::INVALID>("(");
  setupMetadata<TokenType::RightRoundBracket, TypeKind::INVALID>(")");
  setupMetadata<TokenType::Number, TypeKind::INVALID>();
  setupMetadata<TokenType::Identifier, TypeKind::INVALID>();
  setupMetadata<TokenType::EndOfStream, TypeKind::INVALID>();
}

TypePtr HiveTypeParser::parse(std::string_view input) {
  remaining_ = input;
  Result result = parseType();
  VELOX_CHECK(
      remaining_.size() == 0 || TokenType::EndOfStream == lookAhead(),
      "Input remaining after parsing the Hive type \"{}\"\n"
      "Remaining: \"{}\"",
      input,
      remaining_);
  return result.type;
}

Result HiveTypeParser::parseType() {
  Token nt = nextToken();
  VELOX_CHECK(!nt.isEOS(), "Unexpected end of stream parsing type!!!");

  if (!nt.isValidType()) {
    VELOX_FAIL(
        "Unexpected token {} at {}. typeKind = {}",
        nt.value,
        remaining_,
        nt.typeKind());
  }

  if (nt.isPrimitiveType()) {
    if (nt.metadata->tokenString[0] == "decimal") {
      eatToken(TokenType::LeftRoundBracket);
      Token precision = nextToken();
      VELOX_CHECK(
          isPositiveInteger(precision.value),
          "Decimal precision must be a positive integer");
      eatToken(TokenType::Comma);
      Token scale = nextToken();
      VELOX_CHECK(
          isPositiveInteger(scale.value),
          "Decimal scale must be a positive integer");
      eatToken(TokenType::RightRoundBracket);
      return Result{DECIMAL(
          std::atoi(precision.value.data()), std::atoi(scale.value.data()))};
    } else if (nt.metadata->tokenString[0] == "date") {
      return Result{DATE()};
    } else if (nt.metadata->tokenString[0] == "time") {
      return Result{TIME()};
    }
    auto scalarType = createScalarType(nt.typeKind());
    VELOX_CHECK_NOT_NULL(
        scalarType, "Returned a null scalar type for ", nt.typeKind());
    if (nt.metadata->tokenType == TokenType::String &&
        lookAhead() == TokenType::LeftRoundBracket) {
      eatToken(TokenType::LeftRoundBracket);
      Token length = nextToken();
      VELOX_CHECK(
          isPositiveInteger(length.value),
          "Varchar length must be a positive integer");
      eatToken(TokenType::RightRoundBracket);
    }
    return Result{scalarType};
  } else if (nt.isOpaqueType()) {
    eatToken(TokenType::StartSubType);
    std::string_view innerTypeName =
        eatToken(TokenType::Identifier, true).value;
    eatToken(TokenType::EndSubType);

    // TODO: `getTypeIdForOpaqueTypeAlias()` should take a std::string_view so
    // we don't need to needlessly construct a std::string.
    auto typeIndex = getTypeIdForOpaqueTypeAlias(std::string(innerTypeName));
    auto instance = std::make_shared<const OpaqueType>(typeIndex);
    return Result{instance};
  } else {
    ResultList resultList = parseTypeList(TypeKind::ROW == nt.typeKind());
    switch (nt.typeKind()) {
      case velox::TypeKind::ROW:
        return Result{velox::ROW(
            std::move(resultList.names), std::move(resultList.typelist))};
      case velox::TypeKind::MAP: {
        VELOX_CHECK_EQ(
            resultList.typelist.size(),
            2,
            "wrong param count for map type def");
        return Result{
            velox::MAP(resultList.typelist.at(0), resultList.typelist.at(1))};
      }
      case velox::TypeKind::ARRAY: {
        VELOX_CHECK_EQ(
            resultList.typelist.size(),
            1,
            "wrong param count for array type def");
        return Result{velox::ARRAY(resultList.typelist.at(0))};
      }
      default:
        VELOX_FAIL(
            "Unsupported kind: '{}'", TypeKindName::toName(nt.typeKind()));
    }
  }
}

ResultList HiveTypeParser::parseTypeList(bool hasFieldNames) {
  std::vector<TypePtr> subTypeList{};
  std::vector<std::string> names{};

  eatToken(TokenType::StartSubType);
  while (true) {
    if (TokenType::EndSubType == lookAhead()) {
      eatToken(TokenType::EndSubType);
      return ResultList{std::move(subTypeList), std::move(names)};
    }

    std::string_view fieldName;
    if (hasFieldNames) {
      fieldName = eatToken(TokenType::Identifier, true).value;
      eatToken(TokenType::Colon);
      names.emplace_back(fieldName);
    }

    Result result = parseType();
    subTypeList.push_back(result.type);
    if (TokenType::Comma == lookAhead()) {
      eatToken(TokenType::Comma);
    }
  }
}

TokenType HiveTypeParser::lookAhead() const {
  return nextToken(remaining_).tokenType();
}

Token HiveTypeParser::eatToken(TokenType tokenType, bool ignorePredefined) {
  TokenAndRemaining token = nextToken(remaining_, ignorePredefined);
  if (token.tokenType() == tokenType) {
    remaining_ = token.remaining;
    return token;
  }

  VELOX_FAIL("Unexpected token '{}'", token.remaining);
}

Token HiveTypeParser::nextToken(bool ignorePredefined) {
  TokenAndRemaining token = nextToken(remaining_, ignorePredefined);
  remaining_ = token.remaining;
  return token;
}

TokenAndRemaining HiveTypeParser::nextToken(
    std::string_view sv,
    bool ignorePredefined) const {
  while (!sv.empty() && isspace(sv.front())) {
    sv.remove_prefix(1);
  }

  if (sv.empty()) {
    return makeExtendedToken(getMetadata(TokenType::EndOfStream), sv, 0);
  }

  if (!ignorePredefined) {
    for (auto& metadata : metadata_) {
      for (auto& token : metadata->tokenString) {
        std::string_view match(token);
        if (isPrefix(sv, match)) {
          return makeExtendedToken(metadata.get(), sv, match.size());
        }
      }
    }
  }

  auto iter = sv.cbegin();
  size_t len = 0;
  while (isalnum(*iter) || isSupportedSpecialChar(*iter)) {
    ++len;
    ++iter;
  }

  if (len > 0) {
    return makeExtendedToken(getMetadata(TokenType::Identifier), sv, len);
  }

  VELOX_FAIL("Bad Token at '{}'", sv);
}

TokenType Token::tokenType() const {
  return metadata->tokenType;
}

TypeKind Token::typeKind() const {
  return metadata->typeKind;
}

bool Token::isPrimitiveType() const {
  return metadata->isPrimitiveType;
}

bool Token::isValidType() const {
  return metadata->typeKind != TypeKind::INVALID;
}

bool Token::isEOS() const {
  return metadata->tokenType == TokenType::EndOfStream;
}

bool Token::isOpaqueType() const {
  return metadata->tokenType == TokenType::Opaque;
}

int8_t HiveTypeParser::makeTokenId(TokenType tokenType) const {
  return static_cast<int8_t>(tokenType);
}

TokenAndRemaining HiveTypeParser::makeExtendedToken(
    TokenMetadata* tokenMetadata,
    std::string_view sv,
    size_t len) const {
  std::string_view spmatch{sv.data(), len};
  sv.remove_prefix(len);

  TokenAndRemaining result;
  result.metadata = tokenMetadata;
  result.value = spmatch;
  result.remaining = sv;
  return result;
}

TokenMetadata* HiveTypeParser::getMetadata(TokenType type) const {
  auto& value = metadata_[makeTokenId(type)];
  return value.get();
}

} // namespace facebook::velox::type::fbhive
