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

#include "velox/exec/fuzzer/PrestoQueryRunnerTimestampWithTimeZoneTransform.h"
#include "velox/functions/prestosql/types/TimestampWithTimeZoneType.h"
#include "velox/parse/Expressions.h"
#include "velox/type/tz/TimeZoneMap.h"
#include "velox/vector/FlatVector.h"

namespace facebook::velox::exec::test {
std::string TimestampWithTimeZoneTransform::format(
    const int64_t timestampWithTimeZone) const {
  const auto timestamp = unpackTimestampUtc(timestampWithTimeZone);
  const auto timeZoneId = unpackZoneKeyId(timestampWithTimeZone);
  auto* timezonePtr = tz::locateZone(tz::getTimeZoneName(timeZoneId));

  const auto maxResultSize = jodaDateTime_->maxResultSize(timezonePtr);
  std::string str;
  str.resize(maxResultSize);
  const auto resultSize =
      jodaDateTime_->format(timestamp, timezonePtr, maxResultSize, str.data());
  str.resize(resultSize);

  return str;
}

// Convert a Vector of TimestampWithTimeZone to a RowVector of BIGINT and
// VARCHAR (millisUtc and time_zone).
VectorPtr TimestampWithTimeZoneTransform::transform(
    const VectorPtr& vector,
    const SelectivityVector& rows) const {
  VELOX_CHECK(isTimestampWithTimeZoneType(vector->type()));
  DecodedVector decoded(*vector);
  const auto* base = decoded.base()->as<SimpleVector<int64_t>>();

  VectorPtr result;

  if (base->isFlatEncoding()) {
    auto resultFlat = BaseVector::create<FlatVector<StringView>>(
        VARCHAR(), base->size(), base->pool());
    result = resultFlat;

    rows.applyToSelected([&](vector_size_t row) {
      auto index = decoded.index(row);
      if (decoded.isNullAt(row)) {
        resultFlat->setNull(index, true);
      } else {
        std::string str = format(base->valueAt(index));

        resultFlat->set(index, StringView(str));
      }
    });
  } else {
    VELOX_CHECK(base->isConstantEncoding());
    if (rows.countSelected() == 0 || base->isNullAt(0)) {
      result =
          BaseVector::createNullConstant(VARCHAR(), base->size(), base->pool());
    } else {
      std::string str = format(base->valueAt(0));

      result = BaseVector::createConstant(
          VARCHAR(), StringView(str), base->size(), base->pool());
    }
  }

  if (!decoded.isIdentityMapping()) {
    result = decoded.wrap(result, *vector, vector->size());
  }

  return result;
}

// Applies from_unixtime to a RowVector of BIGINT and VARCHAR (millisUtc and
// time_zone) to produce values of type TimestampWithTimeZone.
core::ExprPtr TimestampWithTimeZoneTransform::projectionExpr(
    const TypePtr& type,
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  VELOX_CHECK(isTimestampWithTimeZoneType(type));

  // format_datetime with the ZZZ pattern produces time zones that need to be
  // parsed with either the ZZZ or ZZ pattern, to handle this we try parsing
  // with the ZZZ pattern first, and then the ZZ pattern if that fails.
  // coalesce(try(parse_datetime(..., '... ZZZ')), parse_datetime(..., '...
  // ZZ'))
  return std::make_shared<core::CallExpr>(
      "coalesce",
      std::vector<core::ExprPtr>{
          std::make_shared<core::CallExpr>(
              "try",
              std::vector<core::ExprPtr>{std::make_shared<core::CallExpr>(
                  "parse_datetime",
                  std::vector<core::ExprPtr>{
                      inputExpr,
                      std::make_shared<core::ConstantExpr>(
                          VARCHAR(),
                          variant::create<TypeKind::VARCHAR>(kFormat),
                          std::nullopt)},
                  std::nullopt)},
              std::nullopt),
          std::make_shared<core::CallExpr>(
              "parse_datetime",
              std::vector<core::ExprPtr>{
                  inputExpr,
                  std::make_shared<core::ConstantExpr>(
                      VARCHAR(),
                      variant::create<TypeKind::VARCHAR>(kBackupFormat),
                      std::nullopt)},
              std::nullopt)},
      columnAlias);
}
} // namespace facebook::velox::exec::test
