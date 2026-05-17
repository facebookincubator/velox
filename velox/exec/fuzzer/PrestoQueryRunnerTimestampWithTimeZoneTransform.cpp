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
#include "velox/parse/Expressions.h"

namespace facebook::velox::exec::test {
namespace {
const std::string kTimestampFormat = "yyyy-MM-dd HH:mm:ss.SSS ZZ";
const std::string kZoneFormat = "ZZZ";
} // namespace

// Transports TIMESTAMP WITH TIME ZONE values as a row of
// (offset-qualified local timestamp text, original zone text). The offset
// keeps the local timestamp unambiguous across engines, while the separate zone
// text lets us reconstruct the original zone key for zone-sensitive functions.
core::ExprPtr TimestampWithTimeZoneTransform::projectToTargetType(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  auto timestampText = std::make_shared<core::CallExpr>(
      "format_datetime",
      std::vector<core::ExprPtr>{
          inputExpr,
          std::make_shared<core::ConstantExpr>(
              VARCHAR(),
              variant::create<TypeKind::VARCHAR>(kTimestampFormat),
              std::nullopt)},
      std::nullopt);
  auto zoneText = std::make_shared<core::CallExpr>(
      "format_datetime",
      std::vector<core::ExprPtr>{
          inputExpr,
          std::make_shared<core::ConstantExpr>(
              VARCHAR(),
              variant::create<TypeKind::VARCHAR>(kZoneFormat),
              std::nullopt)},
      std::nullopt);
  return std::make_shared<core::CallExpr>(
      "switch",
      std::vector<core::ExprPtr>{
          std::make_shared<core::CallExpr>(
              "is_null", std::vector<core::ExprPtr>{inputExpr}, std::nullopt),
          std::make_shared<core::ConstantExpr>(
              targetType_, variant::null(TypeKind::ROW), std::nullopt),
          std::make_shared<core::CastExpr>(
              targetType_,
              std::make_shared<core::CallExpr>(
                  "row_constructor",
                  std::vector<core::ExprPtr>{timestampText, zoneText},
                  std::nullopt),
              false,
              std::nullopt)},
      columnAlias);
}

// Reconstructs TIMESTAMP WITH TIME ZONE values from the transport row.
core::ExprPtr TimestampWithTimeZoneTransform::projectToIntermediateType(
    const core::ExprPtr& inputExpr,
    const std::string& columnAlias) const {
  auto timestampText = std::make_shared<core::FieldAccessExpr>(
      "timestamp_text",
      "timestamp_text",
      std::vector<core::ExprPtr>{inputExpr});
  auto zoneText = std::make_shared<core::FieldAccessExpr>(
      "zone_text", "zone_text", std::vector<core::ExprPtr>{inputExpr});
  return std::make_shared<core::CallExpr>(
      "at_timezone",
      std::vector<core::ExprPtr>{
          std::make_shared<core::CallExpr>(
              "parse_datetime",
              std::vector<core::ExprPtr>{
                  timestampText,
                  std::make_shared<core::ConstantExpr>(
                      VARCHAR(),
                      variant::create<TypeKind::VARCHAR>(kTimestampFormat),
                      std::nullopt)},
              std::nullopt),
          zoneText,
      },
      columnAlias);
}
} // namespace facebook::velox::exec::test
