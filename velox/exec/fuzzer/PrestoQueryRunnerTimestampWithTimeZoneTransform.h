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

#pragma once

#include "velox/exec/fuzzer/PrestoQueryRunnerIntermediateTypeTransforms.h"
#include "velox/functions/lib/DateTimeFormatter.h"

namespace facebook::velox::exec::test {
class TimestampWithTimeZoneTransform : public IntermediateTypeTransform {
 public:
  VectorPtr transform(const VectorPtr& vector, const SelectivityVector& rows)
      const override;
  core::ExprPtr projectionExpr(
      const TypePtr& type,
      const core::ExprPtr& inputExpr,
      const std::string& columnAlias) const override;

 private:
  static inline const std::string kFormat = "yyyy-MM-dd HH:mm:ss.SSS ZZZ";
  static inline const std::string kBackupFormat = "yyyy-MM-dd HH:mm:ss.SSS ZZ";

  std::string format(const int64_t timestampWithTimeZone) const;

  std::shared_ptr<functions::DateTimeFormatter> jodaDateTime_ =
      functions::buildJodaDateTimeFormatter(kFormat).value();
};
} // namespace facebook::velox::exec::test
