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

#include "velox/functions/lib/Sequence.h"

namespace facebook::velox::functions {

namespace {

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures() {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  signatures = {
      exec::FunctionSignatureBuilder()
          .returnType("array(bigint)")
          .argumentType("bigint")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(bigint)")
          .argumentType("bigint")
          .argumentType("bigint")
          .argumentType("bigint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(date)")
          .argumentType("date")
          .argumentType("date")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(date)")
          .argumentType("date")
          .argumentType("date")
          .argumentType("interval day to second")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(date)")
          .argumentType("date")
          .argumentType("date")
          .argumentType("interval year to month")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(timestamp)")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .argumentType("interval day to second")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(timestamp)")
          .argumentType("timestamp")
          .argumentType("timestamp")
          .argumentType("interval year to month")
          .build()};
  return signatures;
}

std::shared_ptr<exec::VectorFunction> create(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /*config*/) {
  if (inputArgs[0].type->isDate()) {
    if (inputArgs.size() > 2 && inputArgs[2].type->isIntervalYearMonth()) {
      return std::make_shared<SequenceFunction<int32_t, int32_t>>();
    }
    return std::make_shared<SequenceFunction<int32_t, int64_t>>();
  }

  switch (inputArgs[0].type->kind()) {
    case TypeKind::BIGINT:
      return std::make_shared<SequenceFunction<int64_t, int64_t>>();
    case TypeKind::TIMESTAMP:
      if (inputArgs.size() > 2 && inputArgs[2].type->isIntervalYearMonth()) {
        return std::make_shared<SequenceFunction<Timestamp, int32_t>>();
      }
      return std::make_shared<SequenceFunction<Timestamp, int64_t>>();
    default:
      VELOX_UNREACHABLE();
  }
}
} // namespace

VELOX_DECLARE_STATEFUL_VECTOR_FUNCTION(udf_sequence, signatures(), create);
} // namespace facebook::velox::functions
