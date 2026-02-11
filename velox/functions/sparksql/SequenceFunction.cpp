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
#include "velox/functions/sparksql/SequenceFunction.h"

namespace facebook::velox::functions::sparksql {

std::vector<std::shared_ptr<exec::FunctionSignature>>
sparkSequenceSignatures() {
  // Integer types: 2-arg (start, stop) and 3-arg (start, stop, step).
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures = {
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
          .returnType("array(integer)")
          .argumentType("integer")
          .argumentType("integer")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(integer)")
          .argumentType("integer")
          .argumentType("integer")
          .argumentType("integer")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(smallint)")
          .argumentType("smallint")
          .argumentType("smallint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(smallint)")
          .argumentType("smallint")
          .argumentType("smallint")
          .argumentType("smallint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(tinyint)")
          .argumentType("tinyint")
          .argumentType("tinyint")
          .build(),
      exec::FunctionSignatureBuilder()
          .returnType("array(tinyint)")
          .argumentType("tinyint")
          .argumentType("tinyint")
          .argumentType("tinyint")
          .build(),
  };
  return signatures;
}

std::shared_ptr<exec::VectorFunction> makeSparkSequence(
    const std::string& /* name */,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& /* config */) {
  switch (inputArgs[0].type->kind()) {
    case TypeKind::BIGINT:
      return std::make_shared<SparkSequenceFunction<int64_t>>();
    case TypeKind::INTEGER:
      return std::make_shared<SparkSequenceFunction<int32_t>>();
    case TypeKind::SMALLINT:
      return std::make_shared<SparkSequenceFunction<int16_t>>();
    case TypeKind::TINYINT:
      return std::make_shared<SparkSequenceFunction<int8_t>>();
    default:
      VELOX_UNREACHABLE(
          "Unexpected type for Spark sequence: {}",
          inputArgs[0].type->toString());
  }
}

} // namespace facebook::velox::functions::sparksql
