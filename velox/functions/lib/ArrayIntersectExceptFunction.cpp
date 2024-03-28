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
#include "velox/functions/lib/ArrayIntersectExceptFunction.h"

namespace facebook::velox::functions {

DecodedVector* decodeArrayElements(
    exec::LocalDecodedVector& arrayDecoder,
    exec::LocalDecodedVector& elementsDecoder,
    const SelectivityVector& rows) {
  auto decodedVector = arrayDecoder.get();
  auto baseArrayVector = arrayDecoder->base()->as<ArrayVector>();

  // Decode and acquire array elements vector.
  auto elementsVector = baseArrayVector->elements();
  auto elementsSelectivityRows = toElementRows(
      elementsVector->size(), rows, baseArrayVector, decodedVector->indices());
  elementsDecoder.get()->decode(*elementsVector, elementsSelectivityRows);
  auto decodedElementsVector = elementsDecoder.get();
  return decodedElementsVector;
}

std::vector<std::shared_ptr<exec::FunctionSignature>> signatures(
    const std::string& returnTypeTemplate) {
  std::vector<std::shared_ptr<exec::FunctionSignature>> signatures;
  for (const auto& type : exec::primitiveTypeNames()) {
    signatures.push_back(
        exec::FunctionSignatureBuilder()
            .returnType(
                fmt::format(fmt::runtime(returnTypeTemplate.c_str()), type))
            .argumentType(fmt::format("array({})", type))
            .argumentType(fmt::format("array({})", type))
            .build());
  }
  return signatures;
}

void validateMatchingArrayTypes(
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const std::string& name,
    vector_size_t expectedArgCount) {
  VELOX_USER_CHECK_EQ(
      inputArgs.size(),
      expectedArgCount,
      "{} requires exactly {} parameters",
      name,
      expectedArgCount);

  auto arrayType = inputArgs.front().type;
  VELOX_USER_CHECK_EQ(
      arrayType->kind(),
      TypeKind::ARRAY,
      "{} requires arguments of type ARRAY",
      name);

  for (auto& arg : inputArgs) {
    VELOX_USER_CHECK(
        arrayType->kindEquals(arg.type),
        "{} function requires all arguments of the same type: {} vs. {}",
        name,
        arg.type->toString(),
        arrayType->toString());
  }
}
} // namespace facebook::velox::functions
