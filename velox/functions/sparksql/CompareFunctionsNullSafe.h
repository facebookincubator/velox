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

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions::sparksql {

inline std::vector<std::shared_ptr<exec::FunctionSignature>>
equalNullSafeSignatures() {
  return {exec::FunctionSignatureBuilder()
              .typeVariable("T")
              .returnType("boolean")
              .argumentType("T")
              .argumentType("T")
              .build()};
}

std::shared_ptr<exec::VectorFunction> makeEqualNullSafe(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs);

} // namespace facebook::velox::functions::sparksql
