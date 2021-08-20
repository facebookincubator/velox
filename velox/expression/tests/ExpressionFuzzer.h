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

#include "velox/type/Type.h"

namespace facebook::velox::test {

// Represents one available function signature.
struct CallableSignature {
  // Function name.
  std::string name;

  // Input arguments and return type.
  std::vector<TypePtr> args;
  TypePtr returnType;

  // Convenience print function.
  std::string toString() const;
};

// Generates random expressions based on `signatures` and random input data (via
// VectorFuzzer). Generates `steps` distinct expressions.
void expressionFuzzer(
    std::vector<CallableSignature> signatures,
    size_t steps,
    size_t seed);

} // namespace facebook::velox::test
