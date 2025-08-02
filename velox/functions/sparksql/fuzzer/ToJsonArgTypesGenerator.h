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

#include "velox/expression/fuzzer/ArgTypesGenerator.h"

namespace facebook::velox::functions::sparksql::fuzzer {

class ToJsonArgTypesGenerator : public velox::fuzzer::ArgTypesGenerator {
 public:
  std::vector<TypePtr> generateArgs(
      const exec::FunctionSignature& signature,
      const TypePtr& returnType,
      FuzzerGenerator& rng) override {
    uint32_t choice = static_cast<uint32_t>(rng()) % 3;
    switch (choice) {
      case 0:
        return {ROW({"a", "b"}, {INTEGER(), VARCHAR()})};
      case 1:
        return {ARRAY(DOUBLE())};
      case 2:
        return {MAP(INTEGER(), VARCHAR())};
      default:
        VELOX_UNREACHABLE();
    }
  }
};

} // namespace facebook::velox::functions::sparksql::fuzzer
