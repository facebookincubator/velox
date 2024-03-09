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

#include <folly/String.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "velox/expression/tests/ExpressionFuzzerVerifier.h"
#include "velox/functions/FunctionRegistry.h"

namespace facebook::velox::test {

/// FuzzerRunner leverages ExpressionFuzzerVerifier to create a gtest unit test.
class FuzzerRunner {
 public:
  static int run(
      size_t seed,
      const std::unordered_set<std::string>& skipFunctions,
      const std::unordered_map<std::string, std::string>& queryConfigs);

  static void runFromGtest(
      size_t seed,
      const std::unordered_set<std::string>& skipFunctions,
      const std::unordered_map<std::string, std::string>& queryConfigs);
};

} // namespace facebook::velox::test
