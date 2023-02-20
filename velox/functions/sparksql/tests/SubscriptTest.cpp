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

#include <optional>

#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class SubscriptTest : public SparkFunctionBaseTest {
 protected:
  template <typename T = int64_t>
  std::optional<T> subscriptSimple(
      const std::string& expression,
      const std::vector<VectorPtr>& parameters) {
    auto result =
        evaluate<SimpleVector<T>>(expression, makeRowVector(parameters));
    if (result->size() != 1) {
      throw std::invalid_argument(
          "subscriptSimple expects a single output row.");
    }
    if (result->isNullAt(0)) {
      return std::nullopt;
    }
    return result->valueAt(0);
  }
};

} // namespace
} // namespace facebook::velox::functions::sparksql::test
