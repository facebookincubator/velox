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

#include <cstdint>
#include <limits>
#include <optional>

#include <fmt/format.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/functions/sparksql/tests/SparkFunctionBaseTest.h"

namespace facebook::velox::functions::sparksql::test {
namespace {

class IntervalArithmeticTest : public SparkFunctionBaseTest {
 protected:
  void setAnsiEnabled(bool enabled) {
    queryCtx_->testingOverrideConfigUnsafe(
        {{SparkQueryConfig::qualify(SparkQueryConfig::kAnsiEnabled),
          enabled ? "true" : "false"}});
  }

  template <typename T>
  std::optional<T> unaryminus(const TypePtr& type, std::optional<T> value) {
    return evaluateOnce<T>("unaryminus(c0)", type, value);
  }

  template <typename T>
  std::optional<T>
  add(const TypePtr& type, std::optional<T> left, std::optional<T> right) {
    return evaluateOnce<T>("add(c0, c1)", {type, type}, left, right);
  }

  template <typename T>
  std::optional<T>
  subtract(const TypePtr& type, std::optional<T> left, std::optional<T> right) {
    return evaluateOnce<T>("subtract(c0, c1)", {type, type}, left, right);
  }

  template <typename T>
  std::optional<T> checkedAdd(
      const TypePtr& type,
      std::optional<T> left,
      std::optional<T> right) {
    return evaluateOnce<T>("checked_add(c0, c1)", {type, type}, left, right);
  }

  template <typename T>
  std::optional<T> checkedSubtract(
      const TypePtr& type,
      std::optional<T> left,
      std::optional<T> right) {
    return evaluateOnce<T>(
        "checked_subtract(c0, c1)", {type, type}, left, right);
  }

  template <typename T>
  std::optional<T> tryCheckedAdd(
      const TypePtr& type,
      std::optional<T> left,
      std::optional<T> right) {
    return evaluateOnce<T>(
        "try(checked_add(c0, c1))", {type, type}, left, right);
  }

  template <typename T>
  std::optional<T> tryCheckedSubtract(
      const TypePtr& type,
      std::optional<T> left,
      std::optional<T> right) {
    return evaluateOnce<T>(
        "try(checked_subtract(c0, c1))", {type, type}, left, right);
  }

  template <typename T>
  void testUnaryMinus(const TypePtr& type, T value) {
    for (const bool ansiEnabled : {false, true}) {
      SCOPED_TRACE(fmt::format("ANSI enabled: {}", ansiEnabled));
      setAnsiEnabled(ansiEnabled);

      EXPECT_EQ(unaryminus<T>(type, value), -value);
      EXPECT_EQ(unaryminus<T>(type, -value), value);
      EXPECT_EQ(unaryminus<T>(type, 0), 0);
      VELOX_ASSERT_THROW(
          unaryminus<T>(type, std::numeric_limits<T>::min()),
          "Arithmetic overflow");
    }
  }

  template <typename T>
  void testAdd(const TypePtr& type) {
    for (const bool ansiEnabled : {false, true}) {
      SCOPED_TRACE(fmt::format("ANSI enabled: {}", ansiEnabled));
      setAnsiEnabled(ansiEnabled);

      EXPECT_EQ(add<T>(type, 1, 2), 3);
      EXPECT_EQ(add<T>(type, -2, 1), -1);
      EXPECT_EQ(checkedAdd<T>(type, 1, 2), 3);
      VELOX_ASSERT_THROW(
          add<T>(type, std::numeric_limits<T>::max(), 1),
          "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          add<T>(type, std::numeric_limits<T>::min(), -1),
          "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          checkedAdd<T>(type, std::numeric_limits<T>::max(), 1),
          "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          checkedAdd<T>(type, std::numeric_limits<T>::min(), -1),
          "Arithmetic overflow");
      EXPECT_EQ(
          tryCheckedAdd<T>(type, std::numeric_limits<T>::max(), 1),
          std::nullopt);
    }
  }

  template <typename T>
  void testSubtract(const TypePtr& type) {
    for (const bool ansiEnabled : {false, true}) {
      SCOPED_TRACE(fmt::format("ANSI enabled: {}", ansiEnabled));
      setAnsiEnabled(ansiEnabled);

      EXPECT_EQ(subtract<T>(type, 3, 1), 2);
      EXPECT_EQ(subtract<T>(type, 1, 2), -1);
      EXPECT_EQ(checkedSubtract<T>(type, 3, 1), 2);
      VELOX_ASSERT_THROW(
          subtract<T>(type, std::numeric_limits<T>::min(), 1),
          "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          subtract<T>(type, std::numeric_limits<T>::max(), -1),
          "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          checkedSubtract<T>(type, std::numeric_limits<T>::min(), 1),
          "Arithmetic overflow");
      VELOX_ASSERT_THROW(
          checkedSubtract<T>(type, std::numeric_limits<T>::max(), -1),
          "Arithmetic overflow");
      EXPECT_EQ(
          tryCheckedSubtract<T>(type, std::numeric_limits<T>::min(), 1),
          std::nullopt);
    }
  }
};

TEST_F(IntervalArithmeticTest, unaryMinus) {
  testUnaryMinus<int64_t>(INTERVAL_DAY_TIME(), 1'000);
  testUnaryMinus<int32_t>(INTERVAL_YEAR_MONTH(), 12);
}

TEST_F(IntervalArithmeticTest, add) {
  testAdd<int64_t>(INTERVAL_DAY_TIME());
  testAdd<int32_t>(INTERVAL_YEAR_MONTH());
}

TEST_F(IntervalArithmeticTest, subtract) {
  testSubtract<int64_t>(INTERVAL_DAY_TIME());
  testSubtract<int32_t>(INTERVAL_YEAR_MONTH());
}

} // namespace
} // namespace facebook::velox::functions::sparksql::test
