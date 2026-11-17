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
#include "velox/functions/sparksql/JavaFloatingPoint.h"

#include <algorithm>
#include <bit>
#include <limits>
#include <vector>

#include <folly/container/F14Set.h>
#include <gmock/gmock.h>
#include <gtest/gtest.h>

namespace facebook::velox::functions::sparksql {
namespace {

class JavaFloatingPointTest : public testing::Test {
 protected:
  template <typename T>
  static T nan() {
    return std::numeric_limits<T>::quiet_NaN();
  }

  // A NaN whose bit pattern differs from the canonical one.
  template <typename T>
  static T otherNaN() {
    return std::bit_cast<T>(
        std::bit_cast<JavaFloatingPoint::Bits<T>>(nan<T>()) | 1);
  }

  template <typename T>
  static T infinity() {
    return std::numeric_limits<T>::infinity();
  }

  template <typename T>
  void testBits() {
    const JavaFloatingPoint::Bits<T> canonicalNaN{
        sizeof(T) == 4 ? 0x7fc0'0000 : 0x7ff8'0000'0000'0000};

    EXPECT_EQ(JavaFloatingPoint::bits(nan<T>()), canonicalNaN);
    EXPECT_EQ(JavaFloatingPoint::bits(otherNaN<T>()), canonicalNaN);
    EXPECT_EQ(JavaFloatingPoint::bits(-nan<T>()), canonicalNaN);
    EXPECT_NE(
        JavaFloatingPoint::bits(T{-0.0}), JavaFloatingPoint::bits(T{0.0}));
    EXPECT_EQ(
        JavaFloatingPoint::bits(T{1.5}),
        std::bit_cast<JavaFloatingPoint::Bits<T>>(T{1.5}));
  }

  template <typename T>
  void testEquals() {
    const JavaFloatingPoint::Equals<T> equals;

    EXPECT_TRUE(equals(nan<T>(), otherNaN<T>()));
    EXPECT_TRUE(equals(nan<T>(), -nan<T>()));
    EXPECT_FALSE(equals(T{-0.0}, T{0.0}));
    EXPECT_TRUE(equals(T{1.5}, T{1.5}));
    EXPECT_FALSE(equals(T{1.5}, nan<T>()));
  }

  template <typename T>
  void testHashSetKeys() {
    folly::
        F14FastSet<T, JavaFloatingPoint::Hash<T>, JavaFloatingPoint::Equals<T>>
            keys;
    for (const T key :
         {T{0.0}, T{-0.0}, nan<T>(), otherNaN<T>(), -nan<T>(), T{0.0}}) {
      keys.insert(key);
    }

    EXPECT_EQ(keys.size(), 3);
  }

  template <typename T>
  void testCompare() {
    const std::vector<T> ordered{
        -infinity<T>(),
        T{-1.0},
        T{-0.0},
        T{0.0},
        T{1.0},
        infinity<T>(),
        nan<T>(),
    };

    auto values = ordered;
    std::reverse(values.begin(), values.end());
    std::sort(values.begin(), values.end(), [](T lhs, T rhs) {
      return JavaFloatingPoint::compare(lhs, rhs) < 0;
    });
    EXPECT_THAT(bitsOf(values), testing::ElementsAreArray(bitsOf(ordered)));

    EXPECT_EQ(JavaFloatingPoint::compare(nan<T>(), otherNaN<T>()), 0);
    EXPECT_EQ(JavaFloatingPoint::compare(nan<T>(), -nan<T>()), 0);
    EXPECT_LT(JavaFloatingPoint::compare(T{-0.0}, T{0.0}), 0);
    EXPECT_GT(JavaFloatingPoint::compare(T{0.0}, T{-0.0}), 0);
    EXPECT_EQ(JavaFloatingPoint::compare(T{1.5}, T{1.5}), 0);
  }

  // Compares by bits so that -0.0 and 0.0 do not match each other.
  template <typename T>
  static std::vector<JavaFloatingPoint::Bits<T>> bitsOf(
      const std::vector<T>& values) {
    std::vector<JavaFloatingPoint::Bits<T>> bits;
    bits.reserve(values.size());
    for (const auto value : values) {
      bits.push_back(JavaFloatingPoint::bits(value));
    }
    return bits;
  }
};

TEST_F(JavaFloatingPointTest, bits) {
  testBits<float>();
  testBits<double>();
}

TEST_F(JavaFloatingPointTest, equals) {
  testEquals<float>();
  testEquals<double>();
}

TEST_F(JavaFloatingPointTest, hashSetKeys) {
  testHashSetKeys<float>();
  testHashSetKeys<double>();
}

TEST_F(JavaFloatingPointTest, compare) {
  testCompare<float>();
  testCompare<double>();
}

} // namespace
} // namespace facebook::velox::functions::sparksql
