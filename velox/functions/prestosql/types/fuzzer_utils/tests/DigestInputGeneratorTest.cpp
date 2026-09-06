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

#include <gtest/gtest.h>

#include "velox/functions/lib/QuantileDigest.h"
#include "velox/functions/lib/TDigest.h"
#include "velox/functions/prestosql/tests/utils/FunctionBaseTest.h"
#include "velox/functions/prestosql/types/QDigestType.h"
#include "velox/functions/prestosql/types/TDigestType.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

namespace facebook::velox::fuzzer::test {

namespace {
constexpr vector_size_t kSize = 100;
constexpr double kNullRatio = 0.1;
} // namespace

// A missing registry entry yields random bytes, not an error.
class DigestInputGeneratorTest : public functions::test::FunctionBaseTest {
 protected:
  // Passes no generator, as ExpressionFuzzer::generateArgColumn does.
  template <typename Deserialize>
  void checkRegistrySuppliedValues(
      const TypePtr& type,
      const Deserialize& deserialize) {
    VectorFuzzer::Options options;
    options.nullRatio = kNullRatio;
    VectorFuzzer fuzzer(options, pool(), 123);

    auto vector = fuzzer.fuzzFlat(type, kSize);
    auto flat = vector->template asFlatVector<StringView>();
    ASSERT_NE(flat, nullptr);

    vector_size_t numNonNull = 0;
    for (auto i = 0; i < flat->size(); ++i) {
      if (flat->isNullAt(i)) {
        continue;
      }
      ++numNonNull;
      deserialize(flat->valueAt(i).data());
    }
    // Guards against passing because every value was null.
    EXPECT_GT(numNonNull, kSize / 2);
  }
};

TEST_F(DigestInputGeneratorTest, tdigest) {
  checkRegistrySuppliedValues(TDIGEST(DOUBLE()), [](const char* serialized) {
    auto digest = functions::TDigest<>::fromSerialized(serialized);
    EXPECT_GE(digest.compression(), 10);
  });
}

TEST_F(DigestInputGeneratorTest, qdigest) {
  const auto checkInnerType = [&](const TypePtr& innerType, auto typeTag) {
    using T = decltype(typeTag);
    checkRegistrySuppliedValues(QDIGEST(innerType), [](const char* serialized) {
      std::allocator<T> allocator;
      functions::qdigest::QuantileDigest<T, std::allocator<T>> digest(
          allocator, serialized);
      EXPECT_GT(digest.serializedByteSize(), 0);
    });
  };

  checkInnerType(DOUBLE(), double{});
  checkInnerType(REAL(), float{});
  checkInnerType(BIGINT(), int64_t{});
}

} // namespace facebook::velox::fuzzer::test
