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

#include "velox/common/memory/Memory.h"
#include "velox/vector/DictionaryVector.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;

namespace {

class VectorFuzzerTest : public testing::Test {
 public:
  memory::MemoryPool* pool() const {
    return pool_.get();
  }

  // Asserts that all arrays in the input vector have exactly `containerSize`
  // elements.
  void assertContainerSize(
      VectorPtr vector,
      size_t vectorSize,
      size_t containerSize) {
    auto* arrayMapBase = vector->as<ArrayVectorBase>();
    ASSERT_EQ(vectorSize, vector->size());

    for (size_t i = 0; i < vector->size(); ++i) {
      EXPECT_EQ(containerSize, arrayMapBase->sizeAt(i));
    }
  }

 private:
  std::unique_ptr<memory::MemoryPool> pool_{
      memory::getDefaultScopedMemoryPool()};
};

// TODO: add coverage for other VectorFuzzer methods.

TEST_F(VectorFuzzerTest, flatPrimitive) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 0.5;
  VectorFuzzer fuzzer(opts, pool());
  VectorPtr vector;

  std::vector<TypePtr> types = {
      TINYINT(),
      BIGINT(),
      DOUBLE(),
      BOOLEAN(),
      VARCHAR(),
      VARBINARY(),
      DATE(),
      TIMESTAMP(),
      INTERVAL_DAY_TIME(),
  };

  for (const auto& type : types) {
    vector = fuzzer.fuzzFlat(type);
    ASSERT_EQ(VectorEncoding::Simple::FLAT, vector->encoding());
    ASSERT_TRUE(vector->type()->kindEquals(type));
    ASSERT_EQ(opts.vectorSize, vector->size());
    ASSERT_TRUE(vector->mayHaveNulls());
  }
}

TEST_F(VectorFuzzerTest, flatComplex) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 0.5;
  VectorFuzzer fuzzer(opts, pool());

  // Arrays.
  auto vector = fuzzer.fuzzFlat(ARRAY(BIGINT()));
  ASSERT_EQ(VectorEncoding::Simple::ARRAY, vector->encoding());
  ASSERT_EQ(opts.vectorSize, vector->size());
  ASSERT_TRUE(vector->mayHaveNulls());

  auto elements = vector->as<ArrayVector>()->elements();
  ASSERT_TRUE(elements->type()->kindEquals(BIGINT()));
  ASSERT_EQ(VectorEncoding::Simple::FLAT, elements->encoding());
  ASSERT_EQ(opts.vectorSize * opts.containerLength, elements->size());

  // Maps.
  vector = fuzzer.fuzzFlat(MAP(BIGINT(), DOUBLE()));
  ASSERT_EQ(VectorEncoding::Simple::MAP, vector->encoding());
  ASSERT_EQ(opts.vectorSize, vector->size());
  ASSERT_TRUE(vector->mayHaveNulls());

  auto mapKeys = vector->as<MapVector>()->mapKeys();
  ASSERT_TRUE(mapKeys->type()->kindEquals(BIGINT()));
  ASSERT_EQ(VectorEncoding::Simple::FLAT, mapKeys->encoding());
  ASSERT_EQ(opts.vectorSize * opts.containerLength, mapKeys->size());

  auto mapValues = vector->as<MapVector>()->mapValues();
  ASSERT_TRUE(mapValues->type()->kindEquals(DOUBLE()));
  ASSERT_EQ(VectorEncoding::Simple::FLAT, mapValues->encoding());
  ASSERT_EQ(opts.vectorSize * opts.containerLength, mapValues->size());
}

TEST_F(VectorFuzzerTest, flatNotNull) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 0;
  VectorFuzzer fuzzer(opts, pool());

  auto vector = fuzzer.fuzzFlat(BIGINT());
  ASSERT_FALSE(vector->mayHaveNulls());

  vector = fuzzer.fuzzFlat(ARRAY(BIGINT()));
  ASSERT_FALSE(vector->mayHaveNulls());

  vector = fuzzer.fuzzFlat(MAP(BIGINT(), INTEGER()));
  ASSERT_FALSE(vector->mayHaveNulls());
}

TEST_F(VectorFuzzerTest, dictionary) {
  VectorFuzzer::Options opts;
  VectorFuzzer fuzzer(opts, pool());

  // Generates a flat inner vector without nuls.
  const size_t innerSize = 100;
  auto inner = fuzzer.fuzzFlat(REAL(), innerSize);

  opts.nullRatio = 0.5;
  fuzzer.setOptions(opts);

  // Generate a dictionary with the same size as the inner vector being wrapped.
  auto vector = fuzzer.fuzzDictionary(inner);
  ASSERT_EQ(VectorEncoding::Simple::DICTIONARY, vector->encoding());
  ASSERT_TRUE(vector->mayHaveNulls());
  ASSERT_TRUE(vector->valueVector()->type()->kindEquals(REAL()));
  ASSERT_EQ(innerSize, vector->size());
  ASSERT_EQ(innerSize, vector->valueVector()->size());

  // Generate a dictionary with less elements.
  vector = fuzzer.fuzzDictionary(inner, 10);
  ASSERT_EQ(VectorEncoding::Simple::DICTIONARY, vector->encoding());
  ASSERT_TRUE(vector->mayHaveNulls());
  ASSERT_TRUE(vector->valueVector()->type()->kindEquals(REAL()));
  ASSERT_EQ(10, vector->size());
  ASSERT_EQ(innerSize, vector->valueVector()->size());

  // Generate a dictionary with more elements.
  vector = fuzzer.fuzzDictionary(inner, 1000);
  ASSERT_EQ(VectorEncoding::Simple::DICTIONARY, vector->encoding());
  ASSERT_TRUE(vector->mayHaveNulls());
  ASSERT_TRUE(vector->valueVector()->type()->kindEquals(REAL()));
  ASSERT_EQ(1000, vector->size());
  ASSERT_EQ(innerSize, vector->valueVector()->size());

  // Generate a dictionary without nulls.
  opts.dictionaryHasNulls = false;
  fuzzer.setOptions(opts);
  vector = fuzzer.fuzzDictionary(inner);
  ASSERT_FALSE(vector->mayHaveNulls());
}

TEST_F(VectorFuzzerTest, constants) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 0;
  VectorFuzzer fuzzer(opts, pool());

  // Non-null primitive constants.
  auto vector = fuzzer.fuzzConstant(INTEGER());
  ASSERT_TRUE(vector->type()->kindEquals(INTEGER()));
  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, vector->encoding());
  ASSERT_FALSE(vector->mayHaveNulls());

  vector = fuzzer.fuzzConstant(VARCHAR());
  ASSERT_TRUE(vector->type()->kindEquals(VARCHAR()));
  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, vector->encoding());
  ASSERT_FALSE(vector->mayHaveNulls());

  // Non-null complex types.
  vector = fuzzer.fuzzConstant(MAP(BIGINT(), SMALLINT()));
  ASSERT_TRUE(vector->type()->kindEquals(MAP(BIGINT(), SMALLINT())));
  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, vector->encoding());
  ASSERT_FALSE(vector->mayHaveNulls());

  vector = fuzzer.fuzzConstant(ROW({ARRAY(BIGINT()), SMALLINT()}));
  ASSERT_TRUE(vector->type()->kindEquals(ROW({ARRAY(BIGINT()), SMALLINT()})));
  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, vector->encoding());
  ASSERT_FALSE(vector->mayHaveNulls());
}

TEST_F(VectorFuzzerTest, constantsNull) {
  VectorFuzzer::Options opts;
  opts.nullRatio = 1; // 1 = 100%
  VectorFuzzer fuzzer(opts, pool());

  // Null constants.
  auto vector = fuzzer.fuzzConstant(REAL());
  ASSERT_TRUE(vector->type()->kindEquals(REAL()));
  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, vector->encoding());
  ASSERT_TRUE(vector->mayHaveNulls());

  // Null complex types.
  vector = fuzzer.fuzzConstant(ARRAY(VARCHAR()));
  ASSERT_TRUE(vector->type()->kindEquals(ARRAY(VARCHAR())));
  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, vector->encoding());
  ASSERT_TRUE(vector->mayHaveNulls());
}

TEST_F(VectorFuzzerTest, array) {
  VectorFuzzer::Options opts;
  VectorFuzzer fuzzer(opts, pool());

  // 1 elements per array.
  auto vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), 100);
  ASSERT_TRUE(vector->type()->kindEquals(ARRAY(REAL())));
  assertContainerSize(vector, 100, 1);

  // 10 elements per array.
  vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), 10);
  assertContainerSize(vector, 10, 10);

  // 3 elements per array.
  vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), 33);
  assertContainerSize(vector, 33, 3);

  // 100 elements per array.
  vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), 1);
  assertContainerSize(vector, 1, 100);

  // More array rows than elements.
  vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), 1000);

  auto* arrayVector = vector->as<ArrayVector>();
  ASSERT_EQ(vector->size(), 1000);

  // Check that the first 100 arrays have 1 element, and the remaining have 0.
  for (size_t i = 0; i < 100; ++i) {
    EXPECT_EQ(1, arrayVector->sizeAt(i));
  }
  for (size_t i = 100; i < 1000; ++i) {
    EXPECT_EQ(0, arrayVector->sizeAt(i));
  }

  // Variable number of array elements - just ensure we don't exceed the number
  // of underlying elements.
  opts.containerVariableLength = true;
  fuzzer.setOptions(opts);

  size_t arraySize = 100;

  vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), arraySize);
  ASSERT_EQ(arraySize, vector->size());
  ASSERT_GE(
      100, vector->offsetAt(arraySize - 1) + vector->sizeAt(arraySize - 1));

  arraySize = 33;

  vector = fuzzer.fuzzArray(fuzzer.fuzzFlat(REAL(), 100), arraySize);
  ASSERT_EQ(arraySize, vector->size());
  ASSERT_GE(
      100, vector->offsetAt(arraySize - 1) + vector->sizeAt(arraySize - 1));
}

TEST_F(VectorFuzzerTest, map) {
  VectorFuzzer::Options opts;
  VectorFuzzer fuzzer(opts, pool());

  // 1 elements per array.
  auto vector = fuzzer.fuzzMap(
      fuzzer.fuzzFlat(REAL(), 100), fuzzer.fuzzFlat(BIGINT(), 100), 100);
  ASSERT_TRUE(vector->type()->kindEquals(MAP(REAL(), BIGINT())));
  assertContainerSize(vector, 100, 1);

  // 10 elements per array.
  vector = fuzzer.fuzzMap(
      fuzzer.fuzzFlat(REAL(), 100), fuzzer.fuzzFlat(INTEGER(), 100), 10);
  assertContainerSize(vector, 10, 10);
}

TEST_F(VectorFuzzerTest, assorted) {
  VectorFuzzer::Options opts;
  VectorFuzzer fuzzer(opts, pool());

  auto vector = fuzzer.fuzzMap(
      fuzzer.fuzzDictionary(fuzzer.fuzzFlat(INTEGER(), 10), 100),
      fuzzer.fuzzArray(fuzzer.fuzzConstant(DOUBLE(), 40), 100),
      10);
  ASSERT_TRUE(vector->type()->kindEquals(MAP(INTEGER(), ARRAY(DOUBLE()))));

  // Cast map.
  ASSERT_EQ(VectorEncoding::Simple::MAP, vector->encoding());
  auto map = vector->as<MapVector>();

  // Cast map key.
  ASSERT_EQ(VectorEncoding::Simple::DICTIONARY, map->mapKeys()->encoding());
  auto key =
      map->mapKeys()
          ->as<DictionaryVector<TypeTraits<TypeKind::INTEGER>::NativeType>>();
  ASSERT_EQ(VectorEncoding::Simple::FLAT, key->valueVector()->encoding());

  // Cast map value.
  ASSERT_EQ(VectorEncoding::Simple::ARRAY, map->mapValues()->encoding());
  auto value = map->mapValues()->as<ArrayVector>();

  ASSERT_EQ(VectorEncoding::Simple::CONSTANT, value->elements()->encoding());
}

} // namespace
