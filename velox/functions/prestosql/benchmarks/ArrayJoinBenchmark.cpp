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

#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/common/memory/Memory.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/functions/prestosql/types/JsonRegistration.h"
#include "velox/functions/prestosql/types/JsonType.h"

using namespace facebook::velox;

namespace {

constexpr vector_size_t kNumRows = 1'000;

// Strings of this length are stored inline in the StringView, so joining them
// never touches a string buffer.
constexpr size_t kInlineElementLength = 8;

// Strings of this length exceed the inline capacity of a StringView and live
// in a string buffer, which is what a no-copy result can reference.
constexpr size_t kBufferedElementLength = 32;

// Builds an array vector of 'kNumRows' rows, each holding 'arrayLength'
// strings of 'elementLength' characters. JSON elements are quoted so that
// they are valid JSON scalars.
ArrayVectorPtr makeStringArrays(
    test::VectorMaker& vectorMaker,
    vector_size_t arrayLength,
    size_t elementLength,
    const TypePtr& elementType) {
  const bool quoteElements = isJsonType(elementType);
  return vectorMaker.arrayVector<std::string>(
      kNumRows,
      [arrayLength](vector_size_t /*row*/) { return arrayLength; },
      [elementLength, quoteElements](vector_size_t row, vector_size_t index) {
        const std::string element(
            elementLength, static_cast<char>('a' + (row + index) % 26));
        return quoteElements ? "\"" + element + "\"" : element;
      },
      nullptr,
      ARRAY(elementType));
}

// Builds an ARRAY(VARCHAR) vector whose odd-numbered elements are null, so
// that the null replacement argument is actually exercised.
ArrayVectorPtr makeVarcharArraysWithNulls(
    test::VectorMaker& vectorMaker,
    vector_size_t arrayLength) {
  return vectorMaker.arrayVector<std::string>(
      kNumRows,
      [arrayLength](vector_size_t /*row*/) { return arrayLength; },
      [](vector_size_t index) {
        return std::string(
            kBufferedElementLength, static_cast<char>('a' + index % 26));
      },
      nullptr,
      [](vector_size_t index) { return index % 2 == 1; });
}

ArrayVectorPtr makeBigintArrays(
    test::VectorMaker& vectorMaker,
    vector_size_t arrayLength) {
  return vectorMaker.arrayVector<int64_t>(
      kNumRows,
      [arrayLength](vector_size_t /*row*/) { return arrayLength; },
      [](vector_size_t row, vector_size_t index) {
        return static_cast<int64_t>(row) * 31 + index;
      });
}

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::testingSetInstance(memory::MemoryManager::Options{});

  ExpressionBenchmarkBuilder benchmarkBuilder;
  functions::prestosql::registerAllScalarFunctions();
  registerJsonType();

  auto& vectorMaker = benchmarkBuilder.vectorMaker();

  // Varchar elements that live in a string buffer. A one-element array joins
  // to the element itself, so it can be returned without copying; the longer
  // arrays are controls that no single-element fast path can affect.
  benchmarkBuilder
      .addBenchmarkSet(
          "array_join_varchar",
          vectorMaker.rowVector({
              makeStringArrays(
                  vectorMaker, 1, kBufferedElementLength, VARCHAR()),
              makeStringArrays(
                  vectorMaker, 2, kBufferedElementLength, VARCHAR()),
              makeStringArrays(
                  vectorMaker, 4, kBufferedElementLength, VARCHAR()),
              makeStringArrays(
                  vectorMaker, 32, kBufferedElementLength, VARCHAR()),
          }))
      .addExpression("1_element", "array_join(c0, '_')")
      .addExpression("2_elements", "array_join(c1, '_')")
      .addExpression("4_elements", "array_join(c2, '_')")
      .addExpression("32_elements", "array_join(c3, '_')")
      .disableTesting();

  // The same shape with inlined elements, where returning the element without
  // copying saves the conversion but no buffer reference.
  benchmarkBuilder
      .addBenchmarkSet(
          "array_join_varchar_inlined",
          vectorMaker.rowVector({
              makeStringArrays(vectorMaker, 1, kInlineElementLength, VARCHAR()),
              makeStringArrays(vectorMaker, 2, kInlineElementLength, VARCHAR()),
              makeStringArrays(
                  vectorMaker, 32, kInlineElementLength, VARCHAR()),
          }))
      .addExpression("1_element", "array_join(c0, '_')")
      .addExpression("2_elements", "array_join(c1, '_')")
      .addExpression("32_elements", "array_join(c2, '_')")
      .disableTesting();

  // JSON elements are unescaped into a new string, so they can never be
  // returned without copying however short the array is.
  benchmarkBuilder
      .addBenchmarkSet(
          "array_join_json",
          vectorMaker.rowVector({
              makeStringArrays(vectorMaker, 1, kBufferedElementLength, JSON()),
              makeStringArrays(vectorMaker, 4, kBufferedElementLength, JSON()),
          }))
      .addExpression("1_element", "array_join(c0, '_')")
      .addExpression("4_elements", "array_join(c1, '_')")
      .disableTesting();

  // Non-string elements, which are always converted into a new string.
  benchmarkBuilder
      .addBenchmarkSet(
          "array_join_bigint",
          vectorMaker.rowVector({
              makeBigintArrays(vectorMaker, 1),
              makeBigintArrays(vectorMaker, 32),
          }))
      .addExpression("1_element", "array_join(c0, '_')")
      .addExpression("32_elements", "array_join(c1, '_')")
      .disableTesting();

  // The three-argument form against the two-argument one over identical
  // input, isolating the cost of supplying a null replacement.
  benchmarkBuilder
      .addBenchmarkSet(
          "array_join_null_replacement",
          vectorMaker.rowVector({
              makeVarcharArraysWithNulls(vectorMaker, 4),
          }))
      .addExpression("without_replacement", "array_join(c0, '_')")
      .addExpression(
          "with_replacement",
          "array_join(c0, '_', 'a_null_replacement_longer_than_inline')")
      .disableTesting();

  benchmarkBuilder.registerBenchmarks();

  folly::runBenchmarks();
  return 0;
}
