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
#include "velox/functions/lib/Re2Functions.h"

using namespace facebook;
using namespace facebook::velox;
using namespace facebook::velox::functions;
using namespace facebook::velox::functions::test;
using namespace facebook::velox::memory;
using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);

  exec::registerStatefulVectorFunction("like", likeSignatures(), makeLike);
  ExpressionBenchmarkBuilder benchmarkBuilder;
  const vector_size_t vectorSize = 1000;
  auto vectorMaker = benchmarkBuilder.vectorMaker();

  auto substringInput =
      vectorMaker.flatVector<facebook::velox::StringView>({""});
  auto prefixInput = vectorMaker.flatVector<facebook::velox::StringView>({""});
  auto suffixInput = vectorMaker.flatVector<facebook::velox::StringView>({""});
  auto genericInput = vectorMaker.flatVector<facebook::velox::StringView>({""});

  substringInput->resize(vectorSize);
  prefixInput->resize(vectorSize);
  suffixInput->resize(vectorSize);
  genericInput->resize(vectorSize);

  // Prepare data which contains/prefix with/suffix with the string 'a_b_c'
  for (int i = 0; i < vectorSize; i++) {
    substringInput->set(
        i, StringView::makeInline(fmt::format("{}a_b_c{}", i, i)));
    prefixInput->set(i, StringView::makeInline(fmt::format("a_b_c{}", i)));
    suffixInput->set(i, StringView::makeInline(fmt::format("{}a_b_c", i)));
    genericInput->set(
        i, StringView::makeInline(fmt::format("{}a_b_c{}", i, i)));
  }

  benchmarkBuilder
      .addBenchmarkSet(
          "like",
          vectorMaker.rowVector(
              {"col0", "col1", "col2", "col3"},
              {substringInput, prefixInput, suffixInput, genericInput}))
      .addExpression("like_substring", R"(like (col0, '%a\_b\_c%', '\'))")
      .addExpression("like_prefix", R"(like (col1, 'a\_b\_c%', '\'))")
      .addExpression("like_suffix", R"(like (col2, '%a\_b\_c', '\'))")
      .addExpression("like_generic", R"(like (col3, '%a%b%c'))")
      .withIterations(100)
      .disableTesting();

  benchmarkBuilder.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
