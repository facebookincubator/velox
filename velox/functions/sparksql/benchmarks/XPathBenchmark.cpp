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

#include <string>
#include <string_view>

#include <fmt/format.h>
#include <folly/Benchmark.h>
#include <folly/init/Init.h>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox;

int main(int argc, char** argv) {
  folly::Init init(&argc, &argv);
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});
  functions::sparksql::registerFunctions("");

  ExpressionBenchmarkBuilder builder;
  const auto input = builder.vectorMaker().rowVector(
      {builder.vectorMaker().flatVector<StringView>(1'000, [](vector_size_t) {
        return StringView(
            "<a xmlns:p='urn:p'><b>3.97</b>"
            "<b>1.00000000000000011102230246251565404236316680908203126</b>"
            "</a>");
      })});
  const auto addPair = [&](const std::string& name,
                           std::string_view cached,
                           std::string_view parsed) {
    builder.addBenchmarkSet(name, input)
        .addExpression(
            "parse_per_row", fmt::format("xpath_string(c0, '{}')", parsed))
        .addExpression(
            "cached_literal", fmt::format("xpath_string(c0, '{}')", cached))
        .withIterations(100);
  };
  for (const auto& decimal :
       {std::string("3.97"),
        std::string(
            "1.00000000000000011102230246251565404236316680908203126")}) {
    const auto parsed = fmt::format("number(\"{}\")", decimal);
    addPair(fmt::format("literal_{}", decimal), decimal, parsed);
    addPair(
        fmt::format("predicate_{}", decimal),
        fmt::format("a/b[. = {}]", decimal),
        fmt::format("a/b[. = {}]", parsed));
    addPair(
        fmt::format("namespace_predicate_{}", decimal),
        fmt::format("a/namespace::*[{} = {}]", decimal, decimal),
        fmt::format("a/namespace::*[{} = {}]", parsed, parsed));
  }
  addPair("position", "a/b[1]", "a/b[number(\"1\")]");
  builder.registerBenchmarks();
  builder.testBenchmarks();
  folly::runBenchmarks();
  return 0;
}
