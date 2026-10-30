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
#include <string_view>

#include "velox/benchmarks/ExpressionBenchmarkBuilder.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox;

namespace {

// Compares constant and per-row standard charsets in both error-action modes.
class EncodeBenchmark : public ExpressionBenchmarkBuilder {
 public:
  void addBenchmarks(bool replaceUnmappable) {
    queryCtx_->testingOverrideConfigUnsafe({
        {functions::sparksql::SparkQueryConfig::qualify(
             functions::sparksql::SparkQueryConfig::kLegacyCodingErrorAction),
         replaceUnmappable ? "true" : "false"},
    });
    auto& maker = vectorMaker();
    constexpr vector_size_t kNumRows = 1'024;
    for (const auto* charset :
         {"UTF-8",
          "US-ASCII",
          "ISO-8859-1",
          "UTF-16BE",
          "UTF-16LE",
          "UTF-16",
          "UTF-32"}) {
      const auto addWorkload = [&](const std::string& name,
                                   const std::vector<std::string>& values) {
        auto input = maker.rowVector(
            {"input", "charset"},
            {maker.flatVector<std::string>(
                 kNumRows,
                 [&values](auto row) { return values[row % values.size()]; }),
             maker.flatVector<std::string>(
                 kNumRows, [charset](auto /*row*/) { return charset; })});
        addBenchmarkSet(
            fmt::format(
                "{}_{}_{}",
                replaceUnmappable ? "replacement" : "strict",
                charset,
                name),
            input)
            .addExpression(
                "constant_charset", fmt::format("encode(input, '{}')", charset))
            .addExpression("per_row_charset", "encode(input, charset)")
            .withIterations(100);
      };
      for (const size_t length : {1, 8, 32, 256}) {
        addWorkload(
            fmt::format("ascii_{}bytes", length),
            {std::string(length, 'A'), std::string(length, 'B')});
      }
      const std::string_view charsetName{charset};
      const bool singleByte =
          charsetName == "US-ASCII" || charsetName == "ISO-8859-1";
      if (charsetName == "ISO-8859-1") {
        addWorkload("non_ascii", {"\xC3\xA9", "A\xC2\xA5"});
      } else if (!singleByte) {
        addWorkload("non_ascii", {"\xE2\x82\xAC", "A\xF0\x9F\x98\x80"});
      }
      if (replaceUnmappable || !singleByte) {
        addWorkload("malformed_utf8", {"A\xFF", "\xC3"});
      }
      if (replaceUnmappable && singleByte) {
        addWorkload("unmappable", {"A\xF0\x9F\x98\x80", "\xE2\x80\x8D"});
      }
      addWorkload("empty", {""});
    }
  }
};

} // namespace

int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  EncodeBenchmark strict;
  EncodeBenchmark replacement;
  functions::sparksql::registerFunctions("");
  strict.addBenchmarks(false);
  replacement.addBenchmarks(true);
  strict.testBenchmarks();
  replacement.testBenchmarks();
  strict.registerBenchmarks();
  replacement.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
