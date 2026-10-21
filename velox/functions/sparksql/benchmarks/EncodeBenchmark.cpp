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
#include "velox/functions/Registerer.h"
#include "velox/functions/sparksql/EncodeFunction.h"
#include "velox/functions/sparksql/SparkQueryConfig.h"
#include "velox/functions/sparksql/registration/Register.h"

using namespace facebook::velox;

namespace {

// Resolve the constant charset once, but repeat converter setup for every row.
template <typename T>
struct UncachedEncodeFunction {
  VELOX_DEFINE_FUNCTION_TYPES(T);

  void initialize(
      const std::vector<TypePtr>& /*inputTypes*/,
      const core::QueryConfig& config,
      const arg_type<Varchar>* /*input*/,
      const arg_type<Varchar>* charset) {
    VELOX_CHECK_NOT_NULL(charset);
    const functions::sparksql::SparkQueryConfig sparkConfig{config};
    charsetType_ = functions::sparksql::detail::resolveCharset(
        *charset, sparkConfig.legacyJavaCharsets());
    VELOX_CHECK(
        charsetType_ == functions::sparksql::detail::CharsetType::kLegacy);
    replaceUnmappable_ = sparkConfig.legacyCodingErrorAction();
  }

  Status call(
      out_type<Varbinary>& result,
      const arg_type<Varchar>& input,
      const arg_type<Varchar>& charset) {
    return functions::sparksql::detail::encode(
        result, input, charset, charsetType_, replaceUnmappable_);
  }

 private:
  functions::sparksql::detail::CharsetType charsetType_{
      functions::sparksql::detail::CharsetType::kUnsupported};
  bool replaceUnmappable_{false};
};

// Keep both encoding options in one query-config override.
class EncodeBenchmark : public ExpressionBenchmarkBuilder {
 public:
  void addBenchmarks(bool replaceUnmappable) {
    queryCtx_->testingOverrideConfigUnsafe({
        {functions::sparksql::SparkQueryConfig::qualify(
             functions::sparksql::SparkQueryConfig::kLegacyJavaCharsets),
         "true"},
        {functions::sparksql::SparkQueryConfig::qualify(
             functions::sparksql::SparkQueryConfig::kLegacyCodingErrorAction),
         replaceUnmappable ? "true" : "false"},
    });
    auto& maker = vectorMaker();
    constexpr vector_size_t kNumRows = 1'024;
    for (const auto* charset : {"windows-1252", "windows-31j", "ISO-2022-JP"}) {
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
                "uncached_pre_resolved",
                fmt::format("encode_uncached(input, '{}')", charset))
            .addExpression(
                "cached_constant", fmt::format("encode(input, '{}')", charset))
            .addExpression("per_row_charset", "encode(input, charset)")
            .withIterations(100);
      };
      for (const size_t length : {1, 8, 32}) {
        addWorkload(
            fmt::format("ascii_{}bytes", length),
            {std::string(length, 'A'), std::string(length, 'B')});
      }
      if (std::string_view(charset) == "windows-1252") {
        addWorkload("non_ascii", {"\xE2\x82\xAC", "A\xC2\xA5"});
      } else if (std::string_view(charset) == "windows-31j") {
        addWorkload("non_ascii", {"\xC2\xA5\xE2\x80\xBE", "\xE3\x81\x82"});
      } else {
        addWorkload("non_ascii", {"\xE3\x81\x82", "A\xE3\x81\x84"});
      }
      if (replaceUnmappable) {
        addWorkload("unmappable", {"A\xF0\x9F\x98\x80", "\xE2\x80\x8D"});
      }
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
  registerFunction<UncachedEncodeFunction, Varbinary, Varchar, Varchar>(
      {"encode_uncached"});
  strict.addBenchmarks(false);
  replacement.addBenchmarks(true);
  strict.testBenchmarks();
  replacement.testBenchmarks();
  strict.registerBenchmarks();
  replacement.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
