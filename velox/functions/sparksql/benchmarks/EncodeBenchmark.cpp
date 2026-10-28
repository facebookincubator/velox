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

// Compares constant and dynamic charsets across conversion and zero-copy paths.
class EncodeBenchmark : public ExpressionBenchmarkBuilder {
 public:
  void addBenchmarks(bool replaceUnmappable, bool legacyCharsets) {
    queryCtx_->testingOverrideConfigUnsafe({
        {functions::sparksql::SparkQueryConfig::qualify(
             functions::sparksql::SparkQueryConfig::kLegacyCodingErrorAction),
         replaceUnmappable ? "true" : "false"},
        {functions::sparksql::SparkQueryConfig::qualify(
             functions::sparksql::SparkQueryConfig::kLegacyJavaCharsets),
         legacyCharsets ? "true" : "false"},
    });
    auto& maker = vectorMaker();
    constexpr vector_size_t kNumRows = 1'024;
    const std::vector<const char*> charsets = legacyCharsets
        ? std::vector<const char*>{"IBM037",        "IBM437",
                                   "ISO-8859-2",    "ISO-8859-3",
                                   "ISO-8859-4",    "ISO-8859-5",
                                   "ISO-8859-6",    "ISO-8859-7",
                                   "ISO-8859-8",    "ISO-8859-9",
                                   "x-iso-8859-11", "ISO-8859-13",
                                   "ISO-8859-15",   "ISO-8859-16",
                                   "KOI8-R",        "KOI8-U",
                                   "windows-1250",  "windows-1251",
                                   "windows-1252",  "windows-1253",
                                   "windows-1254",  "windows-1255",
                                   "windows-1256",  "windows-1257",
                                   "windows-1258"}
        : std::vector<const char*>{
              "UTF-8",
              "US-ASCII",
              "ISO-8859-1",
              "UTF-16BE",
              "UTF-16LE",
              "UTF-16",
              "UTF-32"};
    for (const auto* charset : charsets) {
      const auto addWorkload = [&](const std::string& name,
                                   const std::vector<std::string>& values) {
        auto input = maker.rowVector(
            {"input", "charset", "changing_charset"},
            {maker.flatVector<std::string>(
                 kNumRows,
                 [&values](auto row) { return values[row % values.size()]; }),
             maker.flatVector<std::string>(
                 kNumRows, [charset](auto /*row*/) { return charset; }),
             maker.flatVector<std::string>(kNumRows, [charset](auto row) {
               std::string name{charset};
               if (row % 2 != 0) {
                 for (auto& character : name) {
                   if (character >= 'a' && character <= 'z') {
                     character -= 'a' - 'A';
                   } else if (character >= 'A' && character <= 'Z') {
                     character += 'a' - 'A';
                   }
                 }
               }
               return name;
             })});
        addBenchmarkSet(
            fmt::format(
                "{}{}_{}_{}",
                legacyCharsets ? "legacy_" : "",
                replaceUnmappable ? "replacement" : "strict",
                charset,
                name),
            input)
            .addExpression(
                "constant_charset", fmt::format("encode(input, '{}')", charset))
            .addExpression("per_row_charset", "encode(input, charset)")
            .addExpression(
                "changing_charset", "encode(input, changing_charset)")
            .withIterations(100);
      };
      for (const size_t length : {1, 8, 12, 13, 31, 32, 63, 64, 256}) {
        addWorkload(
            fmt::format("ascii_{}bytes", length),
            {std::string(length, 'A'), std::string(length, 'B')});
      }
      const std::string_view charsetName{charset};
      const bool singleByte = legacyCharsets || charsetName == "US-ASCII" ||
          charsetName == "ISO-8859-1";
      if (legacyCharsets) {
        std::string sample = "\xC3\xA9";
        if (charsetName == "ISO-8859-5" || charsetName == "windows-1251" ||
            charsetName.starts_with("KOI8")) {
          sample = "\xD1\x8F";
        } else if (
            charsetName == "ISO-8859-6" || charsetName == "windows-1256") {
          sample = "\xD8\xA7";
        } else if (
            charsetName == "ISO-8859-7" || charsetName == "windows-1253") {
          sample = "\xCE\xB1";
        } else if (
            charsetName == "ISO-8859-8" || charsetName == "windows-1255") {
          sample = "\xD7\x90";
        } else if (charsetName == "x-iso-8859-11") {
          sample = "\xE0\xB8\x81";
        }
        std::string longSample;
        for (size_t i = 0; i < 128; ++i) {
          longSample += sample;
        }
        addWorkload("non_ascii", {sample, "A" + sample});
        addWorkload("non_ascii_long", {longSample});
      } else if (charsetName == "ISO-8859-1") {
        addWorkload("non_ascii", {"\xC3\xA9", "A\xC2\xA5"});
      } else if (!singleByte) {
        addWorkload("non_ascii", {"\xE2\x82\xAC", "A\xF0\x9F\x98\x80"});
      }
      if (!legacyCharsets && charsetName != "US-ASCII") {
        const std::string sample =
            charsetName == "ISO-8859-1" ? "\xC3\xA9" : "\xF0\x9F\x98\x80";
        for (const size_t length :
             {4, 8, 12, 13, 31, 32, 63, 64, 65, 127, 128, 256}) {
          std::string value;
          while (value.size() + sample.size() <= length) {
            value += sample;
          }
          value.append(length - value.size(), 'A');
          addWorkload(fmt::format("non_ascii_{}bytes", length), {value});
        }
      }
      if (replaceUnmappable || !singleByte) {
        addWorkload("malformed_utf8", {"A\xFF", "\xC3"});
        if (!legacyCharsets) {
          for (const size_t length : {4, 12, 13, 63, 64, 65}) {
            std::string value(length - 1, 'A');
            value += '\xFF';
            addWorkload(fmt::format("malformed_utf8_{}bytes", length), {value});
          }
        }
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
  EncodeBenchmark legacyStrict;
  EncodeBenchmark legacyReplacement;
  functions::sparksql::registerFunctions("");
  strict.addBenchmarks(false, false);
  replacement.addBenchmarks(true, false);
  legacyStrict.addBenchmarks(false, true);
  legacyReplacement.addBenchmarks(true, true);
  strict.testBenchmarks();
  replacement.testBenchmarks();
  legacyStrict.testBenchmarks();
  legacyReplacement.testBenchmarks();
  strict.registerBenchmarks();
  replacement.registerBenchmarks();
  legacyStrict.registerBenchmarks();
  legacyReplacement.registerBenchmarks();
  folly::runBenchmarks();
  return 0;
}
