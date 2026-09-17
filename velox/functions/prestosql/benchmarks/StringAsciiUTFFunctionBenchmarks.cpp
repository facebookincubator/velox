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
#include <random>
#include "velox/common/fuzzer/Utils.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#ifdef VELOX_ENABLE_SPARK_FUNCTIONS
#include "velox/functions/sparksql/registration/Register.h"
#endif
#include "velox/vector/fuzzer/VectorFuzzer.h"

using namespace facebook::velox;
using namespace facebook::velox::exec;
using namespace facebook::velox::functions;

namespace {

using fuzzer::UTF8CharList;

class StringAsciiUTFFunctionBenchmark
    : public functions::test::FunctionBenchmarkBase {
 public:
  StringAsciiUTFFunctionBenchmark() : FunctionBenchmarkBase() {
    functions::prestosql::registerStringFunctions();
  }

  void runUpperLower(const std::string& fnName, bool utf) {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    if (utf) {
      opts.charEncodings.clear();
      opts.charEncodings = {UTF8CharList::UNICODE_CASE_SENSITIVE};
    }

    opts.stringLength = 100;
    opts.vectorSize = 100'000;
    VectorFuzzer fuzzer(opts, execCtx_.pool());
    auto vector = fuzzer.fuzzFlat(VARCHAR());

    auto rowVector = vectorMaker_.rowVector({vector});
    auto exprSet =
        compileExpression(fmt::format("{}(c0)", fnName), rowVector->type());

    suspender.dismiss();

    doRun(exprSet, rowVector);
  }

  void runSubStr(bool utf) {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    if (utf) {
      opts.charEncodings.clear();
      opts.charEncodings = {
          UTF8CharList::UNICODE_CASE_SENSITIVE,
          UTF8CharList::EXTENDED_UNICODE,
          UTF8CharList::MATHEMATICAL_SYMBOLS};
    }

    opts.stringLength = 100;
    opts.vectorSize = 10'000;
    VectorFuzzer fuzzer(opts, execCtx_.pool());
    auto vector = fuzzer.fuzzFlat(VARCHAR());

    auto positionVector = BaseVector::createConstant(
        INTEGER(), 25, opts.vectorSize, execCtx_.pool());

    auto rowVector = vectorMaker_.rowVector({vector, positionVector});

    auto exprSet = compileExpression("substr(c0, c1)", rowVector->type());

    suspender.dismiss();
    doRun(exprSet, rowVector);
  }

  void runLPadRPad(const std::string& fnName, bool utf) {
    folly::BenchmarkSuspender suspender;

    VectorFuzzer::Options opts;
    if (utf) {
      opts.charEncodings.clear();
      opts.charEncodings = {
          UTF8CharList::UNICODE_CASE_SENSITIVE,
          UTF8CharList::EXTENDED_UNICODE,
          UTF8CharList::MATHEMATICAL_SYMBOLS};
    }

    opts.stringLength = 10;
    opts.vectorSize = 10'000;
    VectorFuzzer fuzzer(opts, execCtx_.pool());
    auto stringVector = fuzzer.fuzzFlat(VARCHAR());
    auto padStringVector = fuzzer.fuzzFlat(VARCHAR());

    auto sizeVector = BaseVector::createConstant(
        INTEGER(), 55, opts.vectorSize, execCtx_.pool());

    auto rowVector =
        vectorMaker_.rowVector({stringVector, sizeVector, padStringVector});

    auto exprSet = compileExpression(
        fmt::format("{}(c0, c1, c2)", fnName), rowVector->type());

    suspender.dismiss();
    doRun(exprSet, rowVector);
  }

  // One iteration evaluates 10,240 rows, including result allocation and
  // release. Input generation, compilation and value validation are not timed.
  // Length includes two spaces on trimmed rows (one at each end for trim).
  // "all" trims every row, "half" trims odd rows, and "spaces" has no body.
  void runTrim(
      unsigned iterations,
      const std::string& function,
      vector_size_t length,
      std::string_view pattern) {
    folly::BenchmarkSuspender suspender;
    constexpr vector_size_t kBatchSize = 10'240;
    std::mt19937 random(20260912);
    const bool left = !function.ends_with("rtrim");
    const bool right = !function.ends_with("ltrim");
    const auto numLeadingSpaces = left ? (right ? 1 : 2) : 0;
    const auto numTrailingSpaces = right ? (left ? 1 : 2) : 0;
    std::vector<std::optional<std::string>> expected(kBatchSize);
    auto input = vectorMaker_.flatVector<std::string>(
        kBatchSize,
        [&](vector_size_t row) {
          const bool trim = pattern == "all" ||
              (pattern == "half" && row % 2 == 1) ||
              (pattern == "null50" && row % 4 == 3);
          std::string value(length, ' ');
          if (pattern != "spaces") {
            for (auto& character : value) {
              character = 'a' + random() % 26;
            }
          }
          if (trim) {
            value.replace(0, numLeadingSpaces, numLeadingSpaces, ' ');
            value.replace(
                length - numTrailingSpaces,
                numTrailingSpaces,
                numTrailingSpaces,
                ' ');
          }
          expected[row] = pattern == "spaces"
              ? ""
              : (trim ? value.substr(numLeadingSpaces, length - 2) : value);
          return value;
        },
        // NULL50 interleaves NULL, unchanged, NULL, trimmed rows.
        [&](vector_size_t row) { return pattern == "null50" && row % 2 == 0; });
    auto rowVector = vectorMaker_.rowVector({input});
    auto exprSet =
        compileExpression(fmt::format("{}(c0)", function), rowVector->type());
    SelectivityVector rows(kBatchSize);
    int64_t expectedTotalLength = 0;
    {
      auto result = evaluate(exprSet, rowVector, rows);
      const auto* strings = result->asChecked<FlatVector<StringView>>();
      for (vector_size_t row = 0; row < kBatchSize; ++row) {
        VELOX_CHECK_EQ(strings->isNullAt(row), !expected[row].has_value());
        if (expected[row]) {
          VELOX_CHECK_EQ(strings->valueAt(row), StringView(*expected[row]));
          expectedTotalLength += static_cast<int64_t>(expected[row]->size());
        } else {
          --expectedTotalLength;
        }
      }
    }

    suspender.dismiss();
    int64_t totalLength = 0;
    for (unsigned i = 0; i < iterations; ++i) {
      auto result = evaluate(exprSet, rowVector, rows);
      const auto* strings = result->asChecked<FlatVector<StringView>>();
      for (vector_size_t row = 0; row < kBatchSize; ++row) {
        totalLength += strings->isNullAt(row)
            ? -1
            : static_cast<int64_t>(strings->valueAt(row).size());
      }
      folly::doNotOptimizeAway(totalLength);
    }
    suspender.rehire();
    VELOX_CHECK_EQ(totalLength, expectedTotalLength * iterations);
  }

  void doRun(ExprSet& exprSet, const RowVectorPtr& rowVector) {
    uint32_t cnt = 0;
    for (auto i = 0; i < 100; i++) {
      cnt += evaluate(exprSet, rowVector)->size();
    }
    folly::doNotOptimizeAway(cnt);
  }
};

BENCHMARK(utfLower) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runUpperLower("lower", true);
}

BENCHMARK_RELATIVE(asciiLower) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runUpperLower("lower", false);
}

BENCHMARK(utfUpper) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runUpperLower("upper", true);
}

BENCHMARK_RELATIVE(asciiUpper) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runUpperLower("upper", false);
}

BENCHMARK(utfSubStr) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runSubStr(true);
}

BENCHMARK_RELATIVE(asciiSubStr) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runSubStr(false);
}

BENCHMARK(utfLPad) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runLPadRPad("lpad", true);
}

BENCHMARK_RELATIVE(aciiLPad) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runLPadRPad("lpad", false);
}

BENCHMARK(utfRPad) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runLPadRPad("rpad", true);
}

BENCHMARK_RELATIVE(aciiRPad) {
  StringAsciiUTFFunctionBenchmark benchmark;
  benchmark.runLPadRPad("rpad", false);
}

void addTrimBenchmarks(StringAsciiUTFFunctionBenchmark& benchmark) {
  prestosql::registerStringFunctions("presto_");
  std::vector<std::string> dialects{"presto"};
#ifdef VELOX_ENABLE_SPARK_FUNCTIONS
  sparksql::registerFunctions("spark_");
  dialects.emplace_back("spark");
#endif
  for (const auto& dialect : dialects) {
    for (const std::string function : {"rtrim", "ltrim", "trim"}) {
      const auto add = [&](vector_size_t length, const std::string& pattern) {
        folly::addBenchmark(
            __FILE__,
            fmt::format(
                "trim_{}_{}_l{}_{}", dialect, function, length, pattern),
            [&benchmark,
             expression = dialect + "_" + function,
             length,
             pattern](unsigned iterations) {
              benchmark.runTrim(iterations, expression, length, pattern);
              return iterations;
            });
      };
      for (auto length : {10, 12, 13, 64, 256}) {
        for (const auto* pattern : {"none", "all", "half"}) {
          add(length, pattern);
        }
      }
      add(64, "null50");
      add(0, "empty");
      add(64, "spaces");
    }
  }
}
} // namespace

// Preliminary release run, before ascii optimization.
//============================================================================
//../../velox/functions/prestosql/benchmarks/StringAsciiUTFFunctionBenchmarks.cpprelative
// time/iter  iters/s
//============================================================================
// utfLower                                                    67.71ms    14.77
// asciiLower                                        99.84%    67.82ms    14.75
// utfUpper                                                    67.75ms    14.76
// asciiUpper                                        98.22%    68.98ms    14.50
//============================================================================
int main(int argc, char** argv) {
  folly::Init init{&argc, &argv};
  memory::MemoryManager::initialize(memory::MemoryManager::Options{});

  StringAsciiUTFFunctionBenchmark benchmark;
  addTrimBenchmarks(benchmark);
  folly::runBenchmarks();
  return 0;
}
