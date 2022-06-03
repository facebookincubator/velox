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

#include "vector/fuzzer/VectorFuzzer.h"
#include "velox/functions/lib/Re2Functions.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/tpch/gen/TpchGen.h"

using namespace facebook::velox;
using namespace facebook::velox::tpch;
using namespace facebook::velox::functions;
using namespace facebook::velox::functions::test;

DEFINE_int32(vector_size, 10000, "Vector size");
DEFINE_int32(num_rows, 10000, "Number of rows");

namespace {

class LikeFunctionsBenchmark : public FunctionBaseTest,
                               public FunctionBenchmarkBase {
 public:
  explicit LikeFunctionsBenchmark() {
    exec::registerStatefulVectorFunction("like", likeSignatures(), makeLike);

    folly::BenchmarkSuspender kSuspender;
    VectorFuzzer::Options opts;
    opts.vectorSize = FLAGS_vector_size;
    VectorFuzzer fuzzer(opts, FunctionBenchmarkBase::pool());
    inputFuzzer_ = fuzzer.fuzzFlat(VARCHAR());
    kSuspender.dismiss();
  }

  // Generate random string using characters from characterSet.
  std::string generateRandomString(const char* characterSet) {
    vector_size_t characterSetLength = strlen(characterSet);
    vector_size_t outputStringLength;
    vector_size_t minimumLength = 1;
    vector_size_t maximumLength = 10;
    outputStringLength = rand() % maximumLength + minimumLength;
    std::string output;

    for (int i = 0; i < outputStringLength; i++) {
      output += characterSet[rand() % characterSetLength];
    }
    return output;
  }

  std::string generatePattern(
      PatternKind patternKind,
      const std::string& inputString) {
    switch (patternKind) {
      case PatternKind::kExactlyN:
        return std::string(inputString.size(), '_');
      case PatternKind::kAtLeastN:
        return generateRandomString(kWildcardCharacterSet);
      case PatternKind::kPrefix: {
        auto fixedPatternLength =
            std::min(vector_size_t(inputString.size()), 10);
        auto fixedPatternString = inputString.substr(0, fixedPatternLength);
        return fixedPatternString + generateRandomString(kAnyWildcardCharacter);
      }
      case PatternKind::kSuffix: {
        auto fixedPatternStartIdx =
            std::max(vector_size_t(inputString.size() - 10), 0);
        auto fixedPatternString = inputString.substr(fixedPatternStartIdx, 10);
        return generateRandomString(kAnyWildcardCharacter) + fixedPatternString;
      }
      default:
        return inputString;
    }
  }

  size_t run(VectorPtr input, const StringView patternString) {
    const auto data = makeRowVector({input});
    size_t cnt = 0;
    auto result = FunctionBaseTest::evaluate<SimpleVector<bool>>(
        fmt::format("like(c0, '{}')", patternString), data);
    cnt += result->size();
    folly::doNotOptimizeAway(cnt);

    return cnt;
  }

  size_t run(PatternKind patternKind) {
    const auto input = inputFuzzer_->values()->as<StringView>();
    auto patternString = generatePattern(patternKind, input[0].str());
    std::vector<std::string> patternVector(FLAGS_vector_size, patternString);
    const auto data = makeRowVector({inputFuzzer_});
    size_t cnt = 0;

    auto result = FunctionBaseTest::evaluate<SimpleVector<bool>>(
        fmt::format("like(c0, '{}')", patternString), data);
    cnt += result->size();
    folly::doNotOptimizeAway(cnt);

    return cnt;
  }

  // We inherit from FunctionBaseTest so that we can get access to the helpers
  // it defines, but since it is supposed to be a test fixture TestBody() is
  // declared pure virtual.  We must provide an implementation here.
  void TestBody() override {}

 private:
  static constexpr const char* kWildcardCharacterSet = "%_";
  static constexpr const char* kAnyWildcardCharacter = "%";
  VectorPtr inputFuzzer_;
};

std::unique_ptr<LikeFunctionsBenchmark> benchmark;

BENCHMARK_MULTI(wildcardExactlyN) {
  return benchmark->run(facebook::velox::functions::PatternKind::kExactlyN);
}

BENCHMARK_MULTI(wildcardAtLeastN) {
  return benchmark->run(PatternKind::kAtLeastN);
}

BENCHMARK_MULTI(fixedPattern) {
  return benchmark->run(PatternKind::kFixed);
}

BENCHMARK_MULTI(prefixPattern) {
  return benchmark->run(PatternKind::kPrefix);
}

BENCHMARK_MULTI(suffixPattern) {
  return benchmark->run(PatternKind::kSuffix);
}

BENCHMARK_DRAW_LINE();

BENCHMARK_MULTI(tpchQuery2) {
  auto tpchPart = genTpchPart(FLAGS_num_rows);
  auto partTypeVector = tpchPart->childAt(4);
  return benchmark->run(partTypeVector, "%BRASS");
}

BENCHMARK_MULTI(tpchQuery9) {
  auto tpchPart = genTpchPart(FLAGS_num_rows);
  auto partNameVector = tpchPart->childAt(1);
  return benchmark->run(partNameVector, "%green%");
}

BENCHMARK_MULTI(tpchQuery13) {
  auto tpchOrders = genTpchOrders(FLAGS_num_rows);
  auto orderCommentVector = tpchOrders->childAt(8);
  return benchmark->run(orderCommentVector, "%special%requests%");
}

BENCHMARK_MULTI(tpchQuery14) {
  auto tpchPart = genTpchPart(FLAGS_num_rows);
  auto partTypeVector = tpchPart->childAt(4);
  return benchmark->run(partTypeVector, "PROMO%");
}

BENCHMARK_MULTI(tpchQuery16Part) {
  auto tpchPart = genTpchPart(FLAGS_num_rows);
  auto partTypeVector = tpchPart->childAt(4);
  return benchmark->run(partTypeVector, "MEDIUM POLISHED%");
}

BENCHMARK_MULTI(tpchQuery16Supplier) {
  auto tpchSupplier = genTpchSupplier(FLAGS_num_rows);
  auto supplierCommentVector = tpchSupplier->childAt(6);
  return benchmark->run(supplierCommentVector, "%Customer%Complaints%");
}

BENCHMARK_MULTI(tpchQuery20) {
  auto tpchPart = genTpchPart(FLAGS_num_rows);
  auto partNameVector = tpchPart->childAt(1);
  return benchmark->run(partNameVector, "forest%");
}

} // namespace

int main(int argc, char* argv[]) {
  folly::init(&argc, &argv, true);
  benchmark = std::make_unique<LikeFunctionsBenchmark>();
  folly::runBenchmarks();
  benchmark.reset();

  return 0;
}
