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

#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/tpch/gen/TpchGen.h"
#include "velox/functions/Registerer.h"
#include "velox/functions/lib/benchmarks/FunctionBenchmarkBase.h"
#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/vector/FlatVector.h"

#include "velox/vector/BuilderTypeUtils.h"
#include "velox/functions/lib/Re2Functions.h"

using namespace facebook::velox;
using namespace facebook::velox::functions;
using namespace facebook::velox::functions::test;
using namespace facebook::velox::test;
using namespace facebook::velox::tpch;

namespace {

class LikeExpressionBenchmark
   : public FunctionBenchmarkBase, FunctionBaseTest {
public:
 explicit LikeExpressionBenchmark(): FunctionBenchmarkBase() {
   exec::registerStatefulVectorFunction(
       "like", likeSignatures(), makeLike);
 }

 void run(FlatVector<StringView> input, std::string pattern, int times) {
   auto like = [&](std::optional<std::string> str,
                   std::optional<std::string> pattern) {
     return evaluateOnce<bool>("like(c0, '" + *pattern + "')", str);
   };

   for(int j = 0; j < times; j++) {
      for (size_t i = 0; i < input.size(); ++i) {
        if (!input.isNullAt(i)) {
          like(input.valueAt(i).getString(), pattern);
        }
      }
   }
 }

 void TestBody() {
   return;
 }
};

} // namespace

std::unique_ptr<LikeExpressionBenchmark> benchmark;

BENCHMARK(q2) {
  std::string pattern = "%BRASS";
  auto flatVector = genTpchPartType(100);
  benchmark->run(flatVector, pattern, 100);
}

BENCHMARK(q9) {
  std::string pattern = "%green%";
  auto flatVector = genTpchPartName(100);
  benchmark->run(flatVector, pattern, 100);
}

BENCHMARK(q13) {
  std::string pattern = "%special%requests%";
  auto flatVector = genTpchOrderComment(100);
  benchmark->run(flatVector, pattern, 100);
}

BENCHMARK(q14) {
  std::string pattern = "PROMO%";
  auto flatVector = genTpchPartType(100);
  benchmark->run(flatVector, pattern, 100);
}

BENCHMARK(q16_1) {
  std::string pattern = "MEDIUM POLISHED%";
  auto flatVector = genTpchPartType(100);
  benchmark->run(flatVector, pattern, 100);
}

BENCHMARK(q16_2) {
  std::string pattern = "%Customer%Complaints%";
  auto flatVector = genTpchSupplierComment(100);
  benchmark->run(flatVector, pattern, 100);
}

BENCHMARK(q20) {
  std::string pattern = "forest%";
  auto flatVector = genTpchPartName(100);
  benchmark->run(flatVector, pattern, 100);
}

int main(int argc, char* argv[]) {
 benchmark = std::make_unique<LikeExpressionBenchmark>();
 folly::runBenchmarks();
 benchmark.reset();

 return 0;
}
