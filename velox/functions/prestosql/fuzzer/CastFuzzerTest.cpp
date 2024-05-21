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

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <gtest/gtest.h>

#include "velox/dwio/dwrf/writer/Writer.h"
#include "velox/exec/fuzzer/AggregationFuzzerRunner.h"
#include "velox/exec/fuzzer/PrestoQueryRunner.h"
#include "velox/exec/tests/utils/PlanBuilder.h"
#include "velox/expression/Expr.h"
#include "velox/expression/RegisterSpecialForm.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"

DEFINE_int64(
    seed,
    0,
    "Initial seed for random number generator used to reproduce previous "
    "results (0 means start with random seed).");

DEFINE_string(
    presto_url,
    "http://127.0.0.1:8080",
    "Presto coordinator URI along with port. If set, we use Presto "
    "source of truth."
    "--presto_url=http://127.0.0.1:8080");

DEFINE_uint32(
    req_timeout_ms,
    5000,
    "Timeout in milliseconds for HTTP requests made to reference DB, "
    "such as Presto. Example: --req_timeout_ms=2000");

namespace facebook::velox::exec::test {

auto constexpr batchSize = 1000;
auto constexpr kColName = "c0";

class CastFuzzer {
 public:
  CastFuzzer(
      size_t initialSeed,
      std::unique_ptr<PrestoQueryRunner> referenceQueryRunner)
      : referenceQueryRunner_{std::move(referenceQueryRunner)} {
    seed(initialSeed);
    parse::registerTypeResolver();
    exec::registerFunctionCallToSpecialForms();
    facebook::velox::aggregate::prestosql::registerAllAggregateFunctions(
        "", false, true);
  }

  void go() {
    VELOX_CHECK(
        FLAGS_steps > 0 || FLAGS_duration_sec > 0,
        "Either --steps or --duration_sec needs to be greater than zero.")

    auto startTime = std::chrono::system_clock::now();
    size_t iteration = 0;

    while (!isDone(iteration, startTime)) {
      LOG(INFO) << "==============================> Started iteration "
                << iteration << " (seed: " << currentSeed_ << ")";

      TypePtr floatingType = vectorFuzzer_.coinToss(0.5)
          ? static_cast<TypePtr>(DOUBLE())
          : static_cast<TypePtr>(REAL());

      LOG(INFO) << "Choosing type:" << floatingType->toString();
      VELOX_CHECK_NOT_NULL(pool_.get());
      auto input = makeRowVector(floatingType, batchSize);
      auto castInput =
          std::make_shared<facebook::velox::core::FieldAccessTypedExpr>(
              floatingType, std::string(kColName));
      std::vector<facebook::velox::core::TypedExprPtr> expr{
          std::make_shared<facebook::velox::core::CastTypedExpr>(
              VARCHAR(), castInput, false)};

      exec::ExprSet exprSet(expr, &execCtx_, false);
      SelectivityVector rows(input->size());
      VectorPtr veloxResults;

      // Evaluate Velox results.
      evaluate(exprSet, input, rows, veloxResults);

      // Evaluate same results on Presto.
      std::string sql = "SELECT cast (c0 as varchar)  FROM tmp";
      auto prestoResults = referenceQueryRunner_->executeVector(
          sql, {input}, ROW({kColName}, {VARCHAR()}));
      VELOX_CHECK_GT(prestoResults.size(), 0);

      // Compare results, we are using a custom way of comparing results,
      // since results obtained from Java can be off in the least significant
      // digit.
      auto veloxCastVector = veloxResults->asFlatVector<StringView>();
      auto prestoCastVector =
          prestoResults[0]->childAt(0)->asFlatVector<StringView>();

      auto exactMatch = 0;
      auto leastSignificantDigitOff = 0;

      VELOX_CHECK_EQ(veloxCastVector->size(), prestoCastVector->size());

      for (auto i = 0; i < veloxCastVector->size(); i++) {
        auto matchType = compareVarchars(prestoCastVector, veloxCastVector, i);

        if (matchType == MatchType::EXACT_MATCH) {
          exactMatch++;
        } else if (matchType == MatchType::LEAST_SIGNIFICANT_DIGIT_OFF) {
          leastSignificantDigitOff++;
        } else {
          VELOX_FAIL(fmt::format(
              "Cast conversion not matching for: {} to {}",
              prestoCastVector->valueAt(i),
              veloxCastVector->valueAt(i)));
        }
      }

      LOG(INFO) << fmt::format(
          "Results EXACT : {}, LEAST SIGNIFICANT: {}",
          exactMatch,
          leastSignificantDigitOff);

      seed(rng_());
      ++iteration;
    }
  };

 private:
  enum class MatchType {
    EXACT_MATCH = 1,
    LEAST_SIGNIFICANT_DIGIT_OFF,
    NOT_A_MATCH
  };

  /// Compare's two StringView vectors at index i. The StringView vectors are
  /// results of conversion of a floating point type to string. Returns
  /// EXACT_MATCH if both the strings are eactly matching.
  /// LEAST_SIGNIFICANT_DIGIT_OFF if they are off in the least siginficant digit
  /// NOT_A_MATCH if exponents or any other part of mantissa doesnt match.
  MatchType compareVarchars(
      const FlatVector<StringView>* expected,
      const FlatVector<StringView>* actual,
      size_t i) {
    VELOX_CHECK_EQ(expected->size(), actual->size());
    VELOX_CHECK_LT(i, expected->size());

    if (expected->isNullAt(i) && actual->isNullAt(i)) {
      return MatchType::EXACT_MATCH;
    } else if (expected->isNullAt(i) || actual->isNullAt(i)) {
      return MatchType::NOT_A_MATCH;
    }

    auto expectedString = expected->valueAt(i).getString();
    auto actualString = actual->valueAt(i).getString();

    if (expectedString == actualString) {
      return MatchType::EXACT_MATCH;
    }

    auto splitIntoMantissaExponent = [](std::string& val) {
      auto pos = val.find('E');
      return std::tuple(val.substr(0, pos - 1), val.substr(pos + 1));
    };

    auto [expectedMantissa, expectedExponent] =
        splitIntoMantissaExponent(expectedString);
    auto [actualMantissa, actualExponent] =
        splitIntoMantissaExponent(actualString);

    // Exponents should always be equal
    VELOX_CHECK_EQ(actualExponent, expectedExponent);

    if (expectedMantissa == actualMantissa) {
      return MatchType::EXACT_MATCH;
    }

    // Handle case when the last significant digit is off or missing
    auto [smallestLength, largerLength] =
        expectedMantissa.length() < actualMantissa.length()
        ? std::tuple{expectedMantissa.length(), actualMantissa.length()}
        : std::tuple{actualMantissa.length(), expectedMantissa.length()};

    // In practice java precision ranges from 15 - 17,
    // so give some leeway for last 3 significant digits
    // to be rounded.
    if ((largerLength - smallestLength) > 3) {
      return MatchType::NOT_A_MATCH;
    }

    if (expectedMantissa.substr(0, smallestLength) ==
        actualMantissa.substr(0, smallestLength)) {
      return MatchType::LEAST_SIGNIFICANT_DIGIT_OFF;
    }

    // Note we dont handle cases like ..5999 and ..6000 where last n digits get
    // rounded off.
    return MatchType::NOT_A_MATCH;
  }

  void evaluate(
      exec::ExprSet& exprSet,
      const RowVectorPtr& data,
      const SelectivityVector& rows,
      VectorPtr& result) {
    exec::EvalCtx evalCtx(&execCtx_, &exprSet, data.get());
    std::vector<VectorPtr> results{result};
    exprSet.eval(rows, evalCtx, results);

    if (!result) {
      result = results[0];
    }
  }

  RowVectorPtr makeRowVector(const TypePtr& type, vector_size_t size) {
    VectorFuzzer::Options opts;
    opts.vectorSize = size;
    opts.nullRatio = 0;
    VectorPtr input = vectorFuzzer_.fuzzFlat(type);
    return vectorMaker_.rowVector(
        std::vector<std::string>{std::string(kColName)},
        std::vector<VectorPtr>{std::move(input)});
  }

  void seed(size_t seed) {
    currentSeed_ = seed;
    vectorFuzzer_.reSeed(seed);
    rng_.seed(currentSeed_);
  }

  FuzzerGenerator rng_;
  size_t currentSeed_{0};
  std::unique_ptr<PrestoQueryRunner> referenceQueryRunner_;

  std::shared_ptr<memory::MemoryPool> rootPool_{
      memory::memoryManager()->addRootPool()};
  std::shared_ptr<memory::MemoryPool> pool_{rootPool_->addLeafChild("leaf")};
  std::shared_ptr<core::QueryCtx> queryCtx_{std::make_shared<core::QueryCtx>()};
  core::ExecCtx execCtx_{pool_.get(), queryCtx_.get()};
  facebook::velox::test::VectorMaker vectorMaker_{pool_.get()};
  VectorFuzzer vectorFuzzer_{{}, pool_.get()};
};

} // namespace facebook::velox::exec::test

int main(int argc, char** argv) {
  using namespace facebook::velox;

  ::testing::InitGoogleTest(&argc, argv);
  folly::Init init(&argc, &argv);

  memory::MemoryManager::initialize({});
  dwrf::registerDwrfWriterFactory();

  auto referenceQueryRunner = std::make_unique<exec::test::PrestoQueryRunner>(
      FLAGS_presto_url,
      "aggregation_fuzzer",
      static_cast<std::chrono::milliseconds>(FLAGS_req_timeout_ms));

  size_t initialSeed = FLAGS_seed == 0 ? std::time(nullptr) : FLAGS_seed;
  auto castFuzzer = facebook::velox::exec::test::CastFuzzer(
      initialSeed, std::move(referenceQueryRunner));

  castFuzzer.go();
  return RUN_ALL_TESTS();
}
