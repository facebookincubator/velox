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

#include "gtest/gtest.h"

#include "velox/functions/prestosql/tests/FunctionBaseTest.h"
#include "velox/vector/fuzzer/VectorFuzzer.h"
#include "velox/vector/tests/VectorTestBase.h"

using namespace facebook::velox;
using namespace facebook::velox::test;
using functions::test::FunctionBaseTest;

// This test suite tests the simplified eval engine by:
//
//  1. Generating random vectors and encodings using VectorFuzzer;
//  2. Executing expressions against the generate vectors using the simplified
//  and common eval engines
//  3. Asserting that result vectors are the same.
//
class EvalSimplifiedTest : public FunctionBaseTest {
 protected:
  void assertExceptions(
      std::exception_ptr commonPtr,
      std::exception_ptr simplifiedPtr) {
    if (!commonPtr) {
      LOG(ERROR) << "Only simplified path threw exception:";
      std::rethrow_exception(simplifiedPtr);
    } else if (!simplifiedPtr) {
      LOG(ERROR) << "Only common path threw exception:";
      std::rethrow_exception(commonPtr);
    }
  }

  // Generate random (but deterministic) input row vectors.
  RowVectorPtr genRowVector(
      const RowTypePtr& types,
      const VectorFuzzer::Options& fuzzerOpts,
      folly::Random::DefaultGenerator& rng) {
    if (types == nullptr) {
      return nullptr;
    }

    std::vector<VectorPtr> vectors;
    vectors.reserve(types->size());

    for (const auto& type : types->children()) {
      size_t seed = folly::Random::rand32(rng);
      vectors.emplace_back(
          VectorFuzzer(fuzzerOpts, execCtx_.pool(), seed).fuzz(type));
      LOG(INFO) << "\t" << vectors.back()->toString();
    }
    return makeRowVector(vectors);
  }

  void runTest(
      const std::string& exprString,
      const RowTypePtr& types = nullptr) {
    auto expr = makeTypedExpr(exprString, types);

    // Instantiate common eval.
    exec::ExprSet exprSetCommon({expr}, &execCtx_);

    // Instantiate simplified eval.
    exec::ExprSetSimplified exprSetSimplified({expr}, &execCtx_);
    SelectivityVector rows(batchSize_);

    VectorFuzzer::Options fuzzerOpts;
    fuzzerOpts.vectorSize = 100;
    fuzzerOpts.stringVariableLength = true;
    fuzzerOpts.stringLength = 100;
    fuzzerOpts.nullChance = 10;

    for (size_t i = 0; i < iterations_; ++i) {
      LOG(INFO) << "============== Starting iteration with seed: " << seed_;
      folly::Random::DefaultGenerator rng(seed_);

      // Generate the input vectors.
      auto rowVector = genRowVector(types, fuzzerOpts, rng);

      exec::EvalCtx evalCtxCommon(&execCtx_, &exprSetCommon, rowVector.get());
      exec::EvalCtx evalCtxSimplified(
          &execCtx_, &exprSetSimplified, rowVector.get());

      // Evaluate using both engines.
      std::vector<VectorPtr> commonEvalResult{nullptr};
      std::vector<VectorPtr> simplifiedEvalResult{nullptr};

      std::exception_ptr exceptionCommonPtr;
      std::exception_ptr exceptionSimplifiedPtr;

      try {
        exprSetCommon.eval(rows, &evalCtxCommon, &commonEvalResult);
      } catch (const std::exception& e) {
        exceptionCommonPtr = std::current_exception();
      }

      try {
        exprSetSimplified.eval(rows, &evalCtxSimplified, &simplifiedEvalResult);
      } catch (const std::exception& e) {
        exceptionSimplifiedPtr = std::current_exception();
      }

      // Compare results or exceptions (if any). Fail is anything is different.
      if (exceptionCommonPtr || exceptionSimplifiedPtr) {
        assertExceptions(exceptionCommonPtr, exceptionSimplifiedPtr);
      } else {
        assertEqualVectors(
            commonEvalResult.front(), simplifiedEvalResult.front());
      }

      // Update the seed for the next iteration.
      seed_ = folly::Random::rand32(rng);
    }
  }

  const size_t batchSize_{100};
  const size_t iterations_{100};

  // Initial seed. Will get refreshed on each iteration. In order to reproduce
  // a test result, check the seed in the logs and paste it in here.
  int32_t seed_{123456};
};

TEST_F(EvalSimplifiedTest, constantOnly) {
  runTest("1 + 3 * 2 + 10 * 2");
}

TEST_F(EvalSimplifiedTest, constantAndInput) {
  runTest("1 + c0 - 2 + c0", ROW({"c0"}, {BIGINT()}));

  // Let it trigger some overflow exceptions.
  runTest("c0 + c1", ROW({"c0", "c1"}, {TINYINT(), TINYINT()}));
}

TEST_F(EvalSimplifiedTest, strings) {
  runTest("lower(upper(c0))", ROW({"c0"}, {VARCHAR()}));
}

TEST_F(EvalSimplifiedTest, doubles) {
  runTest("ceil(c1) * c0", ROW({"c0", "c1"}, {DOUBLE(), DOUBLE()}));
}

// Ensure that the right exprSet object is instantiated if `kExprEvalSimplified`
// is specified.
TEST_F(EvalSimplifiedTest, queryParameter) {
  queryCtx_->setConfigOverridesUnsafe({
      {core::QueryConfig::kExprEvalSimplified, "true"},
  });

  auto expr = makeTypedExpr("1 + 1", nullptr);
  auto exprSet = exec::makeExprSetFromFlag({expr}, &execCtx_);

  auto* ptr = dynamic_cast<exec::ExprSetSimplified*>(exprSet.get());
  EXPECT_TRUE(ptr != nullptr) << "expected ExprSetSimplified derived object.";
}

TEST_F(EvalSimplifiedTest, specialFormPropagateNulls) {
  // This test verifies an edge case where applyFunctionWithPeeling may produce
  // a result vector which is dictionary encoded and has fewer values than
  // are rows.
  // This can happen when the last value in a column used in an expression is
  // null which causes removeSureNulls to move the end of the SelectivityVector
  // forward.  When we incorrectly use rows.end() as the size of the
  // dictionary when rewrapping the results.
  // Normally this is masked when this vector is used in a function call which
  // produces a new output vector.  However, in SpecialForm expressions, we may
  // return the output untouched, and when we try to add back in the nulls, we
  // get an exception trying to resize the DictionaryVector.
  // This happens with evalSimplified in particular because the nulls are
  // removed early in evaluation.

  // Create a vector with a null in the last position.
  auto c0 = makeFlatVector<int64_t>(
      5,
      [](vector_size_t row) { return row; },
      [](vector_size_t row) { return row == 4; });
  // Create a constant vector, it's important that this vector uses an encoding
  // that can be peeled.
  auto c1 = vectorMaker_.constantVector<int32_t>({10});

  // Indices for dictionary encoding the vector, it's important that this
  // vector is dictionary encoded so the output is dictionary encoded.
  auto c0Indices = makeIndices(5, [](auto row) { return row; });

  auto rowType = ROW({"c0", "c1"}, {c0->type(), c1->type()});
  // Use try which is a SpecialForm expresssion to wrap the expression to
  // trigger the bug.
  auto expr = makeTypedExpr("try(round(\"c0\",\"c1\"))", rowType);
  exec::ExprSetSimplified exprSetSimplified({expr}, &execCtx_);
  auto rowVector = makeRowVector(
      {BaseVector::wrapInDictionary(nullptr, c0Indices, 5, c0), c1});
  exec::EvalCtx evalCtxSimplified(
      &execCtx_, &exprSetSimplified, rowVector.get());
  std::vector<VectorPtr> simplifiedEvalResult{nullptr};
  SelectivityVector rows(5);
  exprSetSimplified.eval(rows, &evalCtxSimplified, &simplifiedEvalResult);

  auto expectedResult = makeFlatVector<int64_t>(
      5,
      [](vector_size_t row) { return row; },
      [](vector_size_t row) { return row == 4; });
  assertEqualVectors(expectedResult, simplifiedEvalResult[0]);
}
