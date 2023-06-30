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

#include "velox/expression/tests/ExpressionVerifier.h"
#include "velox/common/base/Fs.h"
#include "velox/expression/Expr.h"
#include "velox/vector/VectorSaver.h"
#include "velox/vector/tests/utils/VectorMaker.h"

namespace facebook::velox::test {

namespace {
void logRowVector(const RowVectorPtr& rowVector) {
  if (rowVector == nullptr) {
    return;
  }
  LOG(INFO) << rowVector->childrenSize() << " vectors as input:";
  for (const auto& child : rowVector->children()) {
    LOG(INFO) << "\t" << child->toString(/*recursive=*/true);
  }

  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "RowVector contents (" << rowVector->type()->toString()
              << "):";

    for (vector_size_t i = 0; i < rowVector->size(); ++i) {
      LOG(INFO) << "\tAt " << i << ": " << rowVector->toString(i);
    }
  }
}

void compareVectors(
    const VectorPtr& left,
    const VectorPtr& right,
    const SelectivityVector& rows) {
  // Print vector contents if in verbose mode.
  size_t vectorSize = left->size();
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "== Result contents (common vs. simple): ";
    rows.applyToSelected([&](vector_size_t row) {
      LOG(INFO) << fmt::format(
          "At {} [ {} vs {} ]", row, left->toString(row), right->toString(row));
    });
    LOG(INFO) << "===================";
  }

  rows.applyToSelected([&](vector_size_t row) {
    VELOX_CHECK(
        left->equalValueAt(right.get(), row, row),
        "Different results at idx '{}': '{}' vs. '{}'",
        row,
        left->toString(row),
        right->toString(row));
  });

  LOG(INFO) << "All results match.";
}

} // namespace

ResultOrError ExpressionVerifier::verify(
    const std::vector<core::TypedExprPtr>& plans,
    const RowVectorPtr& rowVector,
    VectorPtr&& resultVector,
    bool canThrow,
    std::vector<int> columnsToWrapInLazy) {
  for (int i = 0; i < plans.size(); ++i) {
    LOG(INFO) << "Executing expression " << i << " : " << plans[i]->toString();
  }
  logRowVector(rowVector);

  // Store data and expression in case of reproduction.
  VectorPtr copiedResult;
  std::string sql = "";

  // Complex constants that aren't all expressible in sql
  std::vector<VectorPtr> complexConstants;
  // Deep copy to preserve the initial state of result vector.
  if (!options_.reproPersistPath.empty()) {
    if (resultVector) {
      copiedResult = BaseVector::copy(*resultVector);
    }
    // Disabling constant folding in order to preserve the original expression
    try {
      auto exprs = exec::ExprSet(plans, execCtx_, false).exprs();
      for (int i = 0; i < exprs.size(); ++i) {
        if (i > 0) {
          sql += ", ";
        }
        sql += exprs[i]->toSql(&complexConstants);
      }
    } catch (const std::exception& e) {
      LOG(WARNING) << "Failed to generate SQL: " << e.what();
      sql = "<failed to generate>";
    }
    if (options_.persistAndRunOnce) {
      persistReproInfo(
          rowVector, columnsToWrapInLazy, copiedResult, sql, complexConstants);
    }
  }

  // Execute expression plan using both common and simplified evals.
  std::vector<VectorPtr> commonEvalResult;
  std::vector<VectorPtr> simplifiedEvalResult;
  if (resultVector && resultVector->encoding() == VectorEncoding::Simple::ROW) {
    auto resultRowVector = resultVector->asUnchecked<RowVector>();
    auto children = resultRowVector->children();
    commonEvalResult.resize(children.size());
    simplifiedEvalResult.resize(children.size());
    for (int i = 0; i < children.size(); ++i) {
      commonEvalResult[i] = children[i];
    }
  } else {
    // For backwards compatibility where there was a single result and plan.
    VELOX_CHECK_EQ(plans.size(), 1);
    commonEvalResult.push_back(resultVector);
    simplifiedEvalResult.resize(1);
  }
  std::exception_ptr exceptionCommonPtr;
  std::exception_ptr exceptionSimplifiedPtr;

  VLOG(1) << "Starting common eval execution.";
  SelectivityVector rows{rowVector ? rowVector->size() : 1};

  // Execute with common expression eval path. Some columns of the input row
  // vector will be wrapped in lazy as specified in 'columnsToWrapInLazy'.
  try {
    exec::ExprSet exprSetCommon(
        plans, execCtx_, !options_.disableConstantFolding);
    auto inputRowVector = rowVector;
    if (!columnsToWrapInLazy.empty()) {
      inputRowVector =
          VectorFuzzer::fuzzRowChildrenToLazy(rowVector, columnsToWrapInLazy);
      LOG(INFO) << "Modified inputs for common eval path: ";
      logRowVector(inputRowVector);
    }
    exec::EvalCtx evalCtxCommon(execCtx_, &exprSetCommon, inputRowVector.get());

    exprSetCommon.eval(rows, evalCtxCommon, commonEvalResult);
  } catch (const VeloxUserError&) {
    if (!canThrow) {
      LOG(ERROR)
          << "Common eval wasn't supposed to throw, but it did. Aborting.";
      persistReproInfoIfNeeded(
          rowVector, columnsToWrapInLazy, copiedResult, sql, complexConstants);
      throw;
    }
    exceptionCommonPtr = std::current_exception();
  } catch (...) {
    LOG(ERROR)
        << "Common eval: Exceptions other than VeloxUserError are not allowed.";
    persistReproInfoIfNeeded(
        rowVector, columnsToWrapInLazy, copiedResult, sql, complexConstants);
    throw;
  }

  VLOG(1) << "Starting simplified eval execution.";

  // Execute with simplified expression eval path.
  try {
    exec::ExprSetSimplified exprSetSimplified(plans, execCtx_);
    exec::EvalCtx evalCtxSimplified(
        execCtx_, &exprSetSimplified, rowVector.get());

    exprSetSimplified.eval(rows, evalCtxSimplified, simplifiedEvalResult);
  } catch (const VeloxUserError&) {
    exceptionSimplifiedPtr = std::current_exception();
  } catch (...) {
    LOG(ERROR)
        << "Simplified eval: Exceptions other than VeloxUserError are not allowed.";
    persistReproInfoIfNeeded(
        rowVector, columnsToWrapInLazy, copiedResult, sql, complexConstants);
    throw;
  }

  try {
    // Compare results or exceptions (if any). Fail if anything is different.
    if (exceptionCommonPtr || exceptionSimplifiedPtr) {
      // Throws in case exceptions are not compatible. If they are compatible,
      // return false to signal that the expression failed.
      compareExceptions(exceptionCommonPtr, exceptionSimplifiedPtr);
      return {nullptr, exceptionCommonPtr};
    } else {
      // Throws in case output is different.
      VELOX_CHECK_EQ(commonEvalResult.size(), plans.size());
      VELOX_CHECK_EQ(simplifiedEvalResult.size(), plans.size());
      for (int i = 0; i < plans.size(); ++i) {
        compareVectors(commonEvalResult[i], simplifiedEvalResult[i], rows);
      }
    }
  } catch (...) {
    persistReproInfoIfNeeded(
        rowVector, columnsToWrapInLazy, copiedResult, sql, complexConstants);
    throw;
  }

  if (!options_.reproPersistPath.empty() && options_.persistAndRunOnce) {
    // A guard to make sure it runs only once with persistAndRunOnce flag
    // turned on. It shouldn't reach here normally since the flag is used to
    // persist repro info for crash failures. But if it hasn't crashed by now,
    // we still don't want another iteration.
    LOG(WARNING)
        << "Iteration succeeded with --persist_and_run_once flag enabled "
           "(expecting crash failure)";
    exit(0);
  }

  return {
      VectorMaker(commonEvalResult[0]->pool()).rowVector(commonEvalResult),
      nullptr};
}

void ExpressionVerifier::persistReproInfoIfNeeded(
    const VectorPtr& inputVector,
    const std::vector<int>& columnsToWrapInLazy,
    const VectorPtr& resultVector,
    const std::string& sql,
    const std::vector<VectorPtr>& complexConstants) {
  if (options_.reproPersistPath.empty()) {
    LOG(INFO) << "Skipping persistence because repro path is empty.";
  } else if (!options_.persistAndRunOnce) {
    persistReproInfo(
        inputVector, columnsToWrapInLazy, resultVector, sql, complexConstants);
  }
}

void ExpressionVerifier::persistReproInfo(
    const VectorPtr& inputVector,
    std::vector<int> columnsToWrapInLazy,
    const VectorPtr& resultVector,
    const std::string& sql,
    const std::vector<VectorPtr>& complexConstants) {
  std::string inputPath;
  std::string lazyListPath;
  std::string resultPath;
  std::string sqlPath;
  std::string complexConstantsPath;

  const auto basePath = options_.reproPersistPath.c_str();
  if (!common::generateFileDirectory(basePath)) {
    return;
  }

  // Create a new directory
  auto dirPath = common::generateTempFolderPath(basePath, "expressionVerifier");
  if (!dirPath.has_value()) {
    LOG(INFO) << "Failed to create directory for persisting repro info.";
    return;
  }
  // Saving input vector
  inputPath = fmt::format("{}/{}", dirPath->c_str(), kInputVectorFileName);
  try {
    saveVectorToFile(inputVector.get(), inputPath.c_str());
  } catch (std::exception& e) {
    inputPath = e.what();
  }

  // Saving the list of column indices that are to be wrapped in lazy.
  if (!columnsToWrapInLazy.empty()) {
    lazyListPath =
        fmt::format("{}/{}", dirPath->c_str(), kIndicesOfLazyColumnsFileName);
    try {
      saveStdVectorToFile<int>(columnsToWrapInLazy, lazyListPath.c_str());
    } catch (std::exception& e) {
      lazyListPath = e.what();
    }
  }

  // Saving result vector
  if (resultVector) {
    resultPath = fmt::format("{}/{}", dirPath->c_str(), kResultVectorFileName);
    try {
      saveVectorToFile(resultVector.get(), resultPath.c_str());
    } catch (std::exception& e) {
      resultPath = e.what();
    }
  }

  // Saving sql
  sqlPath = fmt::format("{}/{}", dirPath->c_str(), kExpressionSqlFileName);
  try {
    saveStringToFile(sql, sqlPath.c_str());
  } catch (std::exception& e) {
    sqlPath = e.what();
  }
  // Saving complex constants
  if (!complexConstants.empty()) {
    complexConstantsPath =
        fmt::format("{}/{}", dirPath->c_str(), kComplexConstantsFileName);
    try {
      saveVectorToFile(
          VectorMaker(complexConstants[0]->pool())
              .rowVector(complexConstants)
              .get(),
          complexConstantsPath.c_str());
    } catch (std::exception& e) {
      complexConstantsPath = e.what();
    }
  }

  std::stringstream ss;
  ss << "Persisted input: --fuzzer_repro_path " << dirPath.value();
  ss << " --input_path " << inputPath;
  if (resultVector) {
    ss << " --result_path " << resultPath;
  }
  ss << " --sql_path " << sqlPath;
  if (!columnsToWrapInLazy.empty()) {
    ss << " --lazy_column_list_path " << lazyListPath;
  }
  if (!complexConstants.empty()) {
    ss << " --complex_constant_path " << complexConstantsPath;
  }
  LOG(INFO) << ss.str();
}

namespace {
class MinimalSubExpressionFinder {
 public:
  MinimalSubExpressionFinder(
      ExpressionVerifier&& verifier,
      VectorFuzzer& vectorFuzzer)
      : verifier_(verifier), vectorFuzzer_(vectorFuzzer) {}

  // Tries subexpressions of plan until finding the minimal failing subtree.
  void findMinimalExpression(
      core::TypedExprPtr plan,
      const RowVectorPtr& rowVector,
      const std::vector<int>& columnsToWrapInLazy) {
    if (verifyWithResults(plan, rowVector, columnsToWrapInLazy)) {
      errorExit("Retry should have failed");
    }
    bool minimalFound =
        findMinimalRecursive(plan, rowVector, columnsToWrapInLazy);
    if (minimalFound) {
      errorExit("Found minimal failing expression.");
    } else {
      errorExit("Only the top level expression failed!");
    }
  }

 private:
  // Central point for failure exit.
  void errorExit(const std::string& text) {
    VELOX_FAIL(text);
  }

  // Verifies children of 'plan'. If all succeed, sets minimalFound to
  // true and reruns 'plan' wth and without lazy vectors. Set
  // breakpoint inside this to debug failures.
  bool findMinimalRecursive(
      core::TypedExprPtr plan,
      const RowVectorPtr& rowVector,
      const std::vector<int>& columnsToWrapInLazy) {
    bool anyFailed = false;
    for (auto& input : plan->inputs()) {
      if (!verifyWithResults(input, rowVector, columnsToWrapInLazy)) {
        anyFailed = true;
        bool minimalFound =
            findMinimalRecursive(input, rowVector, columnsToWrapInLazy);
        if (minimalFound) {
          return true;
        }
      }
    }
    if (!anyFailed) {
      LOG(INFO) << "Failed with all children succeeding: " << plan->toString();
      // Re-running the minimum failed. Put breakpoint here to debug.
      verifyWithResults(plan, rowVector, columnsToWrapInLazy);
      if (!columnsToWrapInLazy.empty()) {
        LOG(INFO) << "Trying without lazy:";
        if (verifyWithResults(plan, rowVector, {})) {
          LOG(INFO) << "Minimal failure succeeded without lazy vectors";
        }
      }
      LOG(INFO) << fmt::format(
          "Found minimal failing subexpression: {}", plan->toString());
      return true;
    }

    return false;
  }

  // Verifies a 'plan' against a 'rowVector' with and without pre-existing
  // contents in result vector.
  bool verifyWithResults(
      core::TypedExprPtr plan,
      const RowVectorPtr& rowVector,
      const std::vector<int>& columnsToWrapInLazy) {
    VectorPtr result;
    LOG(INFO) << "Running with empty results vector :" << plan->toString();
    bool emptyResult = verifyPlan(plan, rowVector, columnsToWrapInLazy, result);
    LOG(INFO) << "Running with non empty vector :" << plan->toString();
    result = vectorFuzzer_.fuzzFlat(plan->type());
    bool filledResult =
        verifyPlan(plan, rowVector, columnsToWrapInLazy, result);
    if (emptyResult != filledResult) {
      LOG(ERROR) << fmt::format(
          "Different results for empty vs populated ! Empty result = {} filledResult = {}",
          emptyResult,
          filledResult);
    }
    return filledResult && emptyResult;
  }

  // Runs verification on a plan with a provided result vector.
  // Returns true if the verification is successful.
  bool verifyPlan(
      core::TypedExprPtr plan,
      const RowVectorPtr& rowVector,
      const std::vector<int>& columnsToWrapInLazy,
      VectorPtr results) {
    // Turn off unnecessary logging.
    FLAGS_minloglevel = 2;
    bool success = true;

    try {
      verifier_.verify(
          {plan},
          rowVector,
          results ? BaseVector::copy(*results) : nullptr,
          true, // canThrow
          columnsToWrapInLazy);
    } catch (const std::exception& e) {
      success = false;
    }
    FLAGS_minloglevel = 0;
    return success;
  }

  ExpressionVerifier verifier_;
  VectorFuzzer& vectorFuzzer_;
};
} // namespace

void computeMinimumSubExpression(
    ExpressionVerifier&& minimalVerifier,
    VectorFuzzer& fuzzer,
    const std::vector<core::TypedExprPtr>& plans,
    const RowVectorPtr& rowVector,
    const std::vector<int>& columnsToWrapInLazy) {
  auto finder = MinimalSubExpressionFinder(std::move(minimalVerifier), fuzzer);
  if (plans.size() > 1) {
    LOG(INFO)
        << "Found more than one expression, minimal subexpression might not work"
           " in cases where bugs are due to side effects when evaluating multiple"
           " expressions.";
  }
  for (auto plan : plans) {
    LOG(INFO) << "============================================";
    LOG(INFO) << "Finding minimal subexpression for plan:" << plan->toString();
    finder.findMinimalExpression(plan, rowVector, columnsToWrapInLazy);
    LOG(INFO) << "============================================";
  }
}

} // namespace facebook::velox::test
