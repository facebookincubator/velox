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
#include "velox/functions/lib/HyperscanFunctions.h"
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include "hs.h"
#include "velox/functions/lib/string/StringImpl.h"
#include "velox/vector/FunctionVector.h"

namespace facebook::velox::functions {
namespace {

class MatchConstantPattern final : public exec::VectorFunction {
 public:
  explicit MatchConstantPattern(StringView pattern) {
    hs_compile_error_t* compile_err;
    if (hs_compile(
            pattern.data(),
            // Using single match flag, which can greatly improve performance.
            HS_FLAG_SINGLEMATCH | HS_FLAG_DOTALL,
            HS_MODE_BLOCK,
            NULL,
            &database_,
            &compile_err) != HS_SUCCESS) {
      fprintf(
          stderr,
          "ERROR: Unable to compile pattern \"%s\": %s\n",
          pattern.data(),
          compile_err->message);
      hs_free_compile_error(compile_err);
    }

    if (hs_alloc_scratch(database_, &scratch_) != HS_SUCCESS) {
      fprintf(stderr, "ERROR: Unable to allocate scratch space. Exiting.\n");
      hs_free_database(database_);
    }
  }

  /**
   * Callback function for matching case.
   */
  static int eventHandler(
      unsigned int id,
      unsigned long long from,
      unsigned long long to,
      unsigned int flags,
      void* ctx) {
    *(bool*)ctx = true;
    return 0;
  }

  void apply(
      const SelectivityVector& rows,
      std::vector<VectorPtr>& args,
      const TypePtr& /* outputType */,
      exec::EvalCtx& context,
      VectorPtr& resultRef) const final {
    VELOX_CHECK_EQ(args.size(), 2);
    context.ensureWritable(rows, BOOLEAN(), resultRef);
    FlatVector<bool>& result = *resultRef->as<FlatVector<bool>>();
    exec::LocalDecodedVector toSearch(context, *args[0], rows);

    context.applyToSelectedNoThrow(rows, [&](vector_size_t i) {
      StringView toSearchString = toSearch->valueAt<StringView>(i);
      bool isMatched = false;
      if (hs_scan(
              database_,
              toSearchString.data(),
              toSearchString.size(),
              0,
              scratch_,
              eventHandler,
              &isMatched) != HS_SUCCESS) {
        fprintf(stderr, "ERROR: Unable to scan input buffer. Exiting.\n");
        hs_free_scratch(scratch_);
        hs_free_database(database_);
      }
      result.set(i, isMatched);
    });
  }

 private:
  hs_database_t* database_;
  hs_scratch_t* scratch_ = NULL;
};
} // namespace

std::string printTypesCsv(
    const std::vector<exec::VectorFunctionArg>& inputArgs) {
  std::string result;
  result.reserve(inputArgs.size() * 10);
  for (const auto& input : inputArgs) {
    folly::toAppend(
        result.empty() ? "" : ", ", input.type->toString(), &result);
  }
  return result;
}

std::shared_ptr<exec::VectorFunction> makeHyperscanMatch(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config) {
  if (inputArgs.size() != 2 || !inputArgs[0].type->isVarchar() ||
      !inputArgs[1].type->isVarchar()) {
    VELOX_UNSUPPORTED(
        "{} expected (VARCHAR, VARCHAR) but got ({})",
        name,
        printTypesCsv(inputArgs));
  }

  BaseVector* constantPattern = inputArgs[1].constantValue.get();

  if (constantPattern != nullptr && !constantPattern->isNullAt(0)) {
    return std::make_shared<MatchConstantPattern>(
        constantPattern->as<ConstantVector<StringView>>()->valueAt(0));
  }
  // TODO: support non-constant pattern.
  VELOX_UNREACHABLE();
}

} // namespace facebook::velox::functions
