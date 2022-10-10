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
#include <gtest/gtest.h>

#include "velox/common/memory/Memory.h"
#include "velox/core/QueryCtx.h"
#include "velox/expression/tests/ExpressionVerifier.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/parse/Expressions.h"
#include "velox/parse/ExpressionsParser.h"
#include "velox/parse/TypeResolver.h"
#include "velox/vector/VectorSaver.h"

DEFINE_string(
    input_path,
    "",
    "Path for vector to be restored from disk. This will enable single run "
    "of the fuzzer with the on-disk persisted repro information. This has to "
    "be set with sql_path and optionally result_path.");

DEFINE_string(
    sql_path,
    "",
    "Path for expression SQL to be restored from disk. This will enable "
    "single run of the fuzzer with the on-disk persisted repro information. "
    "This has to be set with input_path and optionally result_path.");

DEFINE_string(
    result_path,
    "",
    "Path for result vector to restore from disk. This is optional for "
    "on-disk reproduction. Don't set if the initial repro result vector is "
    "nullptr");

// TODO(jtan6): Support common mode and simplified mode.
DEFINE_string(
    mode,
    "verify",
    "Mode for expression runner: \n"
    "verify: run expression and verify between common and simplified path.\n"
    "common: run expression only in common path.\n"
    "simplified: run expression only in simplified path.");

using namespace facebook::velox;

int main(int argc, char** argv) {
  facebook::velox::functions::prestosql::registerAllScalarFunctions();

  ::testing::InitGoogleTest(&argc, argv);

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::init(&argc, &argv);

  std::shared_ptr<core::QueryCtx> queryCtx{core::QueryCtx::createForTest()};
  std::unique_ptr<memory::MemoryPool> pool{
      memory::getDefaultScopedMemoryPool()};
  core::ExecCtx execCtx{pool.get(), queryCtx.get()};

  VELOX_CHECK(!FLAGS_input_path.empty());
  VELOX_CHECK(!FLAGS_sql_path.empty());

  auto inputVector = std::dynamic_pointer_cast<RowVector>(
      restoreVectorFromFile(FLAGS_input_path.c_str(), pool.get()));
  VELOX_CHECK_NOT_NULL(inputVector, "Input vector is not a RowVector");
  auto sql = restoreStringFromFile(FLAGS_sql_path.c_str());

  parse::registerTypeResolver();
  parse::ParseOptions options;
  auto typedExpr = core::Expressions::inferTypes(
      parse::parseExpr(sql, options),
      inputVector->type(),
      &memory::getProcessDefaultMemoryManager().getRoot());
  VectorPtr resultVector;
  if (!FLAGS_result_path.empty()) {
    resultVector = restoreVectorFromFile(FLAGS_result_path.c_str(), pool.get());
  }

  if (FLAGS_mode == "verify") {
    test::ExpressionVerifier(&execCtx, {false, ""})
        .verify(typedExpr, inputVector, std::move(resultVector), true);
  } else {
    LOG(ERROR) << "Unknown expression runner mode '" << FLAGS_mode << "'.";
  }
}
