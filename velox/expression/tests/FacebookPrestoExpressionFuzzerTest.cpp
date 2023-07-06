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

#include <folly/String.h>
#include <folly/init/Init.h>
#include <gtest/gtest.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "velox/expression/tests/FuzzerRunner.h"
#include "velox/functions/facebook/prestosql/Register.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"

DEFINE_int64(
    seed,
    123456,
    "Initial seed for random number generator "
    "(use it to reproduce previous results).");

DEFINE_string(
    only,
    "",
    "If specified, Fuzzer will only choose functions from "
    "this comma separated list of function names "
    "(e.g: --only \"split\" or --only \"substr,ltrim\").");

DEFINE_string(
    special_forms,
    "and,or,cast,coalesce,if,switch",
    "Comma-separated list of special forms to use in generated expression. "
    "Supported special forms: and, or, coalesce, if, switch, cast.");

int main(int argc, char** argv) {
  facebook::velox::functions::prestosql::registerAllScalarFacebookOnlyFunctions(
      "");
  facebook::velox::functions::prestosql::registerAllScalarFunctions();

  ::testing::InitGoogleTest(&argc, argv);

  // Calls common init functions in the necessary order, initializing
  // singletons, installing proper signal handlers for better debugging
  // experience, and initialize glog and gflags.
  folly::init(&argc, &argv);

  // The following list are the Spark UDFs that hit issues
  // For rlike you need the following combo in the only list:
  // rlike, md5 and upper

  // TODO: List of the functions that at some point crash or fail and need to
  // be fixed before we can enable.
  std::unordered_set<std::string> skipFunctions = {
      // Fuzzer and the underlying engine are confused about cardinality(HLL)
      // (since HLL is a user defined type), and end up trying to use
      // cardinality passing a VARBINARY (since HLL's implementation uses an
      // alias to VARBINARY).
      "cardinality",
      "in",
      "element_at",
      "width_bucket",
      // Skip concat as it triggers a test failure due to an incorrect
      // expression generation from fuzzer:
      // https://github.com/facebookincubator/velox/issues/5398
      "concat",
  };

  return FuzzerRunner::run(
      FLAGS_only, FLAGS_seed, skipFunctions, FLAGS_special_forms);
}
