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

#include <fmt/format.h>
#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "velox/common/base/Fs.h"
#include "velox/exec/fuzzer/DuckQueryRunner.h"
#include "velox/exec/fuzzer/WindowFuzzerRunner.h"
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include "velox/functions/prestosql/window/WindowFunctionsRegistration.h"

using namespace facebook::velox;

DEFINE_string(
    plan_nodes_path,
    "",
    "Path for plan nodes to be restored from disk. This will enable single run "
    "of the fuzzer with the on-disk persisted plan information.");

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  folly::Init init(&argc, &argv);

  facebook::velox::aggregate::prestosql::registerAllAggregateFunctions(
      "", false);
  facebook::velox::window::prestosql::registerAllWindowFunctions();

  auto duckQueryRunner =
      std::make_unique<facebook::velox::exec::test::DuckQueryRunner>();
  duckQueryRunner->disableAggregateFunctions({
      "skewness",
      // DuckDB results on constant inputs are incorrect. Should be NaN,
      // but DuckDB returns some random value.
      "kurtosis",
      "entropy",
  });

  return exec::test::WindowFuzzerRunner::runRepro(
      FLAGS_plan_nodes_path, std::move(duckQueryRunner));
}
