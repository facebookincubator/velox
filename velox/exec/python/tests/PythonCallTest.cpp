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
#include <folly/FileUtil.h>
#include <folly/String.h>
#include <folly/Subprocess.h>

#include "velox/common/base/tests/GTestUtils.h"
#include "velox/common/testutil/TempFilePath.h"
#include "velox/exec/PlanNodeStats.h"
#include "velox/exec/PythonCallStatNames.h"
#include "velox/exec/python/PythonEnvironmentResolver.h"
#include "velox/exec/tests/utils/AssertQueryBuilder.h"
#include "velox/exec/tests/utils/OperatorTestBase.h"
#include "velox/exec/tests/utils/PlanBuilder.h"

namespace facebook::velox::exec::python {
namespace {

using namespace facebook::velox::exec::test;

// The interpreter the tests run workers with: a program with pyarrow that acts
// as a Python interpreter.
std::string interpreterPath() {
  const char* path = std::getenv("PYTHON_INTERPRETER");
  VELOX_CHECK_NOT_NULL(path, "PYTHON_INTERPRETER must name an interpreter");
  return path;
}

std::string interpreterUri() {
  return fmt::format("file://{}", interpreterPath());
}

// Runs the test interpreter to find its Python version, such as 3.12. Plans
// must name that version: the worker refuses functions pickled with another.
std::string interpreterVersion() {
  const auto script = common::testutil::TempFilePath::create();
  VELOX_CHECK(
      folly::writeFile(
          std::string_view{
              "import sys\n"
              "print(f'{sys.version_info.major}.{sys.version_info.minor}')\n"},
          script->getPath().c_str()));
  folly::Subprocess process(
      std::vector<std::string>{interpreterPath(), script->getPath()},
      folly::Subprocess::Options().pipeStdout());
  const auto output = process.communicate().first;
  VELOX_CHECK(process.wait().exited());
  return folly::trimWhitespace(output).str();
}

// The functions the tests can call, by name: pickles of standard library
// functions, which unpickle in any Python.
const folly::F14FastMap<std::string, PlanBuilder::PythonFunction>& registry() {
  static const folly::F14FastMap<std::string, PlanBuilder::PythonFunction>
      kRegistry{
          {"len", {"cbuiltins\nlen\n.", BIGINT()}},
          {"str", {"cbuiltins\nstr\n.", VARCHAR()}},
          {"truediv", {"coperator\ntruediv\n.", DOUBLE()}},
          {"dict", {"cbuiltins\ndict\n.", MAP(REAL(), DOUBLE())}},
          {"exit", {"cos\n_exit\n.", BIGINT()}},
          {"list", {"cbuiltins\nlist\n.", ARRAY(MAP(REAL(), DOUBLE()))}},
      };
  return kRegistry;
}

class PythonCallTest : public OperatorTestBase {
 protected:
  static void SetUpTestCase() {
    OperatorTestBase::SetUpTestCase();
    pythonVersion_ = interpreterVersion();
  }

  // Returns a PlanBuilder that runs Python calls with the test interpreter.
  static PlanBuilder startPlan() {
    PlanBuilder builder;
    builder.setPythonEnvironment(interpreterUri(), pythonVersion_);
    return builder;
  }

  // Returns an AssertQueryBuilder for 'plan' whose query resolves file://
  // environments.
  AssertQueryBuilder makeQuery(const core::PlanNodePtr& plan) {
    auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
    auto resolvers = std::make_shared<PythonEnvironmentResolvers>();
    resolvers->insert("file", std::make_shared<FileEnvironmentResolver>());
    queryCtx->setRegistry(PythonEnvironmentResolvers::kRegistryKey, resolvers);
    AssertQueryBuilder builder(plan);
    builder.queryCtx(queryCtx);
    return builder;
  }

  static std::string pythonVersion_;
};

std::string PythonCallTest::pythonVersion_;

TEST_F(PythonCallTest, basic) {
  const std::vector<RowVectorPtr> input{
      makeRowVector({makeFlatVector<std::string>({"a", "bb"})}),
      makeRowVector({makeFlatVector<std::string>({"ccc"})}),
  };
  const auto plan = startPlan()
                        .values(input)
                        .pythonCall(registry(), {"len(c0) AS n"})
                        .planNode();
  const auto task = makeQuery(plan).assertResults(makeRowVector({
      makeFlatVector<std::string>({"a", "bb", "ccc"}),
      makeFlatVector<int64_t>({1, 2, 3}),
  }));

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(stats.customCount(PythonCallStatNames::kFunctionWallNanos), 2);
  EXPECT_GT(stats.customSum(PythonCallStatNames::kUnpickleWallNanos), 0);
  EXPECT_GT(stats.customSum(PythonCallStatNames::kWorkerCpuNanos), 0);
  EXPECT_GT(stats.customSum(PythonCallStatNames::kWorkerPeakMemoryBytes), 0);
}

TEST_F(PythonCallTest, emptyInput) {
  const auto plan =
      startPlan()
          .values(makeRowVector({makeFlatVector<std::string>({})}))
          .pythonCall(registry(), {"len(c0)"})
          .planNode();
  const auto task = makeQuery(plan).assertResults(makeRowVector({
      makeFlatVector<std::string>({}),
      makeFlatVector<int64_t>({}),
  }));

  // The worker reports its stats though it ran no batch.
  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_GT(stats.customSum(PythonCallStatNames::kWorkerPeakMemoryBytes), 0);
}

TEST_F(PythonCallTest, manyBatches) {
  std::vector<RowVectorPtr> input;
  for (int64_t i = 0; i < 10; ++i) {
    input.push_back(makeRowVector({makeFlatVector<int64_t>({i})}));
  }
  const auto plan = startPlan()
                        .values(input)
                        .pythonCall(registry(), {"str(c0)"}, {"p0"})
                        .planNode();
  const auto expected = makeRowVector({makeFlatVector<std::string>(
      {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"})});

  for (const auto maxPending : {"1", "2", "4"}) {
    SCOPED_TRACE(maxPending);
    makeQuery(plan)
        .config(core::QueryConfig::kPythonCallMaxPendingBatches, maxPending)
        .assertResults(expected);
  }
}

TEST_F(PythonCallTest, multipleDrivers) {
  // Each of the 4 drivers reads all of the input and runs its own worker.
  const auto input = makeRowVector({makeFlatVector<int64_t>({1, 22})});
  const auto plan = startPlan()
                        .values({input}, /*parallelizable=*/true)
                        .pythonCall(registry(), {"str(c0)"})
                        .planNode();

  makeQuery(plan).maxDrivers(4).assertResults(makeRowVector({
      makeFlatVector<int64_t>({1, 22, 1, 22, 1, 22, 1, 22}),
      makeFlatVector<std::string>({"1", "22", "1", "22", "1", "22", "1", "22"}),
  }));
}

TEST_F(PythonCallTest, nestedAndSharedCalls) {
  // str(c0) runs once; len reads its results in the worker.
  const auto plan =
      startPlan()
          .values(makeRowVector({makeFlatVector<int64_t>({1, 22, 333})}))
          .pythonCall(registry(), {"str(c0) AS s", "len(str(c0)) AS n"})
          .planNode();

  const auto task = makeQuery(plan).assertResults(makeRowVector({
      makeFlatVector<int64_t>({1, 22, 333}),
      makeFlatVector<std::string>({"1", "22", "333"}),
      makeFlatVector<int64_t>({1, 2, 3}),
  }));

  const auto stats = PlanNodeStats::extract(task->taskStats(), plan->id());
  EXPECT_EQ(stats.customStat(PythonCallStatNames::kNumFunctions).max, 2);
}

TEST_F(PythonCallTest, prunedInputs) {
  // The output keeps c1 and drops c0, which only the call reads. The call has
  // no alias, so its column is p0.
  const auto plan = startPlan()
                        .values(makeRowVector({
                            makeFlatVector<std::string>({"a", "bb"}),
                            makeFlatVector<int32_t>({10, 20}),
                        }))
                        .pythonCall(registry(), {"len(c0)"}, {"p0", "c1"})
                        .planNode();

  makeQuery(plan).assertResults(makeRowVector({
      makeFlatVector<int64_t>({1, 2}),
      makeFlatVector<int32_t>({10, 20}),
  }));
}

TEST_F(PythonCallTest, failedRows) {
  const auto input = makeRowVector({
      makeFlatVector<double>({1.0, 1.0}),
      makeFlatVector<double>({2.0, 0.0}),
  });
  const auto makePlan = [&](const std::string& call) {
    return startPlan()
        .values(input)
        .pythonCall(registry(), {call}, {"q"})
        .planNode();
  };

  VELOX_ASSERT_USER_THROW(
      makeQuery(makePlan("truediv(c0, c1) AS q")).countResults(),
      // The message after the colon varies across Python versions.
      "truediv raised ZeroDivisionError: ");

  makeQuery(makePlan("try(truediv(c0, c1)) AS q"))
      .assertResults(
          makeRowVector({makeNullableFlatVector<double>({0.5, std::nullopt})}));
  // TRY covers the calls nested in its call.
  makeQuery(makePlan("try(str(truediv(c0, c1))) AS q"))
      .assertResults(makeRowVector(
          {makeNullableFlatVector<std::string>({"0.5", std::nullopt})}));
}

TEST_F(PythonCallTest, timestamps) {
  // 2025-01-02 03:04:05 UTC, 2025-01-01 19:04:05 in America/Los_Angeles.
  const auto plan =
      startPlan()
          .values(makeRowVector(
              {makeFlatVector<Timestamp>({Timestamp(1'735'787'045, 0)})}))
          .pythonCall(registry(), {"str(c0)"}, {"p0"})
          .planNode();
  const auto assertShown = [&](const std::string& adjust,
                               const std::string& shown) {
    makeQuery(plan)
        .config(core::QueryConfig::kSessionTimezone, "America/Los_Angeles")
        .config(core::QueryConfig::kAdjustTimestampToTimezone, adjust)
        .assertResults(makeRowVector({makeFlatVector<std::string>({shown})}));
  };

  // Without adjustment, timestamps are wall-clock values.
  assertShown("false", "2025-01-02 03:04:05");
  assertShown("true", "2025-01-01 19:04:05");
}

TEST_F(PythonCallTest, wrongResultType) {
  const auto plan =
      startPlan()
          .values(makeRowVector({makeFlatVector<int64_t>({1})}))
          .pythonCall(
              {{"str", {registry().at("str").pickledFunction, BIGINT()}}},
              {"str(c0) AS n"})
          .planNode();

  VELOX_ASSERT_USER_THROW(
      makeQuery(plan).countResults(), "str result must be BIGINT: str '1'");
}

TEST_F(PythonCallTest, duplicateMapKeys) {
  // dict() keeps 0.1 and 0.1000000001 as two keys; as REAL they are equal.
  const auto input = makeRowVector({makeNestedArrayVectorFromJson<double>({
      "[[0.1, 1.0], [0.1000000001, 2.0]]",
      "[[1.0, 3.0]]",
  })});
  const auto makePlan = [&](const std::string& call) {
    return startPlan()
        .values(input)
        .pythonCall(registry(), {call}, {"m"})
        .planNode();
  };

  VELOX_ASSERT_USER_THROW(
      makeQuery(makePlan("dict(c0) AS m")).countResults(),
      "Python function must not return a map with duplicate keys: dict");

  makeQuery(makePlan("try(dict(c0)) AS m"))
      .assertResults(makeRowVector({makeNullableMapVector<float, double>(
          {std::nullopt, {{{1.0f, 3.0}}}})}));
}

TEST_F(PythonCallTest, duplicateMapKeysUnderTryAndNot) {
  // The same call with and without TRY shares one result, and the row with
  // duplicate keys fails the call without TRY.
  const auto plan =
      startPlan()
          .values(makeRowVector({makeNestedArrayVectorFromJson<double>(
              {"[[0.1, 1.0], [0.1000000001, 2.0]]"})}))
          .pythonCall(registry(), {"try(dict(c0)) AS a", "dict(c0) AS b"})
          .planNode();

  VELOX_ASSERT_USER_THROW(
      makeQuery(plan).countResults(),
      "Python function must not return a map with duplicate keys: dict");
}

TEST_F(PythonCallTest, nestedDuplicateMapKeys) {
  // list() returns the maps as they are, with keys that become equal as REAL.
  const auto input = makeRowVector({makeArrayVector(
      {0, 1},
      makeMapVectorFromJson<double, double>(
          {"{0.1: 1.0, 0.1000000001: 2.0}", "{1.0: 3.0}"}))});
  const auto makePlan = [&](const std::string& call) {
    return startPlan()
        .values(input)
        .pythonCall(registry(), {call}, {"m"})
        .planNode();
  };

  VELOX_ASSERT_USER_THROW(
      makeQuery(makePlan("list(c0) AS m")).countResults(),
      "Python function must not return a map with duplicate keys: list");
  makeQuery(makePlan("try(list(c0)) AS m"))
      .assertResults(makeRowVector({makeArrayVector(
          {0, 0}, makeMapVectorFromJson<float, double>({"{1.0: 3.0}"}), {0})}));
}

TEST_F(PythonCallTest, workerFailures) {
  const auto run = [&](std::string environmentUri, std::string pythonVersion) {
    return makeQuery(
               PlanBuilder()
                   .values(makeRowVector({makeFlatVector<std::string>({"a"})}))
                   .setPythonEnvironment(
                       std::move(environmentUri), std::move(pythonVersion))
                   .pythonCall(registry(), {"len(c0) AS n"})
                   .planNode())
        .countResults();
  };

  VELOX_ASSERT_USER_THROW(
      run("unknown://env", pythonVersion_),
      "No resolver for Python environment: unknown");
  VELOX_ASSERT_USER_THROW(
      run("file:///no/such/python", pythonVersion_),
      "Python interpreter must be an executable file: /no/such/python");
  VELOX_ASSERT_USER_THROW(
      run(interpreterUri(), "2.7"),
      fmt::format(
          "Functions must be pickled with the worker's Python {}: 2.7",
          pythonVersion_));
}

TEST_F(PythonCallTest, queryResolvers) {
  // Resolves any URI of its scheme to the test interpreter.
  class TestResolver : public PythonEnvironmentResolver {
   public:
    PythonInterpreter resolveUri(std::string_view /*uri*/) const override {
      return PythonInterpreter{.path = interpreterPath()};
    }
  };

  // A query resolves only the schemes its registry has.
  auto resolvers = std::make_shared<PythonEnvironmentResolvers>();
  resolvers->insert("test", std::make_shared<TestResolver>());
  auto queryCtx = core::QueryCtx::create(driverExecutor_.get());
  queryCtx->setRegistry(PythonEnvironmentResolvers::kRegistryKey, resolvers);
  const auto makePlan = [&](const std::string& environmentUri) {
    return PlanBuilder()
        .values(makeRowVector({makeFlatVector<std::string>({"a"})}))
        .setPythonEnvironment(environmentUri, pythonVersion_)
        .pythonCall(registry(), {"len(c0)"}, {"p0"})
        .planNode();
  };

  makeQuery(makePlan("test://env"))
      .queryCtx(queryCtx)
      .assertResults(makeRowVector({makeFlatVector<int64_t>({1})}));
  VELOX_ASSERT_USER_THROW(
      makeQuery(makePlan(interpreterUri())).queryCtx(queryCtx).countResults(),
      "No resolver for Python environment: file");
  VELOX_ASSERT_THROW(
      AssertQueryBuilder(makePlan("test://env")).countResults(),
      "QueryCtx has no Python environment resolvers");
}

TEST_F(PythonCallTest, workerFailsMidQuery) {
  // os._exit ends the worker without an ERROR.
  VELOX_ASSERT_USER_THROW(
      makeQuery(
          startPlan()
              .values(makeRowVector({makeFlatVector<int64_t>({3})}))
              .pythonCall(registry(), {"exit(c0)"})
              .planNode())
          .countResults(),
      "Python worker exited without reporting an error");
}

} // namespace
} // namespace facebook::velox::exec::python
