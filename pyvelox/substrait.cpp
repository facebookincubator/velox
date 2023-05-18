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

#include "substrait.h" // @manual
#include "context.h"
#include "signatures.h" /// TODO: remove
#include <google/protobuf/util/json_util.h>
#include <fstream>
#include <sstream>
#include "velox/common/base/Exceptions.h"
#include <velox/substrait/SubstraitExecutor.h>
#include <velox/exec/tests/utils/AssertQueryBuilder.h>
#include <velox/functions/prestosql/registration/RegistrationFunctions.h>
#include "velox/functions/prestosql/aggregates/RegisterAggregateFunctions.h"
#include <iostream>


namespace facebook::velox::py {

namespace py = pybind11;

static void readFromFile(
      const std::string& msgPath,
      google::protobuf::Message& msg) {
    // Read json file and resume the Substrait plan.
  std::ifstream msgJson(msgPath);
  VELOX_CHECK(
      !msgJson.fail(), "Failed to open file: {}. {}", msgPath, strerror(errno));
  std::stringstream buffer;
  buffer << msgJson.rdbuf();
  std::string msgData = buffer.str();
  auto status = google::protobuf::util::JsonStringToMessage(msgData, &msg);
  VELOX_CHECK(
      status.ok(),
      "Failed to parse Substrait JSON: {} {}",
      status.code(),
      status.message());
}

// RowVectorPtr runSubstraitQuery(const std::string& planPath) {
//   memory::MemoryPool* pool = PyVeloxContext::getInstance().pool();

//   std::shared_ptr<facebook::velox::substrait::SubstraitVeloxPlanConverter> planConverter =
//       std::make_shared<facebook::velox::substrait::SubstraitVeloxPlanConverter>(pool);

//   ::substrait::Plan substraitPlan;
//   readFromFile(planPath, substraitPlan);

//   auto veloxPlan = planConverter->toVeloxPlan(substraitPlan);
//   auto fragment = std::make_shared<facebook::velox::core::PlanFragment>(veloxPlan);
//   std::shared_ptr<folly::Executor> executor(std::make_shared<folly::CPUThreadPoolExecutor>(1));
//   auto queryCtx = std::make_shared<facebook::velox::core::QueryCtx>(executor.get());
//   auto substrait_task = facebook::velox::exec::Task::create(
//           "0", *fragment, 0, std::move(queryCtx));

//   substrait_task->noMoreSplits("0");
  
//   auto result = substrait_task->next();

//   while (auto tmp = substrait_task->next()) {
//   }
//   std::chrono::microseconds execution_timeout(100);
//   while (!substrait_task->isFinished()) {
//     auto& inline_executor = folly::QueuedImmediateExecutor::instance();
//     auto task_future =
//         substrait_task->stateChangeFuture(execution_timeout.count()).via(&inline_executor);
//     task_future.wait();
//   }
//   return result;
// }

static inline void initializeSubstrait() {
  // auto connector = connector::getConnectorFactory(
  //         connector::hive::HiveConnectorFactory::kHiveConnectorName)
  //         ->newConnector("test-hive", nullptr);
  // facebook::velox::connector::registerConnector(connector);
  PySubstraitContext::getInstance().initialize();
}

static inline void finalizeSubstrait() {
  PySubstraitContext::getInstance().finalize();
}

static inline RowVectorPtr runSubstraitQuery(const std::string& planPath) {
  memory::MemoryPool* pool = PyVeloxContext::getInstance().pool();
  auto hiveConnector =
      connector::getConnectorFactory(
          connector::hive::HiveConnectorFactory::kHiveConnectorName)
          ->newConnector("test-hive", nullptr);
  connector::registerConnector(hiveConnector);
  //PySubstraitContext::initialize();
  /// TODO: wrap this in a struct and see if we get the calling pure virtual function error
  /// Here we need a initialize function to call the register and destructor to call the unregister
  /// then we can register the connector properly. This would require a class. May be do this in the
  /// constructor and destructor of the converter in Substrait API.
  
  std::shared_ptr<facebook::velox::substrait::SubstraitVeloxPlanConverter> planConverter =
      std::make_shared<facebook::velox::substrait::SubstraitVeloxPlanConverter>(pool);

  ::substrait::Plan substraitPlan;
  readFromFile(planPath, substraitPlan);

  auto veloxPlan = planConverter->toVeloxPlan(substraitPlan);
  //connector::unregisterConnector("test-hive");
  return facebook::velox::exec::test::AssertQueryBuilder(veloxPlan).copyResults(pool);
}

// static inline VectorPtr runSubstraitQueryByFile(const std::string& planPath) {
//   return facebook::velox::substrait::RunQueryByFile(planPath);
// }


void addSubstraitBindings(py::module& m, bool asModuleLocalDefinitions) {
  using namespace facebook::velox;
  functions::prestosql::registerAllScalarFunctions();
  aggregate::prestosql::registerAllAggregateFunctions();
  m.def(
      "initialize_substrait",
      &initializeSubstrait,
      "Initializes the modules required for Substrait.");
  m.def(
      "run_substrait_query",
      &runSubstraitQuery,
      "Runs a Substrait query and return output.");
  m.def(
      "finalize_substrait",
      &finalizeSubstrait,
      "Finalizes the modules required for Substrait.");

}
} // namespace facebook::velox::py
