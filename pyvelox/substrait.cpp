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
#include <google/protobuf/util/json_util.h>
#include <fstream>
#include <sstream>
#include "velox/common/base/Exceptions.h"
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

void runSubstraitQuery(const std::string& plan) {
  std::shared_ptr<memory::MemoryPool> pool = memory::getDefaultMemoryPool();
  std::shared_ptr<facebook::velox::substrait::SubstraitVeloxPlanConverter> planConverter =
      std::make_shared<facebook::velox::substrait::SubstraitVeloxPlanConverter>(pool.get());
  auto planPath = "/home/asus/github/fork/velox/velox/substrait/tests/data/substrait_virtualTable.json";

  ::substrait::Plan substraitPlan;
  readFromFile(planPath, substraitPlan);

  auto veloxPlan = planConverter->toVeloxPlan(substraitPlan);
  auto fragment = std::make_shared<facebook::velox::core::PlanFragment>(veloxPlan);
  facebook::velox::exec::Consumer consumer =
      [&](facebook::velox::RowVectorPtr output,
          facebook::velox::ContinueFuture*) -> facebook::velox::exec::BlockingReason {
    if (output) {
      std::cout << output->toString() << std::endl;
      std::cout << output->toString(0, 10) << std::endl;
    }
    return facebook::velox::exec::BlockingReason::kNotBlocked;
  };
  std::shared_ptr<folly::Executor> executor(
      std::make_shared<folly::CPUThreadPoolExecutor>(1));

  auto substrait_task = std::make_shared<facebook::velox::exec::Task>(
      "substrait_task", *fragment, 0,
      std::make_shared<facebook::velox::core::QueryCtx>(executor.get()), consumer);

  facebook::velox::exec::Task::start(substrait_task, 1);

  while (substrait_task->isRunning()) {
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}


void addSubstraitBindings(py::module& m, bool asModuleLocalDefinitions) {

  m.def(
      "run_substrait_query",
      &runSubstraitQuery,
      "Runs a Substrait query and return output.",
      py::arg("plan") = "");

}
} // namespace facebook::velox::py
