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

#include "velox/substrait/SubstraitExecutor.h"
//#include "velox/exec/tests/utils/AssertQueryBuilder.h"

#include <fstream>
#include <sstream>
#include "velox/common/base/Exceptions.h"

namespace facebook::velox::substrait {
    // void readFromFile(
    //     const std::string& msgPath,
    //     google::protobuf::Message& msg) {
    // // Read json file and resume the Substrait plan.
    // std::ifstream msgJson(msgPath);
    // VELOX_CHECK(
    //     !msgJson.fail(), "Failed to open file: {}. {}", msgPath, strerror(errno));
    // std::stringstream buffer;
    // buffer << msgJson.rdbuf();
    // std::string msgData = buffer.str();
    // auto status = google::protobuf::util::JsonStringToMessage(msgData, &msg);
    // VELOX_CHECK(
    //     status.ok(),
    //     "Failed to parse Substrait JSON: {} {}",
    //     status.code(),
    //     status.message());
    // }

    // VectorPtr RunQueryByFile(const std::string& planPath) {
    //     std::shared_ptr<memory::MemoryPool> pool = memory::addDefaultLeafMemoryPool();
    //     std::shared_ptr<facebook::velox::substrait::SubstraitVeloxPlanConverter> planConverter =
    //     std::make_shared<facebook::velox::substrait::SubstraitVeloxPlanConverter>(pool.get());

    //     ::substrait::Plan substraitPlan;
    //     readFromFile(planPath, substraitPlan);

    //     auto veloxPlan = planConverter->toVeloxPlan(substraitPlan);
    //     auto fragment = std::make_shared<facebook::velox::core::PlanFragment>(veloxPlan);

    //     std::shared_ptr<folly::Executor> executor(std::make_shared<folly::CPUThreadPoolExecutor>(1));

    //     auto substrait_task = std::make_shared<facebook::velox::exec::Task>(
    //         "substrait_task", *fragment, 0,
    //         std::make_shared<facebook::velox::core::QueryCtx>(executor.get()));

    //     auto result = substrait_task->next();

    //     while (auto tmp = substrait_task->next()) {
    //     }
    //     return result->childAt(0);
    // }

    // VectorPtr RunQueryByFileV2(const std::string& planPath) {
    //     return 
    // }

    // RowVectorPtr RunQueryByFileV1(const std::string& planPath) {
    //     std::shared_ptr<memory::MemoryPool> pool = memory::getDefaultMemoryPool();
    //     std::shared_ptr<facebook::velox::substrait::SubstraitVeloxPlanConverter> planConverter =
    //     std::make_shared<facebook::velox::substrait::SubstraitVeloxPlanConverter>(pool.get());

    //     ::substrait::Plan substraitPlan;
    //     readFromFile(planPath, substraitPlan);

    //     auto planNode = planConverter->toVeloxPlan(substraitPlan);
    //     return facebook::velox::exec::test::AssertQueryBuilder(planNode).copyResults(pool.get());
    // }
}