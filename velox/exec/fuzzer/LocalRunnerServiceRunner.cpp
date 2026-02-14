/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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
#include <gflags/gflags.h>
#include <gtest/gtest.h>
#include <thrift/lib/cpp2/server/ThriftServer.h>

#include "velox/core/ITypedExpr.h"
#include "velox/core/PlanNode.h"
#include "velox/exec/fuzzer/LocalRunnerService.h"
#include "velox/functions/prestosql/registration/RegistrationFunctions.h"
#include "velox/type/Type.h"

using namespace facebook::velox;
using namespace facebook::velox::runner;

DEFINE_int32(
    port,
    9091,
    "LocalRunnerService port number to be used in conjunction with ExpressionFuzzerTest flag local_runner_port.");

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);

  folly::Init init(&argc, &argv);

  memory::initializeMemoryManager(memory::MemoryManager::Options{});
  Type::registerSerDe();
  core::PlanNode::registerSerDe();
  core::ITypedExpr::registerSerDe();
  functions::prestosql::registerAllScalarFunctions();
  functions::prestosql::registerInternalFunctions();

  std::shared_ptr<apache::thrift::ThriftServer> thriftServer =
      std::make_shared<apache::thrift::ThriftServer>();
  thriftServer->setPort(FLAGS_port);
  thriftServer->setInterface(std::make_shared<LocalRunnerServiceHandler>());
  thriftServer->setNumIOWorkerThreads(1);
  thriftServer->setNumCPUWorkerThreads(1);

  VLOG(1) << "Starting LocalRunnerService";
  thriftServer->serve();

  return 0;
}
