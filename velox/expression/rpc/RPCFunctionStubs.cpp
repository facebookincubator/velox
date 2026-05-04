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

#include "velox/expression/rpc/RPCFunctionStubs.h"

#include <glog/logging.h>

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::exec::rpc {
namespace {

/// A stub VectorFunction that throws on execution.
/// This exists only to register the function's signature with the Velox
/// function registry so the sidecar can discover it via /v1/functions.
/// The Java planner rewrites RPC function calls to RPCNode before they
/// reach the executor, so this stub should never actually be invoked.
class RPCStubFunction : public exec::VectorFunction {
 public:
  explicit RPCStubFunction(const std::string& name) : name_(name) {}

  void apply(
      const SelectivityVector& /*rows*/,
      std::vector<VectorPtr>& /*args*/,
      const TypePtr& /*outputType*/,
      exec::EvalCtx& /*context*/,
      VectorPtr& /*result*/) const override {
    VELOX_FAIL(
        "RPC function '{}' should not be called directly. "
        "The query plan should have been rewritten by RpcFunctionOptimizer "
        "to use RPCNode/RPCOperator.",
        name_);
  }

 private:
  std::string name_;
};

} // namespace

void registerRPCFunctionStub(
    const std::string& name,
    std::vector<std::shared_ptr<exec::FunctionSignature>> signatures) {
  LOG(INFO) << "[RPC] registerRPCFunctionStub: registering Velox stub '" << name
            << "' with " << signatures.size() << " signature(s)";
  exec::registerStatefulVectorFunction(
      name,
      std::move(signatures),
      [name](
          const std::string& /*name*/,
          const std::vector<exec::VectorFunctionArg>& /*inputArgs*/,
          const core::QueryConfig& /*config*/) {
        return std::make_shared<RPCStubFunction>(name);
      });
  LOG(INFO) << "[RPC] registerRPCFunctionStub: successfully registered '"
            << name << "' in Velox function registry";
}

} // namespace facebook::velox::exec::rpc
