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

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::exec::rpc {

/// Registers a stub Velox vector function for an RPC function.
/// The stub has the correct signature but throws on direct execution.
/// This makes the function discoverable via the /v1/functions sidecar endpoint.
///
/// @param name Full 3-part Velox name (e.g., "native.rpc.fb_llm_inference")
/// @param signatures Function signatures (arg types, return type)
void registerRPCFunctionStub(
    const std::string& name,
    std::vector<std::shared_ptr<exec::FunctionSignature>> signatures);

} // namespace facebook::velox::exec::rpc
