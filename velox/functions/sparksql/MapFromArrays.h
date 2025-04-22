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
#include <vector>

#include "velox/expression/Expr.h"
#include "velox/functions/lib/Map.h"

namespace facebook::velox::functions::sparksql {

std::shared_ptr<exec::VectorFunction> makeMapFromArrays(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

std::vector<std::shared_ptr<exec::FunctionSignature>>
getMapFromArraysSignature();

void registerMapFromArrays(const std::string& name);

} // namespace facebook::velox::functions::sparksql
