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

#include "velox/expression/VectorFunction.h"

namespace facebook::velox::functions {

std::vector<std::shared_ptr<exec::FunctionSignature>>
arrayShuffleWithRandomSeedSignatures();

std::vector<std::shared_ptr<exec::FunctionSignature>>
arrayShuffleWithCustomSeedSignatures();

// This function returns metadata with 'deterministic' as false, it is used
// with 'makeArrayShufflewithRandomSeed'.
exec::VectorFunctionMetadata getMetadataForArrayShuffleWithRandomSeed();

// Shuffle with rand seed.
std::shared_ptr<exec::VectorFunction> makeArrayShuffleWithRandomSeed(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

// This function returns metadata with 'deterministic' as true, it is used with
// 'makeArrayShuffleWithCustomSeed'.
exec::VectorFunctionMetadata getMetadataForArrayShuffleWithCustomSeed();

// Shuffle with custom seed (Spark's behavior).
std::shared_ptr<exec::VectorFunction> makeArrayShuffleWithCustomSeed(
    const std::string& name,
    const std::vector<exec::VectorFunctionArg>& inputArgs,
    const core::QueryConfig& config);

} // namespace facebook::velox::functions
