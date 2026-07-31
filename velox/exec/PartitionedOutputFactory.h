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

#include <cstdint>
#include <functional>
#include <memory>

namespace facebook::velox::core {
class PartitionedOutputNode;
} // namespace facebook::velox::core

namespace facebook::velox::exec {

struct DriverCtx;
class Operator;

/// Builds a pipeline's output operator (e.g. PartitionedOutput), bound to the
/// matching output buffer manager. The two are registered together in
/// OutputTransportRegistry so they cannot diverge.
using PartitionedOutputFactory = std::function<std::unique_ptr<Operator>(
    int32_t operatorId,
    DriverCtx* ctx,
    const std::shared_ptr<const core::PartitionedOutputNode>& node,
    bool eagerFlush)>;

} // namespace facebook::velox::exec
