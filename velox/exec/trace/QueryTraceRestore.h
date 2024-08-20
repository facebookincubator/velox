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

#include "velox/core/PlanNode.h"

namespace facebook::velox::exec {

core::PlanNodePtr findPlanNodeById(
    const core::PlanNodePtr& planNode,
    const std::string& id);

std::function<core::PlanNodePtr(std::string, core::PlanNodePtr)> addTableWriter(
    const std::shared_ptr<const core::TableWriteNode>& node);
} // namespace facebook::velox::exec
