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

namespace facebook::velox::exec {

/// Registers the operator translator for StateSourceNode and StateHashJoinNode,
/// which lets a fixed point's per-iteration sub-task plans read its state.
/// Call once at startup.  Not idempotent -- pair it with
/// Operator::unregisterAllOperators() in test teardown.  Nothing registers the
/// executor itself: Task::create builds one directly for a plan containing a
/// FixedPointNode.
void registerFixedPoint();

} // namespace facebook::velox::exec
