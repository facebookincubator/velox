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

#include "velox/exec/tests/utils/Cursor.h"
#include "velox/exec/tests/utils/Rebatch.h"

namespace facebook::velox::exec::test {

/// Reruns the Task of 'params' with plan fuzzing against splits recorded in
/// 'referenceTask'. Checks the results of each fuzzed plan against 'results'.
void checkFuzzedPlans(
    const CursorParameters& params,
    const std::vector<RowVectorPtr>& result,
    const std::shared_ptr<Task>& referencetask);

core::PlanNodePtr fuzzPlan(core::PlanNodePtr input, SelectivityVector& nodeSet);

} // namespace facebook::velox::exec::test
