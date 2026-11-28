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

#include <cstddef>
#include <functional>
#include "velox/exec/TaskStats.h"
#include "velox/exec/fuzzer/ReferenceQueryRunner.h"

namespace facebook::velox::exec {
/// If 'taskStatsCallback' is set, it is called with the stats of every
/// successfully executed Velox task.
void joinFuzzer(
    size_t seed,
    std::unique_ptr<test::ReferenceQueryRunner> referenceQueryRunner,
    std::function<void(const TaskStats&)> taskStatsCallback = nullptr);
} // namespace facebook::velox::exec
