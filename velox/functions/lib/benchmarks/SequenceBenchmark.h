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

#include <initializer_list>
#include <string_view>

namespace facebook::velox::functions::test {
class FunctionBenchmarkBase;

/// Registers sequence benchmarks for the types supported by one dialect.
/// The caller registers sequence and keeps benchmark alive during execution.
void addSequenceBenchmarks(
    const char* file,
    FunctionBenchmarkBase& benchmark,
    std::string_view dialect,
    std::initializer_list<std::string_view> types);
} // namespace facebook::velox::functions::test
