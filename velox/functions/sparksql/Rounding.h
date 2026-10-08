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

#include <string>

namespace facebook::velox::functions::sparksql {

/// Identifies the integral ROUND integration contract (two or three arguments).
inline constexpr const char* kSparkRound = "spark_round";

/// Registers round and its integral and decimal integration names with a
/// prefix. Floating-point round retains the existing binary arithmetic
/// implementation.
void registerRoundFunctions(const std::string& prefix);

} // namespace facebook::velox::functions::sparksql
