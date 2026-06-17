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

namespace facebook::velox::functions::sparksql {

/// Function name constants for the decimal rounding special forms.
inline constexpr const char* kRoundDecimal = "decimal_round";
inline constexpr const char* kCeilDecimal = "decimal_ceil";
inline constexpr const char* kFloorDecimal = "decimal_floor";

/// Registers decimal_round, decimal_ceil, and decimal_floor special forms.
void registerDecimalRoundingForms();

} // namespace facebook::velox::functions::sparksql
