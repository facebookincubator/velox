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

#include <string_view>

namespace facebook::velox::functions {

/// Registers `name` for (time, interval year to month) and (interval year to
/// month, time). A year-month interval cannot change a time of day, so the
/// registered vector function returns its TIME argument unchanged and avoids
/// copying it where the input vector can be reused.
void registerTimePlusIntervalYearMonth(
    std::string_view name,
    std::string_view defaultOwner);

/// Registers `name` for (time, interval year to month) only. Presto defines
/// no subtraction with the interval on the left.
void registerTimeMinusIntervalYearMonth(
    std::string_view name,
    std::string_view defaultOwner);

} // namespace facebook::velox::functions
