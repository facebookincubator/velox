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

/// Registers get_array_item(array(E), index) -> E, Spark's array[index]: an
/// element of the array at the given 0-based index. A negative or
/// out-of-bounds index returns NULL, or throws when Spark ANSI mode is enabled
/// in the query config. Spark's get(array, index) never throws and is
/// registered separately as get.
void registerGetArrayItemFunction(const std::string& name);

} // namespace facebook::velox::functions::sparksql
