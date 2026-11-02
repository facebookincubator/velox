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

/// Registers Spark's map_from_entries under the given name. It builds a map
/// from an array of key-value structs. Entries keep their input order, and a
/// repeated key keeps its first position and its last value, or is an error
/// under the exception policy. A null entry makes the map null, and a null key
/// is an error.
void registerMapFromEntriesFunction(const std::string& name);

} // namespace facebook::velox::functions::sparksql
