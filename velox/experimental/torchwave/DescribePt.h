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
#include <vector>

namespace torch::wave {

/// Registers a NamedTuple type so the pickle reader can deserialize it.
/// 'qualifiedName' is the fully qualified Python type name (e.g.
/// "module.ClassName"). 'fieldNames' lists fields in declaration order.
void registerNamedTuple(
    const std::string& qualifiedName,
    std::vector<std::string> fieldNames);

/// Reads a .pt file and prints the position, name, shape, and dtype of each
/// serialized tensor. Supports NamedTuple types registered via
/// registerNamedTuple.
void describePt(const std::string& path);

} // namespace torch::wave
