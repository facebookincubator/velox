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

namespace facebook::velox::cxl {

/// Tag identifying the CXL memory resource: used to register it with
/// memory::CustomMemoryResourceRegistry and to build a per-query pool via
/// MemoryManager::addCustomRootPool. Set this as the value of the
/// 'relocation_resource_tag' query config to have an operator relocate its
/// payload here; core resolves the tag generically and never references this
/// symbol.
inline constexpr std::string_view kCxlResourceTag{"cxl"};

} // namespace facebook::velox::cxl
