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

// GPU shadow for velox/type/Type.h, which nvcc's front end rejects: its
// constexpr type singletons are not literal types to it, and it reaches
// folly's serializer. A device translation unit holds the TypePtrs
// initialize() receives but never reads one, so the declaration
// SimpleFunctionTags.h makes is enough; what remains is the time constants
// simple function bodies read, copied from the real header.
#pragma once

#include "velox/type/SimpleFunctionApi.h"
#include "velox/type/SimpleFunctionTags.h"
#include "velox/type/TypeKind.h"

namespace facebook::velox {

constexpr long kMillisInSecond = 1000;
constexpr long kMillisInMinute = 60 * kMillisInSecond;
constexpr long kMillisInHour = 60 * kMillisInMinute;
constexpr long kMillisInDay = 24 * kMillisInHour;

} // namespace facebook::velox
