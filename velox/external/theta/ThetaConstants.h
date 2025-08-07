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

// Adapted from Apache DataSketches

#pragma once

#include <climits>
#include "CommonDefs.h"

namespace facebook::velox::common::theta {

/// Theta constants
namespace ThetaConstants {
/// hash table resize factor
using resizeFactor = facebook::velox::common::theta::resizeFactor;
/// default resize factor
const resizeFactor DEFAULT_RESIZE_FACTOR = resizeFactor::X8;

/// max theta - signed max for compatibility with Java
const uint64_t MAX_THETA = LLONG_MAX;
/// min log2 of K
const uint8_t MIN_LG_K = 5;
/// max log2 of K
const uint8_t MAX_LG_K = 26;
/// default log2 of K
const uint8_t DEFAULT_LG_K = 12;
} // namespace ThetaConstants

} // namespace facebook::velox::common::theta
