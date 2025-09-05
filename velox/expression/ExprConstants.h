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

/// This file contains string literals for function names.
/// These literals listed in single place to explicitly list functions that
/// names used once even in case when user specify some prefix on registration.

namespace facebook::velox::expression {

/// These functions registered without prefix and velox code rely on this.
inline constexpr const char* kAnd = "and";
inline constexpr const char* kOr = "or";
inline constexpr const char* kSwitch = "switch";
inline constexpr const char* kIf = "if";
#ifndef VELOX_ENABLE_BACKWARD_COMPATIBILITY
inline constexpr const char* kFail = "fail";
#endif
inline constexpr const char* kCoalesce = "coalesce";
inline constexpr const char* kCast = "cast";
inline constexpr const char* kTryCast = "try_cast";
inline constexpr const char* kTry = "try";
inline constexpr const char* kRowConstructor = "row_constructor";

/// TODO: These two functions are actually registered without prefix.
/// But we think it's bug and should be changed.
namespace old {

inline constexpr const char* kIn = "in";
inline constexpr const char* kIsNull = "is_null";

} // namespace old

/// TODO: If you're using these literals you don't account that these functions
/// created with prefix. In general we should fix these usages or create these
/// functions without prefix. Or move these files to test/benchmark utils.
/// To encourage to fix these usages we put these literals to `bug` namespace.
namespace bug {

inline constexpr const char* kNot = "not";
inline constexpr const char* kEq = "eq";
inline constexpr const char* kNeq = "neq";
inline constexpr const char* kLt = "lt";
inline constexpr const char* kLte = "lte";
inline constexpr const char* kGt = "gt";
inline constexpr const char* kGte = "gte";
inline constexpr const char* kArrayConstructor = "array_constructor";
inline constexpr const char* kBetween = "between";

} // namespace bug
} // namespace facebook::velox::expression
