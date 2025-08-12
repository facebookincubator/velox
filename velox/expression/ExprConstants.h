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

namespace facebook::velox::expression {

/// Eventually replace with std::string_view but many APIs
/// use const std::string& which need to change first.
static const char* const kConjunt = "conjunct";
static const char* const kAnd = "and";
static const char* const kOr = "or";
static const char* const kSwitch = "switch";
static const char* const kIn = "in";
static const char* const kIf = "if";
static const char* const kFail = "fail";
static const char* const kCoalesce = "coalesce";
static const char* const kCast = "cast";
static const char* const kTryCast = "try_cast";
static const char* const kTry = "try";

} // namespace facebook::velox::expression
