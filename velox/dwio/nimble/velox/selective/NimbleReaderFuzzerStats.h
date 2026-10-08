/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

namespace facebook::nimble::fuzzer {

inline constexpr std::string_view kStringDictionaryEncodingPreserved =
    "nimbleStringDictionaryEncodingPreserved";
inline constexpr std::string_view kStringDictionaryEncodingAbandoned =
    "nimbleStringDictionaryEncodingAbandoned";

#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED

void updateStringDictionaryEncodingPreserved();
void updateStringDictionaryEncodingAbandoned();

#else

static inline void updateStringDictionaryEncodingPreserved() {}
static inline void updateStringDictionaryEncodingAbandoned() {}

#endif

} // namespace facebook::nimble::fuzzer
