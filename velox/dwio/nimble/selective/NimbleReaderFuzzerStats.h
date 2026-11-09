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

#include <string>
#include <string_view>

namespace facebook::velox::common {
class Filter;
} // namespace facebook::velox::common

namespace facebook::nimble::fuzzer {

inline constexpr std::string_view kStringDictionaryEncodingPreserved =
    "nimbleStringDictionaryEncodingPreserved";
inline constexpr std::string_view kStringDictionaryEncodingAbandoned =
    "nimbleStringDictionaryEncodingAbandoned";

inline constexpr std::string_view kSmallintDictionaryAwareReads =
    "nimbleSmallintDictionaryAwareReads";
inline constexpr std::string_view kIntegerDictionaryAwareReads =
    "nimbleIntegerDictionaryAwareReads";
inline constexpr std::string_view kBigintDictionaryAwareReads =
    "nimbleBigintDictionaryAwareReads";
inline constexpr std::string_view kRealDictionaryAwareReads =
    "nimbleRealDictionaryAwareReads";
inline constexpr std::string_view kDoubleDictionaryAwareReads =
    "nimbleDoubleDictionaryAwareReads";

/// Returns the stat that counts the reads in 'typeStat' that applied a filter
/// of 'filterKind'. Each completed fused dictionary read adds one to its type's
/// stat, such as kIntegerDictionaryAwareReads, and one to this stat for the
/// kind of its filter, such as "nimbleIntegerDictionaryAwareReads.BigintRange".
/// A read without a filter counts under "AlwaysTrue".
inline std::string dictionaryAwareReadsByFilterKind(
    std::string_view typeStat,
    std::string_view filterKind) {
  return std::string(typeStat) + "." + std::string(filterKind);
}

#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED

void updateStringDictionaryEncodingPreserved();
void updateStringDictionaryEncodingAbandoned();
void updateSmallintDictionaryAwareReads(const velox::common::Filter& filter);
void updateIntegerDictionaryAwareReads(const velox::common::Filter& filter);
void updateBigintDictionaryAwareReads(const velox::common::Filter& filter);
void updateRealDictionaryAwareReads(const velox::common::Filter& filter);
void updateDoubleDictionaryAwareReads(const velox::common::Filter& filter);

#else

static inline void updateStringDictionaryEncodingPreserved() {}
static inline void updateStringDictionaryEncodingAbandoned() {}
static inline void updateSmallintDictionaryAwareReads(
    const velox::common::Filter& /*filter*/) {}
static inline void updateIntegerDictionaryAwareReads(
    const velox::common::Filter& /*filter*/) {}
static inline void updateBigintDictionaryAwareReads(
    const velox::common::Filter& /*filter*/) {}
static inline void updateRealDictionaryAwareReads(
    const velox::common::Filter& /*filter*/) {}
static inline void updateDoubleDictionaryAwareReads(
    const velox::common::Filter& /*filter*/) {}

#endif

} // namespace facebook::nimble::fuzzer
