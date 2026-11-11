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

#include <array>
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

/// Returns the stat that counts the reads in 'typeStat' whose filter is of
/// 'filterKind' and accepts nulls, such as
/// "nimbleIntegerDictionaryAwareReads.BigintRange.filterAcceptsNull".
/// - These reads also count under dictionaryAwareReadsByFilterKind().
/// - Reads without a filter (AlwaysTrue) never count here: they have no
///   filtered output to add null rows to.
inline std::string dictionaryAwareReadsWithFilterAcceptingNull(
    std::string_view typeStat,
    std::string_view filterKind) {
  return dictionaryAwareReadsByFilterKind(typeStat, filterKind) +
      ".filterAcceptsNull";
}

/// Paths of the fused dictionary read hook that the fuzzer counts for reads
/// whose filter accepts nulls:
/// - addValue: the hook got indices one at a time.
/// - addValues: the hook got batches of indices.
/// - mergeBulkNulls.dense and mergeBulkNulls.sparse: a batch merged null rows
///   into filtered output through that branch.
inline constexpr std::string_view kAddValuePath = "addValue";
inline constexpr std::string_view kAddValuesPath = "addValues";
inline constexpr std::string_view kMergeBulkNullsDensePath =
    "mergeBulkNulls.dense";
inline constexpr std::string_view kMergeBulkNullsSparsePath =
    "mergeBulkNulls.sparse";

/// The hook paths above, in report order.
inline constexpr std::array kDictionaryAwareReadPaths{
    kAddValuePath,
    kAddValuesPath,
    kMergeBulkNullsDensePath,
    kMergeBulkNullsSparsePath,
};

/// Returns the stat that counts the reads in 'typeStat' whose filter accepts
/// nulls and that took hook path 'path', such as
/// "nimbleIntegerDictionaryAwareReads.filterAcceptsNull.addValues". A read
/// counts once under each path it took.
inline std::string dictionaryAwareReadsWithFilterAcceptingNullByPath(
    std::string_view typeStat,
    std::string_view path) {
  return std::string(typeStat) + ".filterAcceptsNull." + std::string(path);
}

#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED

/// Records which hook paths one fused dictionary read took.
class DictionaryAwareReadPaths {
 public:
  void recordAddValue() {
    addValue_ = true;
  }

  void recordAddValues() {
    addValues_ = true;
  }

  void recordMergeBulkNullsDense() {
    mergeBulkNullsDense_ = true;
  }

  void recordMergeBulkNullsSparse() {
    mergeBulkNullsSparse_ = true;
  }

  bool addValue() const {
    return addValue_;
  }

  bool addValues() const {
    return addValues_;
  }

  bool mergeBulkNullsDense() const {
    return mergeBulkNullsDense_;
  }

  bool mergeBulkNullsSparse() const {
    return mergeBulkNullsSparse_;
  }

 private:
  // Paths the read took, set by the record methods.
  bool addValue_{false};
  bool addValues_{false};
  bool mergeBulkNullsDense_{false};
  bool mergeBulkNullsSparse_{false};
};

void updateStringDictionaryEncodingPreserved();
void updateStringDictionaryEncodingAbandoned();
void updateSmallintDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths);
void updateIntegerDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths);
void updateBigintDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths);
void updateRealDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths);
void updateDoubleDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths);

#else

/// Records nothing, so production builds pay nothing for recording paths.
class DictionaryAwareReadPaths {
 public:
  void recordAddValue() {}

  void recordAddValues() {}

  void recordMergeBulkNullsDense() {}

  void recordMergeBulkNullsSparse() {}
};

static inline void updateStringDictionaryEncodingPreserved() {}
static inline void updateStringDictionaryEncodingAbandoned() {}
static inline void updateSmallintDictionaryAwareReads(
    const velox::common::Filter& /*filter*/,
    const DictionaryAwareReadPaths& /*paths*/) {}
static inline void updateIntegerDictionaryAwareReads(
    const velox::common::Filter& /*filter*/,
    const DictionaryAwareReadPaths& /*paths*/) {}
static inline void updateBigintDictionaryAwareReads(
    const velox::common::Filter& /*filter*/,
    const DictionaryAwareReadPaths& /*paths*/) {}
static inline void updateRealDictionaryAwareReads(
    const velox::common::Filter& /*filter*/,
    const DictionaryAwareReadPaths& /*paths*/) {}
static inline void updateDoubleDictionaryAwareReads(
    const velox::common::Filter& /*filter*/,
    const DictionaryAwareReadPaths& /*paths*/) {}

#endif

} // namespace facebook::nimble::fuzzer
