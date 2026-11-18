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

#include "velox/dwio/nimble/selective/NimbleReaderFuzzerStats.h"

#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED

#include "velox/common/base/RuntimeMetrics.h"
#include "velox/type/Filter.h"

#include <utility>

namespace facebook::nimble::fuzzer {

namespace {

void update(std::string_view name) {
  velox::addThreadLocalRuntimeStat(name, velox::RuntimeCounter(1));
}

// Counts one read under:
// - 'typeStat' and its stat for the kind of 'filter'.
// - dictionaryAwareReadsWithFilterAcceptingNull() and the stat of each hook
//   path in 'paths', if 'filter' accepts nulls and is not AlwaysTrue (no
//   filter).
void updateDictionaryAwareReads(
    std::string_view typeStat,
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths) {
  update(typeStat);
  update(dictionaryAwareReadsByFilterKind(typeStat, filter.kindName()));
  if (filter.kind() == velox::common::FilterKind::kAlwaysTrue ||
      !filter.nullAllowed()) {
    return;
  }
  update(
      dictionaryAwareReadsWithFilterAcceptingNull(typeStat, filter.kindName()));
  const std::pair<bool, std::string_view> pathsTaken[]{
      {paths.addValue(), kAddValuePath},
      {paths.addValues(), kAddValuesPath},
      {paths.mergeBulkNullsDense(), kMergeBulkNullsDensePath},
      {paths.mergeBulkNullsSparse(), kMergeBulkNullsSparsePath},
  };
  for (const auto& [taken, path] : pathsTaken) {
    if (taken) {
      update(dictionaryAwareReadsWithFilterAcceptingNullByPath(typeStat, path));
    }
  }
}

} // namespace

void updateStringDictionaryEncodingPreserved() {
  update(kStringDictionaryEncodingPreserved);
}

void updateStringDictionaryEncodingAbandoned() {
  update(kStringDictionaryEncodingAbandoned);
}

void updateSmallintDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths) {
  updateDictionaryAwareReads(kSmallintDictionaryAwareReads, filter, paths);
}

void updateIntegerDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths) {
  updateDictionaryAwareReads(kIntegerDictionaryAwareReads, filter, paths);
}

void updateBigintDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths) {
  updateDictionaryAwareReads(kBigintDictionaryAwareReads, filter, paths);
}

void updateRealDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths) {
  updateDictionaryAwareReads(kRealDictionaryAwareReads, filter, paths);
}

void updateDoubleDictionaryAwareReads(
    const velox::common::Filter& filter,
    const DictionaryAwareReadPaths& paths) {
  updateDictionaryAwareReads(kDoubleDictionaryAwareReads, filter, paths);
}

} // namespace facebook::nimble::fuzzer

#endif
