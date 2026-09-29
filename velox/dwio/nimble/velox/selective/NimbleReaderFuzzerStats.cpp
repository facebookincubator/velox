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

#include "velox/dwio/nimble/velox/selective/NimbleReaderFuzzerStats.h"

#ifdef NIMBLE_READER_FUZZER_STATS_ENABLED

#include "velox/common/base/RuntimeMetrics.h"

namespace facebook::nimble::fuzzer {

namespace {

void update(std::string_view name) {
  velox::addThreadLocalRuntimeStat(name, velox::RuntimeCounter(1));
}

} // namespace

void updateStringDictionaryEncodingPreserved() {
  update(kStringDictionaryEncodingPreserved);
}

void updateStringDictionaryEncodingAbandoned() {
  update(kStringDictionaryEncodingAbandoned);
}

} // namespace facebook::nimble::fuzzer

#endif
