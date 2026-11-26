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
#include "velox/connectors/paimon/PaimonMergeEngine.h"

#include <folly/String.h>

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::paimon {

std::string paimonMergeEngineString(PaimonMergeEngine mergeEngine) {
  switch (mergeEngine) {
    case PaimonMergeEngine::kDeduplicate:
      return "deduplicate";
    case PaimonMergeEngine::kPartialUpdate:
      return "partial-update";
    case PaimonMergeEngine::kAggregation:
      return "aggregation";
    case PaimonMergeEngine::kFirstRow:
      return "first-row";
    default:
      VELOX_FAIL(
          "Unknown PaimonMergeEngine: {}", static_cast<int>(mergeEngine));
  }
}

PaimonMergeEngine paimonMergeEngineFromString(std::string_view value) {
  std::string lower{value};
  folly::toLowerAscii(lower);
  for (auto mergeEngine :
       {PaimonMergeEngine::kDeduplicate,
        PaimonMergeEngine::kPartialUpdate,
        PaimonMergeEngine::kAggregation,
        PaimonMergeEngine::kFirstRow}) {
    if (lower == paimonMergeEngineString(mergeEngine)) {
      return mergeEngine;
    }
  }
  VELOX_USER_FAIL("Unknown Paimon merge engine: {}", value);
}

} // namespace facebook::velox::connector::paimon
