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
#include "velox/connectors/hive/paimon/PaimonRowKind.h"

#include "velox/common/base/Exceptions.h"

namespace facebook::velox::connector::hive::paimon {

std::string paimonRowKindString(PaimonRowKind kind) {
  switch (kind) {
    case PaimonRowKind::kInsert:
      return "+I";
    case PaimonRowKind::kUpdateBefore:
      return "-U";
    case PaimonRowKind::kUpdateAfter:
      return "+U";
    case PaimonRowKind::kDelete:
      return "-D";
    default:
      VELOX_FAIL("Unknown PaimonRowKind: {}", static_cast<int>(kind));
  }
}

PaimonRowKind paimonRowKindFromValue(int8_t value) {
  switch (value) {
    case 0:
      return PaimonRowKind::kInsert;
    case 1:
      return PaimonRowKind::kUpdateBefore;
    case 2:
      return PaimonRowKind::kUpdateAfter;
    case 3:
      return PaimonRowKind::kDelete;
    default:
      VELOX_FAIL("Unknown PaimonRowKind value: {}", value);
  }
}

std::string paimonChangelogModeString(PaimonChangelogMode mode) {
  switch (mode) {
    case PaimonChangelogMode::kNone:
      return "NONE";
    case PaimonChangelogMode::kInput:
      return "INPUT";
    case PaimonChangelogMode::kLookup:
      return "LOOKUP";
    case PaimonChangelogMode::kFullCompaction:
      return "FULL_COMPACTION";
    default:
      VELOX_FAIL("Unknown PaimonChangelogMode: {}", static_cast<int>(mode));
  }
}

PaimonChangelogMode paimonChangelogModeFromString(const std::string& str) {
  if (str == "NONE") {
    return PaimonChangelogMode::kNone;
  }
  if (str == "INPUT") {
    return PaimonChangelogMode::kInput;
  }
  if (str == "LOOKUP") {
    return PaimonChangelogMode::kLookup;
  }
  if (str == "FULL_COMPACTION") {
    return PaimonChangelogMode::kFullCompaction;
  }
  VELOX_FAIL("Unknown PaimonChangelogMode: {}", str);
}

} // namespace facebook::velox::connector::hive::paimon
