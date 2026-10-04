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
#include "velox/connectors/hive/paimon/PaimonDeletionFile.h"

#include "velox/common/base/Exceptions.h"
#include "velox/connectors/hive/paimon/PaimonMetadata.h"

namespace facebook::velox::connector::hive::paimon {

PaimonDeletionFile::PaimonDeletionFile(
    std::string _path,
    uint64_t _offset,
    uint64_t _length,
    uint64_t _cardinality)
    : path(std::move(_path)),
      offset(_offset),
      length(_length),
      cardinality(_cardinality) {
  VELOX_CHECK_GT(length, 0, "PaimonDeletionFile length must be > 0");
  VELOX_CHECK_GT(cardinality, 0, "PaimonDeletionFile cardinality must be > 0");
  constexpr uint64_t max = std::numeric_limits<int64_t>::max();
  VELOX_USER_CHECK_LE(offset, max, "Paimon DV offset out of range");
  VELOX_USER_CHECK_LE(
      length, max - offset, "Paimon DV offset + length overflow");
  VELOX_USER_CHECK_LE(cardinality, max, "Paimon DV cardinality out of range");
}

std::string PaimonDeletionFile::toString() const {
  return fmt::format(
      "{{path={}, offset={}, length={}, cardinality={}}}",
      path,
      offset,
      length,
      cardinality);
}

folly::dynamic PaimonDeletionFile::serialize() const {
  folly::dynamic obj = folly::dynamic::object;
  obj["path"] = path;
  obj["offset"] = offset;
  obj["length"] = length;
  obj["cardinality"] = cardinality;
  return obj;
}

// static
PaimonDeletionFile PaimonDeletionFile::create(const folly::dynamic& obj) {
  return PaimonDeletionFile(
      obj["path"].asString(),
      paimonInt(obj, "offset"),
      paimonInt(obj, "length"),
      paimonInt(obj, "cardinality"));
}

} // namespace facebook::velox::connector::hive::paimon
