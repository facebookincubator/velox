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

#include <fmt/format.h>
#include <folly/dynamic.h>
#include <cstdint>
#include <string>

namespace facebook::velox::connector::hive::paimon {

/// A position-deletion reference for an append or primary-key data file.
/// The byte range identifies a Paimon DV payload, not a columnar file. The
/// reader must implement its versioned encoding before accepting this input.
struct PaimonDeletionFile {
  /// @param path Path to the deletion file.
  /// @param offset Byte offset within the container file.
  /// @param length Number of bytes of bitmap data (must be > 0).
  /// @param cardinality Number of deleted rows (must be > 0).
  PaimonDeletionFile(
      std::string path,
      uint64_t offset,
      uint64_t length,
      uint64_t cardinality);

  std::string path;

  // Byte offset within the container file where this bitmap starts.
  uint64_t offset;

  // Number of bytes of bitmap data. Must be > 0.
  uint64_t length;

  // Number of deleted rows (pre-computed for stats without reading bitmap).
  // Must be > 0.
  uint64_t cardinality;

  std::string toString() const;
  folly::dynamic serialize() const;
  static PaimonDeletionFile create(const folly::dynamic& obj);
};

} // namespace facebook::velox::connector::hive::paimon
