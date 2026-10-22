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

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive::paimon {

/// Paimon logical attributes parallel to a Velox Type. Only ROW children have
/// DataField IDs; ARRAY elements and MAP keys/values do not acquire synthetic
/// IDs here. Logical roots and parameters retain distinctions such as
/// CHAR/VARCHAR and timestamp precision that a Velox Type alone cannot express.
struct PaimonTypeMetadata {
  std::string logicalType;
  bool nullable{true};
  std::optional<int32_t> fieldId{};
  /// Length for character/binary types, precision and scale for DECIMAL, or
  /// precision for timestamps. Empty for unparameterized types.
  std::vector<int32_t> parameters{};
  std::vector<PaimonTypeMetadata> children{};

  bool operator==(const PaimonTypeMetadata& other) const;

  /// Checks the entire subtree, including unique field IDs and ROW names.
  /// isField is true only for a ROW child, which must have a DataField ID.
  void validate(const TypePtr& type, bool isField) const;

  folly::dynamic serialize() const;
  /// Decodes metadata wire; validate() checks it against a Velox type.
  static PaimonTypeMetadata create(const folly::dynamic& obj);
};

/// A normalized table schema with stable DataField identities. 'schemaVersion'
/// identifies the source schema JSON format (1-3). Options are retained exactly
/// as supplied; callers prepare version-specific defaults before construction.
/// Type support here describes the model, not a reader's execution capability.
class PaimonSchema {
 public:
  static constexpr int32_t kSerdeVersion = 1;

  /// Validates the type/metadata tree and builds a field ID index. Keys are
  /// ordered DataField IDs of top-level scalar fields. Nonempty primary keys
  /// must include a non-partition field. User field IDs are in [0,
  /// INT32_MAX/2). Nesting deeper than 128 child edges is rejected.
  PaimonSchema(
      int64_t schemaId,
      int32_t schemaVersion,
      RowTypePtr rowType,
      PaimonTypeMetadata metadata,
      std::vector<int32_t> partitionKeys = {},
      std::vector<int32_t> primaryKeys = {},
      std::unordered_map<std::string, std::string> options = {});

  int64_t schemaId() const {
    return schemaId_;
  }
  int32_t schemaVersion() const {
    return schemaVersion_;
  }
  const RowTypePtr& rowType() const {
    return rowType_;
  }
  const PaimonTypeMetadata& metadata() const {
    return metadata_;
  }
  const std::vector<int32_t>& partitionKeys() const {
    return partitionKeys_;
  }
  const std::vector<int32_t>& primaryKeys() const {
    return primaryKeys_;
  }
  const std::unordered_map<std::string, std::string>& options() const {
    return options_;
  }

  /// Child indices in this schema's type tree, not field IDs or output
  /// channels.
  const std::vector<column_index_t>& fieldIndexPath(int32_t fieldId) const;
  TypePtr fieldType(int32_t fieldId) const;
  const PaimonTypeMetadata& fieldMetadata(int32_t fieldId) const;
  /// Resolves a path rooted at a top-level field. Container subscripts select
  /// the element/value subtree and do not add synthetic DataField IDs.
  std::vector<int32_t> fieldIdPath(const common::Subfield& subfield) const;

  /// Uses self-contained Velox type wire, with an explicit version independent
  /// of the source schema format. Type serde registration is required by
  /// create(). Missing fields and unsupported versions are rejected.
  folly::dynamic serialize() const;
  static std::shared_ptr<const PaimonSchema> create(const folly::dynamic& obj);

 private:
  const int64_t schemaId_;
  const int32_t schemaVersion_;
  const RowTypePtr rowType_;
  const PaimonTypeMetadata metadata_;
  const std::vector<int32_t> partitionKeys_;
  const std::vector<int32_t> primaryKeys_;
  const std::unordered_map<std::string, std::string> options_;
  std::unordered_map<int32_t, std::vector<column_index_t>> fieldIndexPaths_;
};

using PaimonSchemaPtr = std::shared_ptr<const PaimonSchema>;

} // namespace facebook::velox::connector::hive::paimon
