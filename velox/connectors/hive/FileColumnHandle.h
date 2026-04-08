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

#include "velox/connectors/Connector.h"
#include "velox/type/Subfield.h"
#include "velox/type/Type.h"

namespace facebook::velox::connector::hive {

/// Type of extraction to apply at one nesting level.
enum class ExtractionStep : uint8_t {
  /// Navigate into a struct field.  Input must be ROW.
  kStructField,
  /// Extract map keys as ARRAY.  Input must be MAP.
  kMapKeys,
  /// Extract map values as ARRAY.  Input must be MAP.
  kMapValues,
  /// Filter map to specific keys.  Input must be MAP.  Type-preserving.
  kMapKeyFilter,
  /// Navigate into array elements.  Input must be ARRAY.
  kArrayElements,
  /// Extract size as BIGINT.  Input must be MAP or ARRAY.  Terminal.
  kSize,
};

/// Base class for one step in the extraction chain.
class ExtractionPathElement {
 public:
  virtual ~ExtractionPathElement() = default;

  /// Return the step type.
  virtual ExtractionStep step() const = 0;

  /// Create a simple step (MapKeys, MapValues, ArrayElements, Size).
  static std::shared_ptr<const ExtractionPathElement> simple(
      ExtractionStep step);

  /// Create a StructField step.
  static std::shared_ptr<const ExtractionPathElement> structField(
      const std::string& name);

  /// Create a MapKeyFilter step with string keys.
  static std::shared_ptr<const ExtractionPathElement> mapKeyFilter(
      std::vector<std::string> keys);

  /// Create a MapKeyFilter step with integer keys.
  static std::shared_ptr<const ExtractionPathElement> mapKeyFilter(
      std::vector<int64_t> keys);
};

using ExtractionPathElementPtr = std::shared_ptr<const ExtractionPathElement>;

/// Simple extraction step without extra data (MapKeys, MapValues,
/// ArrayElements, Size).
class SimpleExtractionPathElement : public ExtractionPathElement {
 public:
  explicit SimpleExtractionPathElement(ExtractionStep step) : step_(step) {}

  ExtractionStep step() const override {
    return step_;
  }

 private:
  ExtractionStep step_;
};

/// Struct field extraction step.
class StructFieldExtractionPathElement : public ExtractionPathElement {
 public:
  explicit StructFieldExtractionPathElement(std::string name)
      : fieldName_(std::move(name)) {}

  ExtractionStep step() const override {
    return ExtractionStep::kStructField;
  }

  const std::string& fieldName() const {
    return fieldName_;
  }

 private:
  std::string fieldName_;
};

/// Map key filter extraction step with string keys.
class StringMapKeyFilterExtractionPathElement : public ExtractionPathElement {
 public:
  explicit StringMapKeyFilterExtractionPathElement(
      std::vector<std::string> keys)
      : filterKeys_(std::move(keys)) {}

  ExtractionStep step() const override {
    return ExtractionStep::kMapKeyFilter;
  }

  const std::vector<std::string>& filterKeys() const {
    return filterKeys_;
  }

 private:
  std::vector<std::string> filterKeys_;
};

/// Map key filter extraction step with integer keys.
class IntMapKeyFilterExtractionPathElement : public ExtractionPathElement {
 public:
  explicit IntMapKeyFilterExtractionPathElement(std::vector<int64_t> keys)
      : filterKeys_(std::move(keys)) {}

  ExtractionStep step() const override {
    return ExtractionStep::kMapKeyFilter;
  }

  const std::vector<int64_t>& filterKeys() const {
    return filterKeys_;
  }

 private:
  std::vector<int64_t> filterKeys_;
};

/// Compare two ExtractionPathElements for equality.  Non-virtual to avoid
/// slicing hazards with virtual operator==.
bool extractionPathElementEquals(
    const ExtractionPathElement& lhs,
    const ExtractionPathElement& rhs);

/// Named extraction chain producing one output column.
struct NamedExtraction {
  /// Output column name in the scan's outputType.
  std::string outputName;

  /// Extraction chain to apply.  Empty means pass-through (no extraction).
  std::vector<ExtractionPathElementPtr> chain;

  /// Output type after applying the chain.
  TypePtr dataType;

  bool operator==(const NamedExtraction& other) const;
};

/// Return the string name for an ExtractionStep enum value.
std::string extractionStepName(ExtractionStep step);

/// Parse an ExtractionStep from its string name.
ExtractionStep extractionStepFromName(const std::string& name);

/// Derive the output type by applying the extraction chain to the input type.
/// Throws if the chain is invalid for the given input type.
TypePtr deriveExtractionOutputType(
    const TypePtr& inputType,
    const std::vector<ExtractionPathElementPtr>& chain);

/// Base class for column handles in file-based connectors.
///
/// Define the common interface for column metadata needed by
/// FileDataSource and FileSplitReader. Connector-specific column handles
/// (HiveColumnHandle, etc.) extend this class.
class FileColumnHandle : public ColumnHandle {
 public:
  /// Classify columns by their role in the scan pipeline.
  enum class ColumnType {
    kPartitionKey,
    kRegular,
    kSynthesized,
    /// A zero-based row number of type BIGINT auto-generated by the connector.
    /// Row numbers are unique within a single file only.
    kRowIndex,
    kRowId,
  };

  virtual ColumnType columnType() const = 0;

  /// The target data type for this column in the output.
  virtual const TypePtr& dataType() const = 0;

  /// Subfields required by the query for pruning complex types.
  virtual const std::vector<common::Subfield>& requiredSubfields() const = 0;

  virtual bool isPartitionKey() const {
    return columnType() == ColumnType::kPartitionKey;
  }

  /// Return true if partition date values are encoded as days since epoch
  /// (e.g., Iceberg) rather than ISO 8601 strings (e.g., Hive).
  virtual bool isPartitionDateValueDaysSinceEpoch() const = 0;

  /// Named extraction chains.  Empty means no extraction (default behavior).
  virtual const std::vector<NamedExtraction>& extractions() const {
    static const std::vector<NamedExtraction> kEmpty;
    return kEmpty;
  }

  /// Optional row-wise post processor applied to this column after reading and
  /// filtering. Must not change the vector size.
  virtual const std::function<void(VectorPtr&)>& postProcessor() const = 0;

  static std::string columnTypeName(ColumnType columnType);

  static ColumnType columnTypeFromName(const std::string& name);
};

using FileColumnHandlePtr = std::shared_ptr<const FileColumnHandle>;
using FileColumnHandleMap =
    std::unordered_map<std::string, FileColumnHandlePtr>;

} // namespace facebook::velox::connector::hive

template <>
struct fmt::formatter<
    facebook::velox::connector::hive::FileColumnHandle::ColumnType>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::FileColumnHandle::ColumnType type,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::FileColumnHandle::columnTypeName(
            type),
        ctx);
  }
};
