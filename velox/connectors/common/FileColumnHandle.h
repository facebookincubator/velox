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

namespace facebook::velox::connector::file {

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

folly::dynamic serializeNamedExtraction(const NamedExtraction& extraction);

NamedExtraction deserializeNamedExtraction(const folly::dynamic& obj);

/// Subfield serde using the string representation in file-based query plans.
folly::dynamic serializeSubfield(const common::Subfield& subfield);
common::Subfield deserializeSubfield(const folly::dynamic& obj);

/// Common column metadata and accessors for file-based connectors. Used by
/// FileDataSource and FileSplitReader. Connector-specific handles extend this
/// class with their own metadata and serialization.
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

  const std::string& name() const override {
    return name_;
  }

  virtual ColumnType columnType() const {
    return columnType_;
  }

  /// The type of this column as defined in the table schema (metastore).
  /// May differ from dataType() when extraction changes the output type.
  virtual const TypePtr& schemaType() const {
    return schemaType_;
  }

  /// The target data type for this column in the output.
  virtual const TypePtr& dataType() const {
    return dataType_;
  }

  /// Applies to columns of complex types: arrays, maps and structs. When a
  /// query uses only some of the subfields, the engine provides the complete
  /// list of required subfields and the connector is free to prune the rest.
  ///
  /// Examples:
  ///  - SELECT a[1], b['x'], x.y FROM t
  ///  - SELECT a FROM t WHERE b['y'] > 10
  ///
  /// Pruning a struct means populating some of the members with null values.
  /// Pruning a map means dropping keys not listed in the required subfields.
  /// Pruning arrays means dropping values with indices larger than maximum
  /// required index.
  virtual const std::vector<common::Subfield>& requiredSubfields() const {
    return requiredSubfields_;
  }

  virtual bool isPartitionKey() const {
    return columnType() == ColumnType::kPartitionKey;
  }

  /// Return true if partition date values are encoded as days since epoch
  /// (e.g., Iceberg) rather than ISO 8601 strings (e.g., Hive).
  virtual bool isPartitionDateValueDaysSinceEpoch() const {
    return partitionDateValueDaysSinceEpoch_;
  }

  /// Named extraction chains. Empty means no extraction. When a single entry
  /// is present, dataType() is that entry's dataType. For multiple entries,
  /// dataType() is a ROW with their outputNames and dataTypes. Mutually
  /// exclusive with requiredSubfields.
  virtual const std::vector<NamedExtraction>& extractions() const {
    return extractions_;
  }

  /// Optional row-wise post processor applied to this column when it is
  /// present in output.
  ///
  /// Must not change the vector size. The top level vector is guaranteed to
  /// be safe to change. Inner vectors and buffers need to check the reference
  /// count before changing in place, or allocate new vectors and buffers.
  ///
  /// For lazy vectors, this is applied after the vector is loaded. It runs
  /// after all filtering, so filters see values before post processing.
  /// ValueHook usage is disabled when a post processor is present.
  virtual const std::function<void(VectorPtr&)>& postProcessor() const {
    return postProcessor_;
  }

  folly::dynamic serialize() const override = 0;

  static std::string columnTypeName(ColumnType columnType);

  static ColumnType columnTypeFromName(const std::string& name);

 protected:
  FileColumnHandle(
      std::string name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr schemaType,
      std::vector<common::Subfield> requiredSubfields,
      std::vector<NamedExtraction> extractions,
      bool partitionDateValueDaysSinceEpoch,
      std::function<void(VectorPtr&)> postProcessor);

  /// Serializes the shared column role, subfields and extractions. Derived
  /// handles add names, types and metadata using their existing wire fields.
  folly::dynamic serializeFileFields(std::string_view typeName) const;

  static void deserializeFileFields(
      const folly::dynamic& obj,
      ColumnType& columnType,
      std::vector<common::Subfield>& requiredSubfields,
      std::vector<NamedExtraction>& extractions);

 private:
  const std::string name_;
  const ColumnType columnType_;
  const TypePtr dataType_;
  const TypePtr schemaType_;
  const std::vector<common::Subfield> requiredSubfields_;
  const std::vector<NamedExtraction> extractions_;
  const bool partitionDateValueDaysSinceEpoch_;
  const std::function<void(VectorPtr&)> postProcessor_;
};

using FileColumnHandlePtr = std::shared_ptr<const FileColumnHandle>;
using FileColumnHandleMap =
    std::unordered_map<std::string, FileColumnHandlePtr>;

} // namespace facebook::velox::connector::file

template <>
struct fmt::formatter<
    facebook::velox::connector::file::FileColumnHandle::ColumnType>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::file::FileColumnHandle::ColumnType type,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::file::FileColumnHandle::columnTypeName(
            type),
        ctx);
  }
};
