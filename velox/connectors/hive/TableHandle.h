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
#include "velox/core/ITypedExpr.h"
#include "velox/type/Filter.h"
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

class HiveColumnHandle : public ColumnHandle {
 public:
  /// NOTE: Make sure to update the mapping in columnTypeNames() when modifying
  /// this.
  enum class ColumnType {
    kPartitionKey,
    kRegular,
    kSynthesized,
    /// A zero-based row number of type BIGINT auto-generated by the connector.
    /// Rows numbers are unique within a single file only.
    kRowIndex,
    kRowId,
  };

  struct ColumnParseParameters {
    enum PartitionDateValueFormat {
      kISO8601,
      kDaysSinceEpoch,
    } partitionDateValueFormat;
  };

  /// NOTE: 'dataType' is the column type in target write table.  'hiveType' is
  /// converted type of the corresponding column in source table which might not
  /// be the same type, and the table scan needs to do data coercion if needs.
  /// The table writer also needs to respect the type difference when processing
  /// input data such as bucket id calculation.
  ///
  /// 'extractions' specifies named extraction chains.  When non-empty,
  /// 'requiredSubfields' must be empty (mutually exclusive).  When a single
  /// extraction is present, 'dataType' is that extraction's dataType.  When
  /// multiple extractions are present, 'dataType' is a ROW type whose fields
  /// are the outputNames with their corresponding dataTypes.
  HiveColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      std::vector<common::Subfield> requiredSubfields = {},
      std::vector<NamedExtraction> extractions = {},
      ColumnParseParameters columnParseParameters = {},
      std::function<void(VectorPtr&)> postProcessor = {});

  /// Legacy constructor without extractions for backward compatibility.
  HiveColumnHandle(
      const std::string& name,
      ColumnType columnType,
      TypePtr dataType,
      TypePtr hiveType,
      std::vector<common::Subfield> requiredSubfields,
      ColumnParseParameters columnParseParameters,
      std::function<void(VectorPtr&)> postProcessor = {})
      : HiveColumnHandle(
            name,
            columnType,
            std::move(dataType),
            std::move(hiveType),
            std::move(requiredSubfields),
            /*extractions=*/{},
            columnParseParameters,
            std::move(postProcessor)) {}

  const std::string& name() const override {
    return name_;
  }

  ColumnType columnType() const {
    return columnType_;
  }

  const TypePtr& dataType() const {
    return dataType_;
  }

  const TypePtr& hiveType() const {
    return hiveType_;
  }

  /// Applies to columns of complex types: arrays, maps and structs.  When a
  /// query uses only some of the subfields, the engine provides the complete
  /// list of required subfields and the connector is free to prune the rest.
  ///
  /// Examples:
  ///  - SELECT a[1], b['x'], x.y FROM t
  ///  - SELECT a FROM t WHERE b['y'] > 10
  ///
  /// Pruning a struct means populating some of the members with null values.
  ///
  /// Pruning a map means dropping keys not listed in the required subfields.
  ///
  /// Pruning arrays means dropping values with indices larger than maximum
  /// required index.
  const std::vector<common::Subfield>& requiredSubfields() const {
    return requiredSubfields_;
  }

  /// Named extraction chains.  Empty means no extraction (current behavior).
  /// When a single entry is present, the column handle's dataType is that
  /// entry's dataType.  When multiple entries are present, the column
  /// handle's dataType is a ROW type whose fields are the outputNames with
  /// their corresponding dataTypes.
  /// Mutually exclusive with requiredSubfields — if extractions is non-empty,
  /// requiredSubfields must be empty.
  const std::vector<NamedExtraction>& extractions() const {
    return extractions_;
  }

  bool isPartitionKey() const {
    return columnType_ == ColumnType::kPartitionKey;
  }

  bool isPartitionDateValueDaysSinceEpoch() const {
    return columnParseParameters_.partitionDateValueFormat ==
        ColumnParseParameters::kDaysSinceEpoch;
  }

  /// Apply some row-wise post processing to this column when it is present in
  /// output.
  ///
  /// It's not allowed to change the size of the vector in the processor.  The
  /// top level vector is guaranteed to be safe to change.  Any inner vectors
  /// and buffers need to check the reference count before doing any change in
  /// place, otherwise you need to allocate new vectors and buffers.
  ///
  /// For lazy vector, this will be applied after the lazy vector is loaded.
  /// This is only applied after all the filtering is done; the filters (both
  /// subfield filters and remaining filter) still apply to values before post
  /// processing.  ValueHook usage will be disabled if a post processor is
  /// present.
  const std::function<void(VectorPtr&)>& postProcessor() const {
    return postProcessor_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ColumnHandlePtr create(const folly::dynamic& obj);

  static std::string columnTypeName(HiveColumnHandle::ColumnType columnType);

  static HiveColumnHandle::ColumnType columnTypeFromName(
      const std::string& name);

  static void registerSerDe();

 private:
  const std::string name_;
  const ColumnType columnType_;
  const TypePtr dataType_;
  const TypePtr hiveType_;
  const std::vector<common::Subfield> requiredSubfields_;
  const std::vector<NamedExtraction> extractions_;
  const ColumnParseParameters columnParseParameters_;
  const std::function<void(VectorPtr&)> postProcessor_;
};

using HiveColumnHandlePtr = std::shared_ptr<const HiveColumnHandle>;
using HiveColumnHandleMap =
    std::unordered_map<std::string, HiveColumnHandlePtr>;

class HiveTableHandle : public ConnectorTableHandle {
 public:
  /// @param sampleRate Sampling rate in (0, 1] range. 0.1 means 10% sampling.
  /// 1.0 means no sampling. Default is no sampling.
  HiveTableHandle(
      std::string connectorId,
      const std::string& tableName,
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter,
      const RowTypePtr& dataColumns = nullptr,
      std::vector<std::string> indexColumns = {},
      const std::unordered_map<std::string, std::string>& tableParameters = {},
      std::vector<HiveColumnHandlePtr> filterColumnHandles = {},
      double sampleRate = 1.0,
      std::string dbName = "");

  /// Legacy constructor without indexColumns parameter for backward
  /// compatibility.
  HiveTableHandle(
      std::string connectorId,
      const std::string& tableName,
      common::SubfieldFilters subfieldFilters,
      const core::TypedExprPtr& remainingFilter,
      const RowTypePtr& dataColumns,
      const std::unordered_map<std::string, std::string>& tableParameters,
      std::vector<HiveColumnHandlePtr> filterColumnHandles,
      double sampleRate = 1.0);

  const std::string& tableName() const {
    return tableName_;
  }

  const std::string& name() const override {
    return tableName();
  }

  bool supportsIndexLookup() const override {
    return !indexColumns_.empty();
  }

  bool needsIndexSplit() const override {
    return true;
  }

  /// Single field filters that can be applied efficiently during file reading.
  const common::SubfieldFilters& subfieldFilters() const {
    return subfieldFilters_;
  }

  /// Everything else that cannot be converted into subfield filters, but still
  /// require the data source to filter out.  This is usually less efficient
  /// than subfield filters but supports arbitrary boolean expression.
  const core::TypedExprPtr& remainingFilter() const {
    return remainingFilter_;
  }

  /// Sampling rate between 0 and 1 (excluding 0). 0.1 means 10%
  /// sampling. 1.0 means no sampling.
  double sampleRate() const {
    return sampleRate_;
  }

  /// Subset of schema of the table that we store in file (i.e.,
  /// non-partitioning columns).  This must be in the exact order as columns in
  /// file (except trailing columns), but with the table schema during read
  /// time.
  ///
  /// This is needed for multiple purposes, including reading TEXTFILE and
  /// handling schema evolution.
  const RowTypePtr& dataColumns() const {
    return dataColumns_;
  }

  /// Returns the names of the index columns for the table.
  const std::vector<std::string>& indexColumns() const {
    return indexColumns_;
  }

  /// Extra parameters to pass down to file format reader layer.  Keys should be
  /// in dwio::common::TableParameter.
  const std::unordered_map<std::string, std::string>& tableParameters() const {
    return tableParameters_;
  }

  /// Extra columns that are used in filters and remaining filters, but not in
  /// the output.  If there is overlap with data source assignments parameter,
  /// the name and types should be the same (the required subfields are taken
  /// from assignments).
  const std::vector<HiveColumnHandlePtr> filterColumnHandles() const {
    return filterColumnHandles_;
  }

  /// Database or namespace name for this table. Used to pass per-table identity
  /// through the Velox pipeline for token dispatch.
  const std::string& dbName() const {
    return dbName_;
  }

  std::string toString() const override;

  folly::dynamic serialize() const override;

  static ConnectorTableHandlePtr create(
      const folly::dynamic& obj,
      void* context);

  static void registerSerDe();

 private:
  const std::string tableName_;
  const common::SubfieldFilters subfieldFilters_;
  const core::TypedExprPtr remainingFilter_;
  const double sampleRate_;
  const RowTypePtr dataColumns_;
  const std::vector<std::string> indexColumns_;
  const std::unordered_map<std::string, std::string> tableParameters_;
  const std::vector<HiveColumnHandlePtr> filterColumnHandles_;
  const std::string dbName_;
};

using HiveTableHandlePtr = std::shared_ptr<const HiveTableHandle>;

} // namespace facebook::velox::connector::hive

template <>
struct fmt::formatter<
    facebook::velox::connector::hive::HiveColumnHandle::ColumnType>
    : formatter<std::string> {
  auto format(
      facebook::velox::connector::hive::HiveColumnHandle::ColumnType type,
      format_context& ctx) const {
    return formatter<std::string>::format(
        facebook::velox::connector::hive::HiveColumnHandle::columnTypeName(
            type),
        ctx);
  }
};
