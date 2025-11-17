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

#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/expression/Expr.h"

namespace facebook::velox::connector::hive::iceberg {

/// Converts Iceberg partition specification to Velox expressions.
class TransformExprBuilder {
 public:
  /// Converts partition specification to a list of typed expressions.
  ///
  /// @param partitionSpec Iceberg partition specification containing transform
  /// definitions for each partition field.
  /// @param partitionChannels Column indices (0-based) in the input RowVector
  /// that correspond to each partition field. Must have the same size as
  /// partitionSpec->fields. Provides the positional mapping from partition spec
  /// fields to input RowVector columns.
  /// @param inputType The row type of the input data. This is necessary for
  /// building expressions because the column names in partitionSpec reference
  /// table schema names, which might not match the column names in inputType
  /// (e.g., inputType may use generated names like c0, c1, c2). The
  /// FieldAccessTypedExpr must be built using the actual column names from
  /// inputType that will be present at runtime. The partitionChannels provide
  /// the positional mapping to locate the correct columns.
  /// @param icebergFuncPrefix Prefix for Iceberg transform function names.
  /// @return Vector of typed expressions, one for each partition field.
  static std::vector<core::TypedExprPtr> toExpressions(
      const IcebergPartitionSpecPtr& partitionSpec,
      const std::vector<column_index_t>& partitionChannels,
      const RowTypePtr& inputType,
      const std::string& icebergFuncPrefix);
};

} // namespace facebook::velox::connector::hive::iceberg
