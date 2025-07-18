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

#include "velox/connectors/hive/iceberg/Transforms.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::connector::hive::iceberg {

class ColumnTransform {
 public:
  /// Constructs a ColumnTransform.
  /// @param columnName The name of the column to transform.
  /// @param transform The transform to apply to the column.
  ColumnTransform(
      const std::string& columnName,
      const std::shared_ptr<Transform>& transform)
      : columnName_(std::move(columnName)), transform_(transform) {}

  /// Returns the name of the transform, such as identity, bucket etc.
  /// @return The transform name.
  const std::string transformName() const {
    return transform_->name();
  }

  /// Returns the name of the partition column.
  /// @return The column name.
  const std::string& columnName() const {
    return columnName_;
  }

  /// Returns the result type of the transform, e.g., INTEGER() for bucket
  /// transform.
  /// @return The result type.
  TypePtr resultType() const {
    return transform_->resultType();
  }

  /// Applies the transform to the specified column in the input row vector.
  /// For nested columns (e.g., struct.field1.field2), this method will
  /// navigate through the nested structure to find the target column.
  /// @param input The input row vector containing the column to transform.
  /// @return The transformed vector.
  [[nodiscard]] VectorPtr transform(const RowVectorPtr& input) const {
    std::vector<std::string> parts;
    // Recursively find the nested column. e.g. struct.layer1.layer2.
    folly::split('.', columnName_, parts);
    VectorPtr currentVector = input->childAt(parts[0]);
    for (auto i = 1; i < parts.size(); ++i) {
      VELOX_CHECK(
          currentVector->type()->isRow(),
          "Column '{}' in path '{}' is not a struct/row type",
          parts[i - 1],
          columnName_);

      auto rowVector = currentVector->asUnchecked<RowVector>();
      currentVector = rowVector->childAt(parts[i]);
    }

    return transform_->apply(currentVector);
  }

  /// Convert iceberg partition values to human-readable string.
  /// @param value The partition value that has been applied the partition
  /// transform.
  /// @return The converted string.
  template <typename T>
  std::string toHumanString(const T& value) const {
    return transform_->toHumanString(value);
  }

 private:
  std::string columnName_;
  std::shared_ptr<Transform> transform_;
};

} // namespace facebook::velox::connector::hive::iceberg
