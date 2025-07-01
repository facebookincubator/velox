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
  ColumnTransform(
      const std::string& columnName,
      const std::shared_ptr<Transform>& transform,
      std::optional<int32_t> parameter)
      : columnName_(std::move(columnName)),
        transform_(transform),
        parameter_(parameter) {}

  const std::string transformName() const {
    return transform_->name();
  }

  const std::string& columnName() const {
    return columnName_;
  }

  TypePtr resultType() const {
    return transform_->resultType();
  }

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

  [[nodiscard]] VectorPtr transformVector(const VectorPtr& block) const {
    return transform_->apply(block);
  }

 private:
  std::string columnName_;
  std::shared_ptr<Transform> transform_;
  std::optional<int32_t> parameter_;
};

} // namespace facebook::velox::connector::hive::iceberg
