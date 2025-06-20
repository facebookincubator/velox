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

#include "velox/connectors/hive/iceberg/PartitionSpec.h"
#include "velox/connectors/hive/iceberg/ColumnTransform.h"
#include "velox/connectors/hive/iceberg/IcebergDataSink.h"

namespace facebook::velox::connector::hive::iceberg {
extern std::shared_ptr<ColumnTransforms> parsePartitionTransformSpecs(
    const std::vector<IcebergPartitionSpec::Field>& fields,
    const std::shared_ptr<const IcebergPartitionSpec::Schema>& schema,
    memory::MemoryPool* pool);

IcebergPartitionSpec::IcebergPartitionSpec(
    int32_t _specId,
    const std::shared_ptr<const Schema>& _schema,
    const std::vector<Field>& _fields,
    memory::MemoryPool* pool)
    : specId(_specId), schema(_schema) {
  columnTransforms = parsePartitionTransformSpecs(_fields, _schema, pool);
}

std::vector<ColumnTransform> IcebergPartitionSpec::getColumnTransforms() const {
  return columnTransforms->getColumnTransforms();
}

} // namespace facebook::velox::connector::hive::iceberg
