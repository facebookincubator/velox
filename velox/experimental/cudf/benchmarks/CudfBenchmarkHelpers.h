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

#include "velox/exec/tests/utils/TpchQueryBuilder.h"
#include "velox/vector/ComplexVector.h"

namespace facebook::velox::cudf_velox {

using TableInfo = exec::test::TpchTableMetadata;

TableInfo readTableInfo(
    const std::string& tableName,
    const std::string& dataPath,
    const std::vector<std::string>& standardColumns,
    dwio::common::FileFormat format,
    memory::MemoryPool* pool);

/// Reads parquet files directly into GPU-resident CudfVectors using cudf's
/// native chunked parquet reader. No CPU intermediate.
std::vector<RowVectorPtr> readParquetIntoCudfVectors(
    const std::vector<std::string>& files,
    const RowTypePtr& outputType,
    const std::unordered_map<std::string, std::string>& fileColumnNames,
    memory::MemoryPool* pool,
    int32_t batchSizeBytes);

/// Registers the GpuValuesAdapter so ValuesNode with CudfVectors produces
/// GPU output directly (no CudfFromVelox conversion).
void registerGpuValuesAdapter();

} // namespace facebook::velox::cudf_velox
