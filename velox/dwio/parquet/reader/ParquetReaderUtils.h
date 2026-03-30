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

#include "velox/type/Type.h"

// Forward declarations.
namespace facebook::velox::dwio::common {
struct ColumnReaderStatistics;
class SeekableInputStream;
} // namespace facebook::velox::dwio::common

namespace facebook::velox::parquet {

struct ParquetDictionaryReadContext {
  const char* pageData{nullptr};
  dwio::common::SeekableInputStream* inputStream{nullptr};
  const char** bufferStart{nullptr};
  const char** bufferEnd{nullptr};
  dwio::common::ColumnReaderStatistics* stats{nullptr};
};

/// Parquet stores integers of 8, 16, 32 bits with 32 bits and 64 as 64 bits.
inline int32_t parquetSizeOfIntKind(TypeKind kind) {
  switch (kind) {
    case TypeKind::TINYINT:
    case TypeKind::SMALLINT:
    case TypeKind::INTEGER:
      return 4;
    case TypeKind::BIGINT:
      return 8;
    case TypeKind::HUGEINT:
      return 16;
    default:
      VELOX_FAIL("Not an integer TypeKind");
  }
}

} // namespace facebook::velox::parquet
