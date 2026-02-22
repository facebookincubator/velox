/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

// Adapted from Apache Arrow.

#pragma once

#include <cstdint>

#include "velox/dwio/parquet/writer/arrow/Platform.h"
#include "velox/dwio/parquet/writer/arrow/Types.h"
#include "velox/dwio/parquet/writer/arrow/tests/Hasher.h"

namespace facebook::velox::parquet::arrow {

class PARQUET_EXPORT XxHasher : public Hasher {
 public:
  uint64_t hash(int32_t value) const override;
  uint64_t hash(int64_t value) const override;
  uint64_t hash(float value) const override;
  uint64_t hash(double value) const override;
  uint64_t hash(const Int96* value) const override;
  uint64_t hash(const ByteArray* value) const override;
  uint64_t hash(const FLBA* val, uint32_t len) const override;

  void hashes(const int32_t* values, int numValues, uint64_t* hashes)
      const override;
  void hashes(const int64_t* values, int numValues, uint64_t* hashes)
      const override;
  void hashes(const float* values, int numValues, uint64_t* hashes)
      const override;
  void hashes(const double* values, int numValues, uint64_t* hashes)
      const override;
  void hashes(const Int96* values, int numValues, uint64_t* hashes)
      const override;
  void hashes(const ByteArray* values, int numValues, uint64_t* hashes)
      const override;
  void hashes(
      const FLBA* values,
      uint32_t typeLen,
      int numValues,
      uint64_t* hashes) const override;

  static constexpr int kParquetBloomXxHashSeed = 0;
};

} // namespace facebook::velox::parquet::arrow
