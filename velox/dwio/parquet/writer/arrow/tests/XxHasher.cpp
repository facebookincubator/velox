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

// Adapted from Apache Arrow.

#include "velox/dwio/parquet/writer/arrow/tests/XxHasher.h"

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace facebook::velox::parquet::arrow {

namespace {
template <typename T>
uint64_t xxHashHelper(T value, uint32_t seed) {
  return XXH64(reinterpret_cast<const void*>(&value), sizeof(T), seed);
}

template <typename T>
void xxHashesHelper(
    const T* values,
    uint32_t seed,
    int numValues,
    uint64_t* results) {
  for (int i = 0; i < numValues; ++i) {
    results[i] = xxHashHelper(values[i], seed);
  }
}

} // namespace

uint64_t XxHasher::hash(int32_t value) const {
  return xxHashHelper(value, kParquetBloomXxHashSeed);
}

uint64_t XxHasher::hash(int64_t value) const {
  return xxHashHelper(value, kParquetBloomXxHashSeed);
}

uint64_t XxHasher::hash(float value) const {
  return xxHashHelper(value, kParquetBloomXxHashSeed);
}

uint64_t XxHasher::hash(double value) const {
  return xxHashHelper(value, kParquetBloomXxHashSeed);
}

uint64_t XxHasher::hash(const FLBA* value, uint32_t len) const {
  return XXH64(
      reinterpret_cast<const void*>(value->ptr), len, kParquetBloomXxHashSeed);
}

uint64_t XxHasher::hash(const Int96* value) const {
  return XXH64(
      reinterpret_cast<const void*>(value->value),
      sizeof(value->value),
      kParquetBloomXxHashSeed);
}

uint64_t XxHasher::hash(const ByteArray* value) const {
  return XXH64(
      reinterpret_cast<const void*>(value->ptr),
      value->len,
      kParquetBloomXxHashSeed);
}

void XxHasher::hashes(const int32_t* values, int numValues, uint64_t* hashes)
    const {
  xxHashesHelper(values, kParquetBloomXxHashSeed, numValues, hashes);
}

void XxHasher::hashes(const int64_t* values, int numValues, uint64_t* hashes)
    const {
  xxHashesHelper(values, kParquetBloomXxHashSeed, numValues, hashes);
}

void XxHasher::hashes(const float* values, int numValues, uint64_t* hashes)
    const {
  xxHashesHelper(values, kParquetBloomXxHashSeed, numValues, hashes);
}

void XxHasher::hashes(const double* values, int numValues, uint64_t* hashes)
    const {
  xxHashesHelper(values, kParquetBloomXxHashSeed, numValues, hashes);
}

void XxHasher::hashes(const Int96* values, int numValues, uint64_t* hashes)
    const {
  for (int i = 0; i < numValues; ++i) {
    hashes[i] = XXH64(
        reinterpret_cast<const void*>(values[i].value),
        sizeof(values[i].value),
        kParquetBloomXxHashSeed);
  }
}

void XxHasher::hashes(const ByteArray* values, int numValues, uint64_t* hashes)
    const {
  for (int i = 0; i < numValues; ++i) {
    hashes[i] = XXH64(
        reinterpret_cast<const void*>(values[i].ptr),
        values[i].len,
        kParquetBloomXxHashSeed);
  }
}

void XxHasher::hashes(
    const FLBA* values,
    uint32_t typeLen,
    int numValues,
    uint64_t* hashes) const {
  for (int i = 0; i < numValues; ++i) {
    hashes[i] = XXH64(
        reinterpret_cast<const void*>(values[i].ptr),
        typeLen,
        kParquetBloomXxHashSeed);
  }
}

} // namespace facebook::velox::parquet::arrow
