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

// Adapted from Apache Arrow:
// https://github.com/apache/arrow/blob/apache-arrow-15.0.0/cpp/src/arrow/util/spaced.h
// Copyright 2016-2024 The Apache Software Foundation
//
// The original file has the following license header:
//
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

#pragma once

#include <cassert>
#include <cstdint>
#include <cstring>

#include <arrow/util/bit_run_reader.h>

namespace facebook::velox::parquet::arrow::util::internal {

/// Copies the values of 'src' whose bits are set in 'validBits' to 'output'
/// and returns the number of copied values. 'numValues' is the number of
/// values in 'src'. 'validBitsOffset' is the offset of the first bit in
/// 'validBits'.
template <typename T>
inline int spacedCompress(
    const T* src,
    int numValues,
    const uint8_t* validBits,
    int64_t validBitsOffset,
    T* output) {
  int numValidValues = 0;

  ::arrow::internal::SetBitRunReader reader(
      validBits, validBitsOffset, numValues);
  while (true) {
    const auto run = reader.NextRun();
    if (run.length == 0) {
      break;
    }
    std::memcpy(
        output + numValidValues, src + run.position, run.length * sizeof(T));
    numValidValues += static_cast<int32_t>(run.length);
  }

  return numValidValues;
}

/// Moves the values densely packed at the front of 'buffer' to the positions
/// whose bits are set in 'validBits' and zero-fills the other positions.
/// Returns 'numValues'. 'numValues' is the number of values including nulls.
/// 'nullCount' is the number of nulls. 'validBitsOffset' is the offset of the
/// first bit in 'validBits'.
template <typename T>
inline int spacedExpand(
    T* buffer,
    int numValues,
    int nullCount,
    const uint8_t* validBits,
    int64_t validBitsOffset) {
  // Point to end as we add the spacing from the back.
  int decodeIndex = numValues - nullCount;

  // Depending on the number of nulls, some of the value slots in buffer may
  // be uninitialized, and this will cause valgrind warnings / potentially UB.
  std::memset(
      static_cast<void*>(buffer + decodeIndex), 0, nullCount * sizeof(T));
  if (decodeIndex == 0) {
    // All nulls, nothing more to do.
    return numValues;
  }

  ::arrow::internal::ReverseSetBitRunReader reader(
      validBits, validBitsOffset, numValues);
  while (true) {
    const auto run = reader.NextRun();
    if (run.length == 0) {
      break;
    }
    decodeIndex -= static_cast<int32_t>(run.length);
    assert(decodeIndex >= 0);
    std::memmove(
        buffer + run.position, buffer + decodeIndex, run.length * sizeof(T));
  }

  // Otherwise caller gave an incorrect nullCount.
  assert(decodeIndex == 0);
  return numValues;
}

} // namespace facebook::velox::parquet::arrow::util::internal
