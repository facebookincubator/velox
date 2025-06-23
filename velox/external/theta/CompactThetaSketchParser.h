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

// Adapted from Apache DataSketches
#pragma once

#include <string>

namespace facebook::velox::common::theta {

template <typename T>
T wholeBytesToHoldBits(T bits) {
  static_assert(std::is_integral<T>::value, "integral type expected");
  return (bits >> 3) + ((bits & 7) > 0);
}

template <bool dummy>
class CompactThetaSketchParser {
 public:
  struct CompactThetaSketchData {
    bool isEmpty;
    bool isOrdered;
    uint16_t seedHash;
    uint32_t numEntries;
    uint64_t theta;
    const void* entriesStartPtr;
    uint8_t entryBits;
  };

  static CompactThetaSketchData parse(
      const void* ptr,
      size_t size,
      uint64_t seed,
      bool dump_on_error = false);

 private:
  // offsets are in sizeof(type)
  static const size_t COMPACT_SKETCH_PRE_LONGS_BYTE = 0;
  static const size_t COMPACT_SKETCH_SERIAL_VERSION_BYTE = 1;
  static const size_t COMPACT_SKETCH_TYPE_BYTE = 2;
  static const size_t COMPACT_SKETCH_FLAGS_BYTE = 5;
  static const size_t COMPACT_SKETCH_SEED_HASH_U16 = 3;
  static const size_t COMPACT_SKETCH_SINGLE_ENTRY_U64 = 1; // ver 3
  static const size_t COMPACT_SKETCH_NUM_ENTRIES_U32 = 2; // ver 1-3
  static const size_t COMPACT_SKETCH_ENTRIES_EXACT_U64 = 2; // ver 1-3
  static const size_t COMPACT_SKETCH_ENTRIES_ESTIMATION_U64 = 3; // ver 1-3
  static const size_t COMPACT_SKETCH_THETA_U64 = 2; // ver 1-3
  static const size_t COMPACT_SKETCH_V4_ENTRY_BITS_BYTE = 3;
  static const size_t COMPACT_SKETCH_V4_NUM_ENTRIES_BYTES_BYTE = 4;
  static const size_t COMPACT_SKETCH_V4_THETA_U64 = 1;
  static const size_t COMPACT_SKETCH_V4_PACKED_DATA_EXACT_BYTE = 8;
  static const size_t COMPACT_SKETCH_V4_PACKED_DATA_ESTIMATION_BYTE = 16;

  static const uint8_t COMPACT_SKETCH_IS_EMPTY_FLAG = 2;
  static const uint8_t COMPACT_SKETCH_IS_ORDERED_FLAG = 4;

  static const uint8_t COMPACT_SKETCH_TYPE = 3;

  static void checkMemorySize(
      const void* ptr,
      size_t actual_bytes,
      size_t expected_bytes,
      bool dump_on_error);
  static std::string hexDump(const uint8_t* ptr, size_t size);
};

} // namespace facebook::velox::common::theta

#include "CompactThetaSketchParser.cpp"
