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

#include "velox/experimental/wave/common/Cuda.h"

/// Sample header for testing Cuda.h

namespace facebook::velox::wave {

constexpr uint32_t kPrime32 = 1815531889;

/// Struct for the state of a probe into MockTable. A struct of arrays. Each
/// thread fills its state at threadIdx.x so the neighbors can decide who does
/// what.
struct MockProbe {
  static constexpr int32_t kMinBlockSize = 128;
  static constexpr int32_t kMaxBlocks = 8192 / kMinBlockSize;
  // First row of blockIdx.x. Index into 'partitions' of the update kernel.
  int32_t begin[kMaxBlocks];
  // first row of next TB.
  int32_t end[kMaxBlocks];
  uint16_t failFill[kMaxBlocks];
  // The row of input that corresponds to the probe. Subscript is threadIdx.x +
  // blockDim.x * blockIdx.x.
  int32_t start[8192];
  // Whether the lane is a hit. Subscript is threadIdx.x + blockDim.x *
  // blockIdx.x.
  bool isHit[8192];
  // Whether the probe needs to cross to the next partition. Subscript is
  // threadIdx.x + blockDim.x * blockIdx.x.
  bool isOverflow[8192];

  // Temp area for gathering failed probes. A TB has blockDim.x elements
  // starting at blockIdx.x * blockDim.x.
  uint16_t failIdx[8192];
};

/// A return state record for MockTable on GPU. There is one per TB per 8K
/// batch.
struct MockStatus {
  // Number of failed rows for the TB. Fail can be a would overflow to another
  // TB's partition or not having a new row to insert. The row/partition of the
  // failed rows are compacted starting at 'firstIn8K'
  uint16_t numFailed;
  // The index in the 8K batch of 'this' for the first in the TB's partition.
  uint16_t beginIn8K;
  // The index of the first above the last in the 8K batch.
  uint16_t endIn8K;
  // The index of the first unconsumed row in the insertable row supply of the
  // TB.
  uint16_t lastConsumed;
  // Up to 5 indices into the table for groups with high skew. Host can decide
  // to deskew by replicating accumulators. -1 means no key.
  int32_t skewedKey[5];
};

/// Host side collection of pointers to device memory for updating a
/// MockTable. The layout is: MockStatus array, one element per TB
/// per 8K batch. Array of partition numbers, 8K elements for each
/// 8K batch. Array of row numbers, 8K elements per 8K batch. The
/// status array and the contiguous partition and row number arrays
/// are copied back to host as return status. The remaining fields
/// are not copied to host. Array of 64 bit hash numbers, one per
/// row of input. Array of 64 bit column values, one array of
/// numRows entries per column. Key columns are before non-key
/// columns.
struct MockTableBatch {
  MockStatus* status;
  uint16_t* partitions;
  uint16_t* rows;
  uint64_t* hashes;
  int64_t* columnData;

  // Unified memory array of pointers to start of each column.
  int64_t** columns;

  // Host side shadow of status, partitions and rows. These are copied from the
  // input status, partitions and rows after the batch is done.
  MockStatus* returnStatus;
  uint16_t* returnPartitions;
  uint16_t* returnRows;
};

/// Hash table, works in CPU and GPU.
struct MockTable {
  uint32_t sizeMask;

  // Size / 64K.
  int32_t partitionSize{0};

  // Mask to get partition base.
  uint32_t partitionMask{0};

  // Number of entries not at their first location.
  int64_t numCollisions{0};
  int64_t** rows;

  // Number of times an insert goes outside of the thread block of the hash
  // number.
  int64_t numNextBlock{0};

  // Max number of entries tried before finding a place to insert.
  int32_t maxCollisionSteps{0};

  // Number of times a partition boundary was crossed when inserting a key. One
  // insert can cross multiple boundaries.
  int64_t numNextPartition{0};

  int32_t numRows{0};

  // Size of row, includes keys and dependents, aligned to 8.
  int32_t rowSize{0};

  // Number of key and dependent columns.
  uint8_t numColumns;

  // Payload.
  char* columns{nullptr};
};

struct WideParams {
  int32_t size;
  int32_t* numbers;
  int32_t stride;
  int32_t repeat;
  char data[4000];
  void* result;
};

class TestStream : public Stream {
 public:
  static constexpr int32_t kGroupBlockSize = 256;

  // Queues a kernel to add 1 to numbers[0...size - 1]. The kernel repeats
  // 'repeat' times.
  void
  addOne(int32_t* numbers, int size, int32_t repeat = 1, int32_t width = 10240);

  void addOneWide(
      int32_t* numbers,
      int32_t size,
      int32_t repeat = 1,
      int32_t width = 10240);

  /// Increments each of 'numbers by a deterministic pseudorandom
  /// increment from 'lookup'.  If 'emptyWarps' is true, odd warps do
  /// no work but still sync with the other ones with __syncthreads().
  /// If 'emptyThreads' is true, odd lanes do no work and even lanes
  /// do their work instead.
  void addOneRandom(
      int32_t* numbers,
      const int32_t* lookup,
      int size,
      int32_t repeat = 1,
      int32_t width = 10240,
      bool emptyWarps = false,
      bool emptyLanes = false);

  static int32_t sort8KTempSize();

  // Makes random lookup keys and increments, starting at 'startCount'
  // columns[0] is keys. 'powerOfTwo' is the next power of two from
  // 'keyRange'. If 'powerOfTwo' is 0 the key columns are set to
  // zero. Otherwise the key column values are incremented by a a
  // delta + index of column where delta for element 0 is startCount &
  // (powerOfTwo - 1).
  void makeInput(
      int32_t numRows,
      int32_t keyRange,
      int32_t powerOfTwo,
      int32_t startCount,
      uint64_t* hash,
      uint8_t numColumns,
      int64_t** columns);

  /// Calculates a hash of each key, stores it in hash and then calculates a 16
  /// bit partition number. Gives each row a sequence number. Sorts by
  /// partition, so that the row numbers are in partition order in 'rows'.
  void partition8K(
      int32_t numRows,
      uint8_t shift,
      int64_t* keys,
      uint64_t* hashes,
      uint16_t* partitions,
      uint16_t* rows);

  void update8K(
      int32_t numRows,
      uint64_t* hash,
      uint16_t* partitions,
      uint16_t* rowNumbers,
      int64_t** columns,
      MockProbe* probe,
      MockStatus* mockStatus,
      MockTable* table);
};

} // namespace facebook::velox::wave
