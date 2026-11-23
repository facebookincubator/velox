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

#include "velox/experimental/cudf/vector/CudfVector.h"

#include <cudf/hashing.hpp>
#include <cudf/partitioning.hpp>

#include <deque>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

namespace facebook::velox::cudf_velox {

// Describes how rows are routed to targets. A hash of the keys puts every row
// in one of 'numBuckets' buckets, the way cudf::hash_partition() assigns rows
// to that many partitions, and target i takes the buckets
// [bucketBounds[i], bucketBounds[i + 1]).
//
// Every routing of one PartitionedBufferedState uses the same keys, seed and
// bucket count, so a row is in the same bucket whenever it is routed. Only the
// bucket bounds differ from one routing to the next.
struct HashRouting {
  std::vector<cudf::size_type> keyIndices;
  uint32_t seed{cudf::DEFAULT_HASH_SEED};
  int32_t numBuckets{1};
  // Ascending bucket numbers, one more than there are targets. The routed
  // rows must all be in the buckets [bucketBounds.front(),
  // bucketBounds.back()).
  std::vector<int32_t> bucketBounds;

  int32_t numTargets() const {
    return static_cast<int32_t>(bucketBounds.size()) - 1;
  }
};

// Reorders 'input' so that rows with the same routing target are contiguous.
// Returns the reordered table and routing.numTargets() + 1 offsets: target i
// holds rows [offsets[i], offsets[i + 1]). Costs one cudf::hash_partition()
// of the table, whatever the number of targets.
[[nodiscard]] std::
    pair<std::unique_ptr<cudf::table>, std::vector<cudf::size_type>>
    partitionByHashRouting(
        cudf::table_view input,
        const HashRouting& routing,
        cuda::stream_ref stream,
        rmm::device_async_resource_ref mr);

struct InputChunk {
  InputChunk() : stream(cudaStream_t{cudaStreamDefault}) {}

  InputChunk(
      memory::MemoryPool* pool,
      TypePtr type,
      cudf::table_view view,
      cuda::stream_ref stream,
      CudfVectorPtr owner,
      std::shared_ptr<cudf::table> tableOwner = nullptr,
      uint64_t flatSize = 0)
      : pool(pool),
        type(std::move(type)),
        view(view),
        stream(stream),
        owner(std::move(owner)),
        tableOwner(std::move(tableOwner)),
        flatSize(flatSize) {}

  memory::MemoryPool* pool{nullptr};
  TypePtr type;
  cudf::table_view view;
  cuda::stream_ref stream;
  CudfVectorPtr owner;
  std::shared_ptr<cudf::table> tableOwner;
  // Approximate size of the rows in bytes. Set by whoever makes the chunk,
  // and zero where nothing uses it.
  uint64_t flatSize{0};

  size_t size() const {
    return static_cast<size_t>(view.num_rows());
  }

  bool empty() const {
    return size() == 0;
  }
};

// Splits 'input' into one chunk per routing target. The chunks are slices of
// one table allocated from 'mr', which they share, and each chunk's flat size
// is its share of that table by rows. A target that gets no rows gets an empty
// chunk.
[[nodiscard]] std::vector<InputChunk> routeChunkByHash(
    const InputChunk& input,
    const HashRouting& routing,
    rmm::device_async_resource_ref mr);

// Serve as the opaque base type for strategy-owned leaf state.
class BufferedState {
 public:
  virtual ~BufferedState() = default;
};

class BufferedStateOps {
 public:
  virtual ~BufferedStateOps() = default;

  // Convert one raw input batch into a prepared chunk that PBS can route.
  virtual InputChunk prepareInput(CudfVectorPtr rawInput) = 0;

  // Return a cheap, conservative upper bound on the row count after adding
  // `input` to `leaf`.
  virtual size_t estimatedMergedRowUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const = 0;

  // Return a cheap, conservative upper bound on leafFlatSize() after adding
  // `input` to `leaf`. Only called when PBS has a byte limit.
  virtual uint64_t estimatedMergedBytesUpperBound(
      const BufferedState& leaf,
      const InputChunk& input) const = 0;

  // Create a new leaf from the first chunk routed to it.
  virtual std::unique_ptr<BufferedState> createLeaf(InputChunk input) = 0;

  // Absorb one prepared chunk into an existing leaf.
  virtual void addInputToLeaf(BufferedState& leaf, InputChunk input) = 0;

  // Report the logical row count PBS should track for this leaf.
  virtual size_t leafRowCount(const BufferedState& leaf) const = 0;

  // Report the approximate flat size of this leaf state in bytes.
  // TODO (dm): This should be part of BufferedState
  virtual uint64_t leafFlatSize(const BufferedState& leaf) const = 0;

  // Split one prepared chunk into one chunk per routing target. The input
  // remains valid.
  virtual std::vector<InputChunk> routeInput(
      const InputChunk& input,
      const HashRouting& routing) = 0;

  // Split one leaf into one child state per routing target. The leaf is
  // consumed, so that its memory can be released as soon as its rows have been
  // moved.
  virtual std::vector<std::unique_ptr<BufferedState>> repartitionLeaf(
      std::unique_ptr<BufferedState> leaf,
      const HashRouting& routing) = 0;

  // Finalize one leaf and return one output batch.
  virtual CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) = 0;

  // Return partition-key indices in the prepared input schema.
  virtual const std::vector<cudf::size_type>& keyIndices() const = 0;
};

// Owns hash-partitioned leaves for operators that must buffer grouped state
// until all input is received.
//
// PBS only manages the leaves and their row limits. It does not know the
// concrete type of state stored in a leaf. The caller supplies a
// `BufferedStateOps` implementation that defines:
// - how raw input becomes an `InputChunk`,
// - what concrete `BufferedState` subclass each leaf holds,
// - how a chunk is added to a leaf,
// - how a leaf reports its logical row count,
// - how prepared input is routed to the leaves,
// - how an overflowing leaf is repartitioned into child leaves,
// - and how one leaf is finalized into one output batch.
//
// Use PBS when all rows for the same grouping key must reach the same leaf,
// but one in-memory leaf may grow past a safe row limit.
//
// A hash of the grouping keys puts every row in one of a fixed number of
// buckets. A leaf holds a contiguous range of buckets, and the leaves together
// cover all of them. PBS starts with one leaf that holds every bucket and
// needs no routing. A leaf that overflows is replaced by leaves for parts of
// its bucket range; no other leaf is touched. Since a row's bucket never
// changes, an incoming batch is hash partitioned once, straight to its leaves,
// however often they have been split.
//
// A leaf is split when it would pass the row limit or the byte limit. The
// bucket count bounds the number of leaves, and a leaf that holds a single
// bucket cannot be split. Such a leaf keeps growing past both limits, up to
// the hard row limit. The byte limit is a target only: sizes are estimates,
// and passing it never fails.
//
// Typical usage:
// 1. Define a `BufferedStateOps` implementation for the operator.
// 2. Store any operator-specific state inside a `BufferedState` subclass.
// 3. Construct PBS with that strategy and a max leaf row count.
// 4. Call `addInput()` for each incoming `CudfVector`.
// 5. At the end, call `drainNextOutput()` until it returns `nullptr`.
//
// `InputChunk` may hold either a borrowed `cudf::table_view` or a materialized
// table through its owning `CudfVectorPtr`. This allows callers such as final
// groupby to avoid eager materialization when a permuted view is sufficient.
class PartitionedBufferedState {
 public:
  // Seed of the routing hash. Differs from the seeds of the exchanges that may
  // have partitioned the input by the same keys, so that their partitioning
  // says nothing about the buckets used here.
  static constexpr uint32_t kDefaultHashSeed =
      cudf::DEFAULT_HASH_SEED ^ 0xc2b2ae35U;

  // cudf::hash_partition() has a fast path for up to this many partitions.
  static constexpr int32_t kDefaultNumBuckets = 1024;

  static constexpr uint64_t kNoByteLimit = std::numeric_limits<uint64_t>::max();

  struct Options {
    // A leaf that would hold more rows than this is split.
    size_t maxRowsPerLeaf{0};
    // The most rows that a leaf can hold. Only a leaf that cannot be split
    // grows beyond maxRowsPerLeaf.
    size_t hardMaxRowsPerLeaf{std::numeric_limits<cudf::size_type>::max()};
    // A leaf whose flat size would exceed this many bytes is split.
    uint64_t maxBytesPerLeaf{kNoByteLimit};
    int32_t numBuckets{kDefaultNumBuckets};
    uint32_t hashSeed{kDefaultHashSeed};
  };

  PartitionedBufferedState(
      std::unique_ptr<BufferedStateOps> ops,
      const Options& options);

  void addInput(CudfVectorPtr rawInput);

  CudfVectorPtr drainNextOutput();

  bool empty() const;

  // Number of leaves, including the ones that hold no rows.
  size_t numLeaves() const {
    return leaves_.size();
  }

 private:
  struct Leaf {
    // The leaf holds the rows of the buckets [bucketBegin, bucketEnd).
    int32_t bucketBegin{0};
    int32_t bucketEnd{0};
    size_t rows{0};
    std::unique_ptr<BufferedState> state;
  };

  void insert(size_t leafIndex, InputChunk bufferedInput);

  void splitLeafAndAddInput(size_t leafIndex, InputChunk bufferedInput);

  void ensureLeafWithinLimit(size_t leafIndex);

  bool hasByteLimit() const {
    return options_.maxBytesPerLeaf != kNoByteLimit;
  }

  // Rebuilds the bucket bounds of the leaves after a split.
  void updateRouting();

  std::unique_ptr<BufferedStateOps> ops_;
  const Options options_;
  // In bucket order.
  std::vector<Leaf> leaves_;
  // Routes an incoming batch to 'leaves_'.
  HashRouting routing_;
  bool routingStale_{false};
  size_t nextLeafToDrain_{0};
};

// Owns one active leaf for operators that may flush early instead of
// repartitioning on overflow.
//
// Use this manager for partial and intermediate aggregation steps. It reuses
// the same `BufferedStateOps` contract as PBS, but emits a leaf when it becomes
// full instead of splitting it into child partitions.
class FlushableBufferedState {
 public:
  FlushableBufferedState(
      std::unique_ptr<BufferedStateOps> ops,
      size_t flushRowLimit,
      uint64_t flushByteLimit);

  void addInput(CudfVectorPtr rawInput);

  CudfVectorPtr getOutput(bool noMoreInput);

  bool empty() const;

 private:
  bool shouldFlushActiveLeaf() const;

  CudfVectorPtr popPendingOutput();

  void finalizeActiveLeaf();

  std::unique_ptr<BufferedStateOps> ops_;
  const size_t flushRowLimit_;
  const uint64_t flushByteLimit_;
  size_t currentLeafRows_{0};
  std::unique_ptr<BufferedState> currentLeaf_;
  std::deque<CudfVectorPtr> pendingOutputs_;
};

} // namespace facebook::velox::cudf_velox
