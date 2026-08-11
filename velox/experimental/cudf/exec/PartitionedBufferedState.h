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
#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

struct PartitionSpec {
  int32_t numPartitions;
  std::vector<cudf::size_type> keyIndices;
  cudf::hash_id hashId{cudf::hash_id::HASH_MURMUR3};
  uint32_t seed{cudf::DEFAULT_HASH_SEED};
};

enum class InputChunkStorage : uint8_t {
  // The chunk is a view whose storage is retained by 'owner' or 'tableOwner'.
  kBorrowed,

  // 'owner' describes the complete table referenced by 'view'. The owner can
  // still be shared temporarily while routing the chunk, but a persistent leaf
  // must materialize if it is not the sole owner.
  kOwned,
};

struct InputChunk {
  InputChunk() : stream(rmm::cuda_stream_default) {}

  InputChunk(
      memory::MemoryPool* pool,
      TypePtr type,
      cudf::table_view view,
      rmm::cuda_stream_view stream,
      CudfVectorPtr owner,
      std::shared_ptr<cudf::table> tableOwner,
      InputChunkStorage storage)
      : pool(pool),
        type(std::move(type)),
        view(view),
        stream(stream),
        owner(std::move(owner)),
        tableOwner(std::move(tableOwner)),
        storage(storage) {}

  InputChunk(const InputChunk&) = delete;
  InputChunk& operator=(const InputChunk&) = delete;
  InputChunk(InputChunk&&) = default;
  InputChunk& operator=(InputChunk&&) = default;

  memory::MemoryPool* pool{nullptr};
  TypePtr type;
  cudf::table_view view;
  rmm::cuda_stream_view stream;
  CudfVectorPtr owner;
  std::shared_ptr<cudf::table> tableOwner;
  InputChunkStorage storage{InputChunkStorage::kBorrowed};

  size_t size() const {
    return static_cast<size_t>(view.num_rows());
  }

  bool empty() const {
    return size() == 0;
  }

  // Returns true when 'view' is the complete table view of 'owner'.
  bool ownsFullTable() const;

  // Consumes this chunk and returns one backed by an independently owned
  // CudfVector. Reuses an already complete, unique owner; otherwise copies the
  // exact view through 'mr'.
  InputChunk materialize(rmm::device_async_resource_ref mr) &&;
};

// Move-only host representation of an independently owned CudfVector.
//
// Device column buffers are copied independently into pinned host buffers, so
// spilling does not require a same-sized device allocation to pack the table.
// Pinned bytes are charged to 'hostPool'. spill() is transactional with
// respect to host allocation failure: 'resident' remains a valid device vector
// unless all D2H copies complete successfully.
class SpilledCudfVector {
 public:
  ~SpilledCudfVector();

  SpilledCudfVector(SpilledCudfVector&&) noexcept;
  SpilledCudfVector& operator=(SpilledCudfVector&&) noexcept;

  SpilledCudfVector(const SpilledCudfVector&) = delete;
  SpilledCudfVector& operator=(const SpilledCudfVector&) = delete;

  static SpilledCudfVector spill(
      CudfVectorPtr& resident,
      memory::MemoryPool* hostPool);

  // Restores through 'mr' without consuming the host copy. The caller may
  // release this object after restore() returns because H2D has completed.
  CudfVectorPtr restore(rmm::device_async_resource_ref mr) const;

  uint64_t deviceBytes() const;

  uint64_t hostBytes() const;

 private:
  struct Impl;

  explicit SpilledCudfVector(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

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

  // Create a new leaf from the first chunk routed to it.
  virtual std::unique_ptr<BufferedState> createLeaf(InputChunk input) = 0;

  // Create one leaf directly from two transient chunks routed to the same
  // child. Strategies should override this when they can merge the chunks
  // without first materializing either one. The default preserves the old
  // create-then-add behavior.
  virtual std::unique_ptr<BufferedState> createLeafFromInputs(
      InputChunk first,
      InputChunk second) {
    auto leaf = createLeaf(std::move(first));
    addInputToLeaf(*leaf, std::move(second));
    return leaf;
  }

  // Absorb one prepared chunk into an existing leaf.
  virtual void addInputToLeaf(BufferedState& leaf, InputChunk input) = 0;

  // Report the logical row count PBS should track for this leaf.
  virtual size_t leafRowCount(const BufferedState& leaf) const = 0;

  // Report the approximate flat size of this leaf state in bytes.
  // TODO (dm): This should be part of BufferedState
  virtual uint64_t leafFlatSize(const BufferedState& leaf) const = 0;

  // Return independently reclaimable device bytes for a resident leaf. The
  // default keeps non-spill-aware strategies out of GPU reclamation.
  virtual uint64_t leafReclaimableBytes(const BufferedState& /*leaf*/) const {
    return 0;
  }

  // Evict a resident leaf to non-device storage. Implementations must not
  // release the resident state until the replacement is complete.
  virtual void spillLeaf(BufferedState& /*leaf*/) {
    VELOX_UNSUPPORTED("Buffered state does not support spilling");
  }

  // Make a spilled leaf resident before an operation reads or mutates it.
  // The default is a no-op for strategies that never spill.
  virtual void restoreLeaf(BufferedState& /*leaf*/) {}

  // Partition one prepared chunk according to an internal node's partition
  // spec and return one child chunk per partition. The input remains valid so
  // PBS can retry with a different seed if partitioning makes no progress.
  virtual std::vector<InputChunk> partitionInput(
      const InputChunk& input,
      const PartitionSpec& spec) = 0;

  // Partition one overflowing leaf into transient child chunks according to
  // `spec`. The leaf remains valid so PBS can retry with a different seed if
  // partitioning makes no progress. After a successful attempt, PBS destroys
  // the old leaf before turning these chunks into persistent child states.
  virtual std::vector<InputChunk> partitionLeaf(
      const BufferedState& leaf,
      const PartitionSpec& spec) = 0;

  // Finalize one leaf and return one output batch.
  virtual CudfVectorPtr finalizeLeaf(std::unique_ptr<BufferedState> leaf) = 0;

  // Return partition-key indices in the prepared input schema.
  virtual const std::vector<cudf::size_type>& keyIndices() const = 0;
};

// Owns a recursive hash-partition tree for operators that must buffer grouped
// state until all input is received.
//
// PBS only manages the partition tree and leaf row limits. It does not know
// the concrete type of state stored in a leaf. The caller supplies a
// `BufferedStateOps` implementation that defines:
// - how raw input becomes an `InputChunk`,
// - what concrete `BufferedState` subclass each leaf holds,
// - how a chunk is added to a leaf,
// - how a leaf reports its logical row count,
// - how prepared input is partitioned at internal nodes,
// - how an overflowing leaf is repartitioned into child leaves,
// - and how one leaf is finalized into one output batch.
//
// Use PBS when all rows for the same grouping key must continue to route to the
// same subtree, but one in-memory leaf may grow past a safe row limit. PBS
// keeps splitting oversized leaves into more hash partitions and then drains
// one finalized leaf at a time at the end.
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
  struct Node {
    size_t leafRows{0};
    std::unique_ptr<BufferedState> leafState;
    std::optional<PartitionSpec> partitionSpec;
    std::vector<std::unique_ptr<Node>> children;

    bool isLeaf() const {
      return !partitionSpec.has_value();
    }
  };

  PartitionedBufferedState(
      std::unique_ptr<BufferedStateOps> ops,
      size_t maxRowsPerLeaf,
      uint32_t initialHashSeed = cudf::DEFAULT_HASH_SEED);

  void addInput(CudfVectorPtr rawInput);

  CudfVectorPtr drainNextOutput();

  bool empty() const;

  // Returns device bytes held by independently spillable resident leaves.
  // A leaf currently being read or mutated is deliberately excluded.
  uint64_t reclaimableBytes() const;

  // Spills largest resident leaves first until at least 'targetBytes' have
  // been selected. A zero target spills every eligible resident leaf.
  // Returns the sum of the leaves' pre-spill device sizes.
  uint64_t reclaim(uint64_t targetBytes);

 private:
  class ActiveLeafGuard {
   public:
    ActiveLeafGuard(PartitionedBufferedState& owner, Node& node);
    ~ActiveLeafGuard();

    ActiveLeafGuard(const ActiveLeafGuard&) = delete;
    ActiveLeafGuard& operator=(const ActiveLeafGuard&) = delete;

   private:
    PartitionedBufferedState& owner_;
    Node* previous_;
  };

  void insert(Node& node, InputChunk bufferedInput);

  void splitLeaf(Node& node);

  void splitLeafAndAddInput(Node& node, InputChunk bufferedInput);

  CudfVectorPtr drainNextOutput(Node& node);

  PartitionSpec makePartitionSpec(size_t totalRows);

  void ensureLeafWithinLimit(Node& node);

  std::vector<InputChunk> partitionInput(
      const InputChunk& input,
      const PartitionSpec& spec);

  void restoreLeaf(Node& node);

  void collectReclaimableLeaves(
      Node& node,
      std::vector<std::pair<Node*, uint64_t>>& leaves) const;

  std::unique_ptr<BufferedStateOps> ops_;
  const size_t maxRowsPerLeaf_;
  std::unique_ptr<Node> root_;
  uint32_t nextHashSeed_;

  // This is intentionally operator/PBS state, not thread-local state. Velox
  // task pause guarantees prevent concurrent driver mutation during external
  // reclaim; reentrant reclaim on the calling thread observes this guard.
  Node* activeLeaf_{nullptr};
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
