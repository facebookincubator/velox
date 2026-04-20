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

#include <memory>
#include <optional>
#include <vector>

namespace facebook::velox::cudf_velox {

// Owns a recursive hash-partition tree for operators that must buffer or
// partitioned state until all input is received.
//
// PBS only manages the partition tree and leaf row limits. It does not know
// the concrete type of state stored in a leaf. The caller supplies a
// `LeafStateOps` implementation that defines:
// - how raw input becomes an `InputChunk`,
// - what concrete `LeafState` subclass each leaf holds,
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
// 1. Define a `LeafStateOps` implementation for the operator.
// 2. Store any operator-specific state inside a `LeafState` subclass.
// 3. Construct PBS with that strategy and a max leaf row count.
// 4. Call `addInput()` for each incoming `CudfVector`.
// 5. At the end, call `drainNextOutput()` until it returns `nullptr`.
//
// `InputChunk` may hold either a borrowed `cudf::table_view` or a materialized
// table through its owning `CudfVectorPtr`. This allows callers such as final
// groupby to avoid eager materialization when a permuted view is sufficient.
class PartitionedBufferedState {
 public:
  struct PartitionSpec {
    int32_t numPartitions;
    std::vector<cudf::size_type> keyIndices;
    cudf::hash_id hashId{cudf::hash_id::HASH_MURMUR3};
    uint32_t seed{cudf::DEFAULT_HASH_SEED};
  };

  struct InputChunk {
    InputChunk() : stream(rmm::cuda_stream_default) {}

    InputChunk(
        memory::MemoryPool* pool,
        TypePtr type,
        cudf::table_view view,
        rmm::cuda_stream_view stream,
        CudfVectorPtr owner)
        : pool(pool),
          type(std::move(type)),
          view(view),
          stream(stream),
          owner(std::move(owner)) {}

    memory::MemoryPool* pool{nullptr};
    TypePtr type;
    cudf::table_view view;
    rmm::cuda_stream_view stream;
    CudfVectorPtr owner;

    size_t size() const {
      return static_cast<size_t>(view.num_rows());
    }

    bool empty() const {
      return size() == 0;
    }
  };

  class LeafState {
   public:
    virtual ~LeafState() = default;
  };

  struct Node {
    size_t leafRows{0};
    std::unique_ptr<LeafState> leafState;
    std::optional<PartitionSpec> split;
    std::vector<std::unique_ptr<Node>> children;

    bool isLeaf() const {
      return !split.has_value();
    }
  };

  class LeafStateOps {
   public:
    virtual ~LeafStateOps() = default;

    // Convert one raw input batch into a prepared chunk that PBS can route.
    virtual InputChunk prepareInput(CudfVectorPtr rawInput) = 0;

    // Return a cheap, conservative upper bound on the row count after adding
    // `input` to `leaf`.
    virtual size_t estimatedMergedRowUpperBound(
        const LeafState& leaf,
        const InputChunk& input) const = 0;

    // Create a new leaf from the first chunk routed to it.
    virtual std::unique_ptr<LeafState> createLeaf(InputChunk input) = 0;

    // Absorb one prepared chunk into an existing leaf.
    virtual void addInputToLeaf(LeafState& leaf, InputChunk input) = 0;

    // Report the logical row count PBS should track for this leaf.
    virtual size_t leafRowCount(const LeafState& leaf) const = 0;

    // Partition one prepared chunk according to an internal node's partition
    // spec and return one child chunk per partition.
    virtual std::vector<InputChunk> partitionInput(
        InputChunk input,
        const PartitionSpec& spec) = 0;

    // Split one overflowing leaf into child leaves according to `spec` and
    // return one child state per partition.
    virtual std::vector<std::unique_ptr<LeafState>> repartitionLeaf(
        std::unique_ptr<LeafState> leaf,
        const PartitionSpec& spec) = 0;

    // Finalize one leaf and return one output batch.
    virtual CudfVectorPtr finalizeLeaf(std::unique_ptr<LeafState> leaf) = 0;

    // Return partition-key indices in the prepared input schema.
    virtual const std::vector<cudf::size_type>& keyIndices() const = 0;
  };

  PartitionedBufferedState(
      std::unique_ptr<LeafStateOps> ops,
      size_t maxRowsPerLeaf,
      uint32_t initialHashSeed = cudf::DEFAULT_HASH_SEED);

  void addInput(CudfVectorPtr rawInput);

  CudfVectorPtr drainNextOutput();

  bool empty() const;

 private:
  void insert(Node& node, InputChunk bufferedInput);

  void splitLeaf(Node& node);

  void splitLeaf(Node& node, InputChunk bufferedInput);

  CudfVectorPtr drainNextOutput(Node& node);

  PartitionSpec makePartitionSpec(size_t totalRows);

  void ensureLeafWithinLimit(Node& node);

  std::vector<InputChunk> partitionInput(
      InputChunk input,
      const PartitionSpec& spec);

  std::unique_ptr<LeafStateOps> ops_;
  const size_t maxRowsPerLeaf_;
  std::unique_ptr<Node> root_;
  uint32_t nextHashSeed_;
};

} // namespace facebook::velox::cudf_velox
