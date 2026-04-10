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

class PartitionedBufferedState {
 public:
  struct PartitionSpec {
    int32_t numPartitions;
    std::vector<cudf::size_type> keyIndices;
    cudf::hash_id hashId{cudf::hash_id::HASH_MURMUR3};
    uint32_t seed{cudf::DEFAULT_HASH_SEED};
  };

  struct Node {
    CudfVectorPtr leafData;
    std::optional<PartitionSpec> split;
    std::vector<std::unique_ptr<Node>> children;

    bool isLeaf() const {
      return !split.has_value();
    }
  };

  class BufferedStateOps {
   public:
    virtual ~BufferedStateOps() = default;

    virtual CudfVectorPtr compactInputBatch(CudfVectorPtr rawInput) = 0;

    virtual CudfVectorPtr mergeBuffered(
        CudfVectorPtr left,
        CudfVectorPtr right) = 0;

    virtual CudfVectorPtr finalizeLeaf(CudfVectorPtr bufferedLeaf) = 0;

    virtual TypePtr bufferedType() const = 0;

    virtual TypePtr outputType() const = 0;

    virtual const std::vector<cudf::size_type>& keyIndices() const = 0;
  };

  class HashPartitioner {
   public:
    virtual ~HashPartitioner() = default;

    virtual std::vector<CudfVectorPtr> partition(
        const CudfVectorPtr& input,
        const TypePtr& tableType,
        const PartitionSpec& spec) const = 0;
  };

  PartitionedBufferedState(
      std::unique_ptr<BufferedStateOps> ops,
      size_t maxRowsPerLeaf,
      std::unique_ptr<HashPartitioner> partitioner = nullptr,
      uint32_t initialHashSeed = cudf::DEFAULT_HASH_SEED);

  void addInput(CudfVectorPtr rawInput);

  CudfVectorPtr drainNextOutput();

  bool empty() const;

 private:
  void insert(Node& node, CudfVectorPtr bufferedInput);

  void splitLeaf(Node& node, CudfVectorPtr bufferedInput);

  CudfVectorPtr drainNextOutput(Node& node);

  PartitionSpec makePartitionSpec(size_t totalRows);

  std::vector<CudfVectorPtr> partitionInput(
      const CudfVectorPtr& input,
      const PartitionSpec& spec) const;

  const std::unique_ptr<BufferedStateOps> ops_;
  std::unique_ptr<HashPartitioner> partitioner_;
  const size_t maxRowsPerLeaf_;
  std::unique_ptr<Node> root_;
  uint32_t nextHashSeed_;
};

} // namespace facebook::velox::cudf_velox
