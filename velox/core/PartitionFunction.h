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

#include "velox/vector/ComplexVector.h"

namespace facebook::velox::core {

/// Calculates partition number for each row of the specified vector.
class PartitionFunction {
 public:
  virtual ~PartitionFunction() = default;

  /// @param input RowVector to split into partitions.
  /// @param [out] partitions Computed partition numbers for each row in
  /// 'input'.
  /// @return Returns partition number in case all rows of 'input' are
  /// assigned to the same partition. In this case 'partitions' vector is left
  /// unchanged. Used to optimize round-robin partitioning in local exchange.
  virtual std::optional<uint32_t> partition(
      const RowVector& input,
      std::vector<uint32_t>& partitions) = 0;
};

/// Factory class for creating PartitionFunction instances.
class PartitionFunctionSpec : public ISerializable {
 public:
  /// If 'localExchange' is true, the partition function is used for local
  /// exchange within a velox task.
  virtual std::unique_ptr<PartitionFunction> create(
      int numPartitions,
      bool localExchange = false) const = 0;

  virtual ~PartitionFunctionSpec() = default;

  virtual std::string toString() const = 0;
};

using PartitionFunctionSpecPtr = std::shared_ptr<const PartitionFunctionSpec>;

class GatherPartitionFunctionSpec : public PartitionFunctionSpec {
 public:
  std::unique_ptr<PartitionFunction> create(
      int /*numPartitions*/,
      bool /*localExchange*/) const override {
    VELOX_UNREACHABLE();
  }

  std::string toString() const override {
    return "gather";
  }

  folly::dynamic serialize() const override {
    folly::dynamic obj = folly::dynamic::object;
    obj["name"] = "GatherPartitionFunctionSpec";
    return obj;
  }

  static PartitionFunctionSpecPtr deserialize(
      const folly::dynamic& /* obj */,
      void* /* context */) {
    return std::make_shared<GatherPartitionFunctionSpec>();
  }
};

} // namespace facebook::velox::core
