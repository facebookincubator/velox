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
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "velox/buffer/Buffer.h"
#include "velox/dwio/nimble/common/Buffer.h"
#include "velox/dwio/nimble/common/SchemaReader.h"
#include "velox/dwio/nimble/common/Vector.h"
#include "velox/dwio/nimble/encodings/selection/EncodingSelectionPolicy.h"
#include "velox/dwio/nimble/stats/ColumnStatistics.h"

namespace facebook::nimble {

// NOTE: we don't currently support the deduplicated stats in production
// yet.
enum class StatStreamType : uint8_t {
  VALUE_COUNT,
  NULL_COUNT,
  LOGICAL_SIZE,
  PHYSICAL_SIZE,
  INTEGRAL_MIN,
  INTEGRAL_MAX,
  FLOATING_POINT_MIN,
  FLOATING_POINT_MAX,
  STRING_MIN,
  STRING_MAX,
  DEDUPLICATED_VALUE_COUNT,
  DEDUPLICATED_LOGICAL_SIZE,
};

// The encoding like layout for each type of column stat
class VectorizedStatistic {
  using physicalType = uint64_t;

 public:
  explicit VectorizedStatistic(velox::memory::MemoryPool* pool)
      : isValid_{pool, 0} {}
  virtual ~VectorizedStatistic() = default;

  template <typename T>
  T* as() {
    static_assert(std::is_base_of_v<VectorizedStatistic, T>);
    return dynamic_cast<T*>(this);
  }

  StatStreamType type() const {
    return type_;
  }

  virtual std::string_view serialize(
      const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator,
      nimble::Buffer& buffer) = 0;

  virtual void deserializeFrom(
      std::string_view payload,
      velox::memory::MemoryPool& pool) = 0;

  static std::unique_ptr<VectorizedStatistic> create(
      StatStreamType streamType,
      velox::memory::MemoryPool* pool);

 protected:
  StatStreamType type_;
  // Not applicable to the default stat, but specific vectorized
  // stats might have invalid stats or not applicable for specific
  // (sub)columns.
  nimble::Vector<bool> isValid_;
  std::vector<velox::BufferPtr> stringBuffers_;
};

template <typename T>
class TypedVectorizedStatistic : public VectorizedStatistic {
 public:
  explicit TypedVectorizedStatistic(velox::memory::MemoryPool* pool)
      : VectorizedStatistic{pool}, values_{pool, 0} {}
  virtual ~TypedVectorizedStatistic() = default;

  void append(std::optional<T> value);

  std::optional<T> valueAt(size_t idx) const;

  std::string_view serialize(
      const EncodingSelectionPolicyCreator& encodingSelectionPolicyCreator,
      nimble::Buffer& buffer) override;

  void deserializeFrom(
      std::string_view payload,
      velox::memory::MemoryPool& pool) override;

 protected:
  nimble::Vector<T> values_;
};

using DefaultVectorizedStatistics = TypedVectorizedStatistic<uint64_t>;
using IntegralVectorizedStatistics = TypedVectorizedStatistic<int64_t>;
using FloatingPointVectorizedStatistics = TypedVectorizedStatistic<double>;
using StringVectorizedStatistics = TypedVectorizedStatistic<std::string_view>;
using DeduplicatedVectorizedStatistics = TypedVectorizedStatistic<uint64_t>;

class VectorizedFileStats {
 public:
  /// Options for building file and stripe vectorized statistics.
  struct Options {
    /// Longest string min/max, in bytes, kept in vectorized stats. Longer
    /// values are written as absent, matching the DWRF writer's default limit.
    uint32_t stringStatsLengthLimit{64};
  };

  VectorizedFileStats(
      const std::vector<ColumnStatistics*>& columnStats,
      velox::memory::MemoryPool* pool)
      : VectorizedFileStats(columnStats, pool, Options{}) {}

  VectorizedFileStats(
      const std::vector<ColumnStatistics*>& columnStats,
      velox::memory::MemoryPool* pool,
      Options options);

  std::string_view serialize(nimble::Buffer& buffer);

  static std::unique_ptr<VectorizedFileStats> deserialize(
      const std::string_view payload,
      velox::memory::MemoryPool& pool);

  std::vector<std::unique_ptr<ColumnStatistics>> toColumnStatistics(
      const velox::TypePtr& schema,
      const std::shared_ptr<const nimble::Type>& nimbleType);

 private:
  VectorizedFileStats(
      const std::vector<StatStreamType>& streamTypes,
      const std::vector<uint64_t>& streamLengths,
      std::string_view statStreams,
      velox::memory::MemoryPool& pool);

  std::vector<StatStreamType> getStatStreamTypes() const;
  std::vector<StatStreamType> collateStatStreamTypes(
      const std::vector<ColumnStatistics*>& columnStats);
  void createVectorizedStats(
      const std::vector<ColumnStatistics*>& columnStats,
      velox::memory::MemoryPool* pool);
  void addColumnStat(ColumnStatistics* stat);

  std::unique_ptr<VectorizedStatistic> deserializeVectorizedStatistic(
      StatStreamType streamType,
      std::string_view payload,
      velox::memory::MemoryPool& pool);

  // Options applied when constructing the vectorized statistics streams.
  Options options_;
  std::set<StatType> statTypes_;
  std::map<StatStreamType, std::unique_ptr<VectorizedStatistic>> statStreams_;
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_ =
      [encodingFactory = ManualEncodingSelectionPolicyFactory{}](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };
};

namespace detail {
class DecodedStatStream;
struct StatNodeInfo;
} // namespace detail

/// Like VectorizedFileStats, but decodes on first use. Thread-safe.
class LazyVectorizedFileStats {
 public:
  /// Copies 'payload' and parses only its header.
  LazyVectorizedFileStats(
      std::string_view payload,
      velox::TypePtr schema,
      velox::memory::MemoryPool& pool);

  ~LazyVectorizedFileStats();

  /// Builds the node's statistics on first call. Throws on a corrupt section.
  const ColumnStatistics* columnStatistics(uint32_t nodeId) const;

  /// Builds no ColumnStatistics.
  std::optional<uint64_t> logicalSize(uint32_t nodeId) const;

 private:
  // One slot per StatStreamType value.
  static constexpr size_t kNumStatStreamTypes =
      static_cast<size_t>(StatStreamType::DEDUPLICATED_LOGICAL_SIZE) + 1;

  // The methods below require 'mutex_'.

  // Built from 'schema_' on first call.
  const std::vector<detail::StatNodeInfo>& nodes() const;

  // Decodes on first call.
  template <typename T>
  const detail::DecodedStatStream* stream(StatStreamType type) const;

  // Decodes the stream on first use. Throws if 'index' is past the end.
  template <typename T>
  std::optional<T> valueAt(StatStreamType type, uint32_t index) const;

  // Like valueAt(), but throws if absent.
  uint64_t defaultValueAt(StatStreamType type, uint32_t nodeId) const;

  // Reads the entries at 'nodeId' and the min/max at 'node.typedIndex'.
  std::unique_ptr<ColumnStatistics> buildColumnStatistics(
      uint32_t nodeId,
      const detail::StatNodeInfo& node) const;

  // Allocates the section copy and the decoded streams.
  velox::memory::MemoryPool& pool_;
  // Gives each node its statistics kind.
  const velox::TypePtr schema_;
  // Declared before the views into it, so it outlives them.
  velox::BufferPtr payload_;
  // Indexed by StatStreamType.
  std::array<std::optional<std::string_view>, kNumStatStreamTypes>
      streamPayloads_;
  // Guards 'decodedStreams_', 'nodes_' and 'columnStats_'.
  mutable std::mutex mutex_;
  // Each decoded on first use, indexed by StatStreamType.
  mutable std::
      array<std::unique_ptr<detail::DecodedStatStream>, kNumStatStreamTypes>
          decodedStreams_;
  // Kind and typed index of each node, built on first use.
  mutable std::vector<detail::StatNodeInfo> nodes_;
  // Each built on first request, indexed by node id.
  mutable std::vector<std::unique_ptr<ColumnStatistics>> columnStats_;
};

} // namespace facebook::nimble
