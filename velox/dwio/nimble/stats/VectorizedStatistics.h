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

#include <cstdint>
#include <map>
#include <memory>
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

namespace detail {
class LazyFileStatsReader;
} // namespace detail

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

  ~VectorizedFileStats();

  std::string_view serialize(nimble::Buffer& buffer);

  // TODO: Delete once the lazy path is fully rolled out.
  static std::unique_ptr<VectorizedFileStats> deserialize(
      const std::string_view payload,
      velox::memory::MemoryPool& pool);

  // TODO: Delete once the lazy path is fully rolled out.
  std::vector<std::unique_ptr<ColumnStatistics>> toColumnStatistics(
      const velox::TypePtr& schema,
      const std::shared_ptr<const nimble::Type>& nimbleType);

  /// Parses only the header of 'payload'. Reads each stat stream in place
  /// through an encoding view, created on first use, instead of decoding it.
  /// 'payloadHolder' keeps 'payload' valid, or is null if the caller does.
  /// The result answers only columnStatistics() and logicalSize(), which are
  /// thread-safe.
  static std::unique_ptr<VectorizedFileStats> deserializeLazily(
      std::string_view payload,
      std::shared_ptr<const void> payloadHolder,
      velox::TypePtr schema,
      velox::memory::MemoryPool& pool);

  /// Builds the node's statistics on first call. Throws on a corrupt section.
  /// Requires deserializeLazily().
  const ColumnStatistics* FOLLY_NULLABLE
  columnStatistics(uint32_t nodeId) const;

  /// Builds no ColumnStatistics. Requires deserializeLazily().
  std::optional<uint64_t> logicalSize(uint32_t nodeId) const;

 private:
  VectorizedFileStats(
      const std::vector<StatStreamType>& streamTypes,
      const std::vector<uint64_t>& streamLengths,
      std::string_view statStreams,
      velox::memory::MemoryPool& pool);

  explicit VectorizedFileStats(
      std::unique_ptr<detail::LazyFileStatsReader> lazyReader);

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

  // Returns 'lazyReader_'. Throws if this object did not come from
  // deserializeLazily().
  const detail::LazyFileStatsReader& lazyReader() const;

  // Options applied when constructing the vectorized statistics streams.
  Options options_;
  std::set<StatType> statTypes_;
  std::map<StatStreamType, std::unique_ptr<VectorizedStatistic>> statStreams_;
  EncodingSelectionPolicyCreator encodingSelectionPolicyCreator_ =
      [encodingFactory = ManualEncodingSelectionPolicyFactory{}](
          DataType dataType) -> std::unique_ptr<EncodingSelectionPolicyBase> {
    return encodingFactory.createPolicy(dataType);
  };
  // Set only by deserializeLazily(), which leaves the members above empty.
  std::unique_ptr<detail::LazyFileStatsReader> lazyReader_;
};

} // namespace facebook::nimble
