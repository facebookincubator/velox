/*
 * Copyright (c) Meta Platforms, Inc. and its affiliates.
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

#include "velox/functions/prestosql/aggregates/VectorIndexPackAggregate.h"

#include <cmath>
#include <limits>

#include "velox/common/vectorindex/LocalVectorIndex.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

using common::vectorindex::EmbeddingNormalizer;
using common::vectorindex::Embeddings;
using common::vectorindex::LocalVectorIndex;
using common::vectorindex::LocalVectorIndexOptions;
using common::vectorindex::VectorIndexKind;
using common::vectorindex::VectorIndexMetric;
using common::vectorindex::VectorIndexMetricName;
using common::vectorindex::VectorIndexMetricValue;

class VectorIndexPackAggregate {
 public:
  using InputType = Row<
      /*cluster_id*/ int64_t,
      /*cluster_centroid*/ Array<float>,
      /*metric*/ Varchar>;

  // Carries the metric from the constant argument, which steps that merge
  // intermediate results do not receive.
  using IntermediateType = Row<
      /*metric*/ int32_t,
      /*cluster_ids*/ Array<int64_t>,
      /*centroids, one after another*/ Array<float>>;

  // Positions of the fields of 'IntermediateType'.
  static constexpr int32_t kMetric = 0;
  static constexpr int32_t kClusterIds = 1;
  static constexpr int32_t kCentroids = 2;

  using OutputType = Varbinary;

  // Reads and checks the constant 'metric' argument once, before any rows
  // arrive.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    VELOX_CHECK_EQ(constantInputs.size(), 3);
    const auto& metric = constantInputs[2];
    VELOX_USER_CHECK_NOT_NULL(metric, "Metric must be a constant");
    VELOX_USER_CHECK(!metric->isNullAt(0), "Metric must not be null");
    const auto metricName =
        metric->as<ConstantVector<StringView>>()->valueAt(0);
    constantMetric_ = VectorIndexMetricName::tryToVectorIndexMetric(metricName);
    VELOX_USER_CHECK(
        constantMetric_.has_value(),
        "Unsupported vector index metric: {}",
        metricName);
  }

  struct AccumulatorType {
    // The length of every centroid; 0 until the first one arrives.
    int32_t dimension{0};
    std::vector<int64_t, StlAllocator<int64_t>> clusterIds;
    // The centroids, normalized for the metric, one after another, in the
    // order 'clusterIds' lists them.
    std::vector<float, StlAllocator<float>> centroids;
    // The group's metric: from the constant argument for raw input, and from
    // the intermediate results otherwise.
    std::optional<VectorIndexMetric> metric;
    // The function object shared by all groups, which holds the metric from
    // the constant argument.
    VectorIndexPackAggregate* fn;

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* allocator,
        VectorIndexPackAggregate* fn)
        : clusterIds{StlAllocator<int64_t>(allocator)},
          centroids{StlAllocator<float>(allocator)},
          fn{fn} {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        int64_t clusterId,
        exec::arg_type<Array<float>> centroid,
        exec::arg_type<Varchar> /*metric*/) {
      VELOX_USER_CHECK_GE(clusterId, 0, "Cluster id must not be negative");
      if (!metric.has_value()) {
        metric = fn->constantMetric();
      }
      setDimension(centroid.size());
      for (auto value : centroid) {
        VELOX_USER_CHECK(
            value.has_value(), "Cluster centroid must not contain nulls");
        addValue(value.value());
      }
      EmbeddingNormalizer::normalize(
          groupMetric(), std::span(centroids).last(dimension));
      clusterIds.push_back(clusterId);
    }

    // Intermediate results may come from vector_index_pack_merge, so their
    // structure is checked.
    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<IntermediateType> other) {
      VELOX_USER_CHECK(
          other.at<kMetric>().has_value(),
          "Intermediate vector index codec must include the metric");
      VELOX_USER_CHECK(
          other.at<kClusterIds>().has_value(),
          "Intermediate vector index codec must include the cluster ids");
      VELOX_USER_CHECK(
          other.at<kCentroids>().has_value(),
          "Intermediate vector index codec must include the centroids");
      const auto otherMetric =
          VectorIndexMetricValue::fromValue(*other.at<kMetric>());
      if (!metric.has_value()) {
        metric = otherMetric;
      }
      VELOX_USER_CHECK(
          otherMetric == *metric,
          "Intermediate vector index codecs of a group must have the same metric");
      const auto& otherIds = *other.at<kClusterIds>();
      const auto& otherCentroids = *other.at<kCentroids>();
      VELOX_USER_CHECK_GT(
          otherIds.size(),
          0,
          "Intermediate vector index codec must include a cluster");
      VELOX_USER_CHECK_EQ(
          otherCentroids.size() % otherIds.size(),
          0,
          "Intermediate vector index codec must have the same number of values per centroid");
      setDimension(otherCentroids.size() / otherIds.size());
      for (auto id : otherIds) {
        VELOX_USER_CHECK(
            id.has_value(), "Intermediate cluster ids must not be null");
        clusterIds.push_back(id.value());
      }
      for (auto value : otherCentroids) {
        VELOX_USER_CHECK(
            value.has_value(),
            "Intermediate cluster centroids must not contain nulls");
        addValue(value.value());
      }
    }

    void writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      VELOX_USER_CHECK_LE(
          centroids.size(),
          std::numeric_limits<vector_size_t>::max(),
          "Too many centroid values in a vector index codec");
      out.get_writer_at<kMetric>() =
          VectorIndexMetricValue::toValue(groupMetric());
      auto& idsWriter = out.get_writer_at<kClusterIds>();
      for (auto id : clusterIds) {
        idsWriter.add_item() = id;
      }
      auto& centroidsWriter = out.get_writer_at<kCentroids>();
      for (auto value : centroids) {
        centroidsWriter.add_item() = value;
      }
    }

    void writeFinalResult(exec::out_type<OutputType>& out) {
      const int64_t numClusters = clusterIds.size();
      // Puts the centroids in cluster-id order in place: each swap moves one
      // centroid to its final position. Each id moves with its centroid, so
      // the accumulator still represents the same input.
      for (int64_t i = 0; i < numClusters; ++i) {
        while (clusterIds[i] != i) {
          const auto target = clusterIds[i];
          VELOX_USER_CHECK_GE(
              target,
              0,
              "Cluster ids must be in [0, number of clusters) range");
          VELOX_USER_CHECK_LT(
              target,
              numClusters,
              "Cluster ids must be in [0, number of clusters) range");
          VELOX_USER_CHECK_NE(
              clusterIds[target],
              target,
              "Cluster id must appear once: {}",
              target);
          std::swap_ranges(
              centroids.begin() + i * dimension,
              centroids.begin() + (i + 1) * dimension,
              centroids.begin() + target * dimension);
          std::swap(clusterIds[i], clusterIds[target]);
        }
      }
      out.reserve(
          centroids.size() * sizeof(float) + numClusters * sizeof(int64_t));
      LocalVectorIndex::build(
          VectorIndexKind::kFlat,
          groupMetric(),
          LocalVectorIndexOptions{},
          Embeddings(centroids, dimension),
          clusterIds,
          [&](std::string_view chunk) { out.append(chunk); });
    }

   private:
    VectorIndexMetric groupMetric() const {
      VELOX_CHECK(metric.has_value());
      return *metric;
    }

    void addValue(float value) {
      VELOX_USER_CHECK(
          std::isfinite(value),
          "Cluster centroid must not contain NaN or infinity");
      centroids.push_back(value);
    }

    void setDimension(int64_t size) {
      VELOX_USER_CHECK_GT(size, 0, "Cluster centroid must not be empty");
      if (dimension == 0) {
        dimension = size;
        return;
      }
      VELOX_USER_CHECK_EQ(
          size, dimension, "All cluster centroids must have the same length");
    }
  };

 private:
  VectorIndexMetric constantMetric() const {
    VELOX_CHECK(constantMetric_.has_value());
    return *constantMetric_;
  }

  // Set by setConstantInputs() in steps that read raw input.
  std::optional<VectorIndexMetric> constantMetric_;
};

} // namespace

void registerVectorIndexPackAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("varbinary")
          .intermediateType("row(integer,array(bigint),array(real))")
          .argumentType("bigint")
          .argumentType("array(real)")
          .constantArgumentType("varchar")
          .build(),
  };

  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& /*config*/)
          -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<
            exec::SimpleAggregateAdapter<VectorIndexPackAggregate>>(
            step, argTypes, resultType);
      },
      {.ignoreNullInputs = true},
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
