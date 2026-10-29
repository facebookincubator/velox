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

#include "velox/common/vectorindex/VectorIndexCodec.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

using common::vectorindex::VectorIndexCodec;

class VectorIndexPackAggregate {
 public:
  using InputType = Row<
      /*cluster_id*/ int64_t,
      /*cluster_centroid*/ Array<float>,
      /*metric*/ Varchar>;

  using IntermediateType = Row<
      /*cluster_ids*/ Array<int64_t>,
      /*centroids, one after another*/ Array<float>>;

  // Positions of the fields of 'IntermediateType'.
  static constexpr int32_t kClusterIds = 0;
  static constexpr int32_t kCentroids = 1;

  using OutputType = Varbinary;

  // Checks the constant 'metric' argument once, before any rows arrive.
  // 'cosine' is the only metric, so the codec does not need to keep it.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    VELOX_CHECK_EQ(constantInputs.size(), 3);
    const auto& metric = constantInputs[2];
    VELOX_USER_CHECK_NOT_NULL(metric, "Metric must be a constant");
    VELOX_USER_CHECK(!metric->isNullAt(0), "Metric must not be null");
    VELOX_USER_CHECK_EQ(
        metric->as<ConstantVector<StringView>>()->valueAt(0),
        "cosine",
        "Unsupported vector index metric");
  }

  struct AccumulatorType {
    // The length of every centroid; 0 until the first one arrives.
    int32_t dimension{0};
    std::vector<int64_t, StlAllocator<int64_t>> clusterIds;
    // The centroids, one after another, in the order 'clusterIds' lists them.
    std::vector<float, StlAllocator<float>> centroids;

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* allocator,
        VectorIndexPackAggregate* /*fn*/)
        : clusterIds{StlAllocator<int64_t>(allocator)},
          centroids{StlAllocator<float>(allocator)} {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        int64_t clusterId,
        exec::arg_type<Array<float>> centroid,
        exec::arg_type<Varchar> /*metric*/) {
      VELOX_USER_CHECK_GE(clusterId, 0, "Cluster id must not be negative");
      setDimension(centroid.size());
      for (auto value : centroid) {
        VELOX_USER_CHECK(
            value.has_value(), "Cluster centroid must not contain nulls");
        centroids.push_back(value.value());
      }
      clusterIds.push_back(clusterId);
    }

    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<IntermediateType> other) {
      VELOX_CHECK(other.at<kClusterIds>().has_value());
      VELOX_CHECK(other.at<kCentroids>().has_value());
      const auto& otherIds = *other.at<kClusterIds>();
      const auto& otherCentroids = *other.at<kCentroids>();
      if (otherIds.size() == 0) {
        return;
      }

      VELOX_CHECK_EQ(otherCentroids.size() % otherIds.size(), 0);
      setDimension(otherCentroids.size() / otherIds.size());
      for (auto id : otherIds) {
        clusterIds.push_back(id.value());
      }
      for (auto value : otherCentroids) {
        centroids.push_back(value.value());
      }
    }

    void writeIntermediateResult(exec::out_type<IntermediateType>& out) {
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
      // centroid to its final position. Each id moves with its centroid, and
      // pack() only normalizes the centroids, so the accumulator still
      // represents the same input.
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
      out.reserve(centroids.size() * sizeof(float));
      VectorIndexCodec::pack(
          centroids.data(),
          numClusters,
          dimension,
          common::vectorindex::VectorIndexMetric::kCosine,
          [&](std::string_view chunk) { out.append(chunk); });
    }

   private:
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
};

} // namespace

void registerVectorIndexPackAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("varbinary")
          .intermediateType("row(array(bigint),array(real))")
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
