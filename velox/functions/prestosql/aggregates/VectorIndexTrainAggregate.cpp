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
#include "velox/functions/prestosql/aggregates/VectorIndexTrainAggregate.h"

#include <numeric>

#include "velox/common/vectorindex/FaissLocalVectorIndex.h"
#include "velox/core/QueryConfig.h"
#include "velox/exec/Aggregate.h"
#include "velox/exec/SimpleAggregateAdapter.h"
#include "velox/expression/FunctionSignature.h"
#include "velox/functions/prestosql/aggregates/EmbeddingGroup.h"
#include "velox/functions/prestosql/aggregates/VectorIndexOptionsArgument.h"

namespace facebook::velox::aggregate::prestosql {
namespace {

using common::vectorindex::CodecOptions;
using common::vectorindex::Embeddings;
using common::vectorindex::FaissLocalVectorIndex;
using common::vectorindex::LocalVectorIndexOptions;
using common::vectorindex::VectorIndexKind;
using common::vectorindex::VectorIndexMetric;
using common::vectorindex::VectorIndexMetricName;
using common::vectorindex::VectorIndexMetricValue;

// The settings from the constant arguments.
struct Settings {
  VectorIndexMetric metric;
  int64_t numClusters;
  CodecOptions codecOptions;

  bool operator==(const Settings&) const = default;
};

class VectorIndexTrainAggregate {
 public:
  using InputType = Row<
      /*embedding*/ Array<float>,
      /*id*/ int64_t,
      /*num_clusters*/ int64_t,
      /*metric*/ Varchar,
      /*options*/ Map<Varchar, Varchar>>;

  // Carries the settings from the constant arguments, which steps that merge
  // intermediate results do not receive.
  using IntermediateType = Row<
      /*metric*/ int32_t,
      /*num_clusters*/ int64_t,
      /*max_flat_clusters*/ int64_t,
      /*ids*/ Array<int64_t>,
      /*embeddings, one after another*/ Array<float>>;

  // Positions of the fields of 'IntermediateType'.
  static constexpr int32_t kMetric = 0;
  static constexpr int32_t kNumClusters = 1;
  static constexpr int32_t kMaxFlatClusters = 2;
  static constexpr int32_t kIds = 3;
  static constexpr int32_t kEmbeddings = 4;

  using OutputType = VectorIndex;

  static constexpr bool is_reducing_ = false;

  // Reads the number of threads training runs on.
  void initialize(
      core::AggregationNode::Step /*step*/,
      const std::vector<TypePtr>& /*argTypes*/,
      const TypePtr& /*resultType*/,
      const core::QueryConfig& config) {
    numThreads_ = config.vectorIndexTrainThreads();
    VELOX_USER_CHECK_GT(
        numThreads_,
        0,
        "Session property must be positive: {}",
        core::QueryConfig::kVectorIndexTrainThreads);
  }

  // Reads and checks the constant arguments once, before any rows arrive.
  void setConstantInputs(const std::vector<VectorPtr>& constantInputs) {
    VELOX_CHECK_EQ(constantInputs.size(), 5);
    const auto& numClustersVector = constantInputs[2];
    const auto& metricVector = constantInputs[3];
    VELOX_USER_CHECK_NOT_NULL(
        numClustersVector, "Number of clusters must be a constant");
    VELOX_USER_CHECK_NOT_NULL(metricVector, "Metric must be a constant");
    VELOX_USER_CHECK(
        !numClustersVector->isNullAt(0), "Number of clusters must not be null");
    VELOX_USER_CHECK(!metricVector->isNullAt(0), "Metric must not be null");

    const auto numClusters =
        numClustersVector->as<ConstantVector<int64_t>>()->valueAt(0);
    VELOX_USER_CHECK_GT(numClusters, 0, "Number of clusters must be positive");
    const auto metricName =
        metricVector->as<ConstantVector<StringView>>()->valueAt(0);
    const auto parsedMetric =
        VectorIndexMetricName::tryToVectorIndexMetric(metricName);
    VELOX_USER_CHECK(
        parsedMetric.has_value(),
        "Unsupported vector index metric: {}",
        metricName);
    constantSettings_ = Settings{
        .metric = *parsedMetric,
        .numClusters = numClusters,
        .codecOptions = CodecOptions::parse(
            VectorIndexOptionsArgument::read(constantInputs[4])),
    };
  }

  struct AccumulatorType {
    EmbeddingGroup group;
    // The group's settings: from the constant arguments for raw input, and
    // from the intermediate results otherwise.
    std::optional<Settings> settings;
    // The function object shared by all groups, which holds the settings from
    // the constant arguments and the number of threads.
    VectorIndexTrainAggregate* function;

    static constexpr bool is_fixed_size_ = false;
    static constexpr bool use_external_memory_ = true;

    AccumulatorType() = delete;

    explicit AccumulatorType(
        HashStringAllocator* allocator,
        VectorIndexTrainAggregate* function)
        : group{allocator}, function{function} {}

    void addInput(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<Array<float>> embedding,
        int64_t id,
        int64_t /*numClusters*/,
        exec::arg_type<Varchar> /*metric*/,
        exec::arg_type<Map<Varchar, Varchar>> /*options*/) {
      if (!settings.has_value()) {
        settings = function->constantSettings();
      }
      group.add(embedding, id, groupSettings().metric);
    }

    // Intermediate results may come from vector_index_train_merge, so their
    // structure is checked.
    void combine(
        HashStringAllocator* /*allocator*/,
        exec::arg_type<IntermediateType> other) {
      VELOX_USER_CHECK(
          other.at<kMetric>().has_value(),
          "Intermediate vector index training must include the metric");
      VELOX_USER_CHECK(
          other.at<kNumClusters>().has_value(),
          "Intermediate vector index training must include the number of clusters");
      VELOX_USER_CHECK(
          other.at<kMaxFlatClusters>().has_value(),
          "Intermediate vector index training must include the codec options");
      VELOX_USER_CHECK(
          other.at<kIds>().has_value(),
          "Intermediate vector index training must include the ids");
      VELOX_USER_CHECK(
          other.at<kEmbeddings>().has_value(),
          "Intermediate vector index training must include the embeddings");
      VELOX_USER_CHECK_GT(
          *other.at<kNumClusters>(), 0, "Number of clusters must be positive");
      VELOX_USER_CHECK_GT(
          *other.at<kMaxFlatClusters>(),
          0,
          "Vector index codec option must be positive: {}",
          CodecOptions::kMaxFlatClusters);
      mergeSettings({
          .metric = VectorIndexMetricValue::fromValue(*other.at<kMetric>()),
          .numClusters = *other.at<kNumClusters>(),
          .codecOptions = {.maxFlatClusters = *other.at<kMaxFlatClusters>()},
      });
      group.addIntermediate(*other.at<kIds>(), *other.at<kEmbeddings>());
    }

    void writeIntermediateResult(exec::out_type<IntermediateType>& out) {
      const auto& groupSettings = this->groupSettings();
      out.get_writer_at<kMetric>() =
          VectorIndexMetricValue::toValue(groupSettings.metric);
      out.get_writer_at<kNumClusters>() = groupSettings.numClusters;
      out.get_writer_at<kMaxFlatClusters>() =
          groupSettings.codecOptions.maxFlatClusters;
      group.writeIntermediate(
          out.get_writer_at<kIds>(), out.get_writer_at<kEmbeddings>());
    }

    void writeFinalResult(exec::out_type<OutputType>& out) {
      const auto& finalSettings = groupSettings();
      const auto contents = group.gatherSortedById();
      const auto embeddings = contents.embeddings();
      const auto dimension = embeddings.dimension();
      const auto centroids = FaissLocalVectorIndex::trainCentroids(
          finalSettings.metric,
          finalSettings.numClusters,
          embeddings,
          finalSettings.codecOptions,
          function->numThreads_);
      std::vector<int64_t> clusterIds(centroids.size() / dimension);
      std::iota(clusterIds.begin(), clusterIds.end(), 0);
      out.reserve(
          centroids.size() * sizeof(float) +
          clusterIds.size() * sizeof(int64_t));
      FaissLocalVectorIndex::build(
          VectorIndexKind::kFlat,
          finalSettings.metric,
          LocalVectorIndexOptions{},
          Embeddings(centroids, dimension),
          clusterIds,
          [&](std::string_view chunk) { out.append(chunk); });
    }

   private:
    const Settings& groupSettings() const {
      VELOX_CHECK(settings.has_value());
      return *settings;
    }

    // Takes the settings of the first intermediate result and checks that
    // later ones match.
    void mergeSettings(const Settings& other) {
      if (!settings.has_value()) {
        settings = other;
        return;
      }
      VELOX_USER_CHECK(
          other == *settings,
          "Intermediate vector index trainings of a group must have the same settings");
    }
  };

 private:
  const Settings& constantSettings() const {
    VELOX_CHECK(constantSettings_.has_value());
    return *constantSettings_;
  }

  // Set by setConstantInputs() in steps that read raw input.
  std::optional<Settings> constantSettings_;

  // Set by initialize().
  int32_t numThreads_{0};
};

} // namespace

void registerVectorIndexTrainAggregate(
    const std::vector<std::string>& names,
    bool withCompanionFunctions,
    bool overwrite) {
  std::vector<std::shared_ptr<exec::AggregateFunctionSignature>> signatures{
      exec::AggregateFunctionSignatureBuilder()
          .returnType("vector_index")
          .intermediateType(
              "row(integer,bigint,bigint,array(bigint),array(real))")
          .argumentType("array(real)")
          .argumentType("bigint")
          .constantArgumentType("bigint")
          .constantArgumentType("varchar")
          .constantArgumentType("map(varchar,varchar)")
          .build(),
  };

  exec::registerAggregateFunction(
      names,
      std::move(signatures),
      [](core::AggregationNode::Step step,
         const std::vector<TypePtr>& argTypes,
         const TypePtr& resultType,
         const core::QueryConfig& config) -> std::unique_ptr<exec::Aggregate> {
        return std::make_unique<
            exec::SimpleAggregateAdapter<VectorIndexTrainAggregate>>(
            step, argTypes, resultType, &config);
      },
      {.ignoreNullInputs = true},
      withCompanionFunctions,
      overwrite);
}

} // namespace facebook::velox::aggregate::prestosql
